"""
LLM Prompt-Strategy Benchmark
Models  : Qwen2.5-3B + Llama-3.2-3B (default); optionally Qwen2.5-1.5B
Dataset : MMLU 10-subject subset
Strategies tested:
  1. zero_shot  — plain question → letter (current baseline)
  2. role       — subject-expert role prefix
  3. few_shot   — 3 in-context examples per subject (first 3 samples used as demos)
  4. cot        — "think step by step" + parse "Answer: X" from output

Fixed: n_ctx=2048, temperature=0.0
Output : outputs/tables/llm_prompt_strategies.csv
         outputs/tables/llm_prompt_examples.json

Usage:
  # Default: Qwen2.5-3B + Llama-3.2-3B, strategies: zero_shot / role / few_shot
  python scripts/benchmark_llm_prompt_strategies.py

  # Also include Qwen2.5-1.5B
  python scripts/benchmark_llm_prompt_strategies.py --all-models

  # CoT only (slow, ~18x per question; runs ONLY cot, skips zero_shot/role/few_shot)
  python scripts/benchmark_llm_prompt_strategies.py --cot

  # CoT only + all models (including Qwen2.5-1.5B)
  python scripts/benchmark_llm_prompt_strategies.py --all-models --cot
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime

# ── CUDA DLL fix for llama-cpp-python on Windows ──────────────────────────────
for _dll_dir in [
    r"D:\Softeware\miniconda\envs\llm\Library\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
]:
    if os.path.isdir(_dll_dir):
        os.add_dll_directory(_dll_dir)

import pandas as pd
from datasets import load_from_disk

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import LLM1_PATH, LLM1_5B_PATH, LLM2_PATH, MMLU_BENCH_DIR, MMLU_SUBJECTS, OUT_TABLES

try:
    import torch
    _CUDA        = torch.cuda.is_available()
    N_GPU_LAYERS = -1 if _CUDA else 0
except ImportError:
    _CUDA        = False
    N_GPU_LAYERS = -1

# ── Constants ─────────────────────────────────────────────────────────────────
CHOICES  = ["A", "B", "C", "D"]
N_CTX    = 2048
FEW_SHOT_N  = 3         # number of in-context examples per subject
COT_MAX_PER_SUBJ = 50  # max questions per subject for CoT (speed limit)

SUBJECT_NAMES = {
    "college_computer_science": "computer science",
    "college_physics":          "physics",
    "high_school_world_history": "world history",
    "high_school_biology":      "biology",
    "professional_medicine":    "medicine",
}

MODELS_DEFAULT = [
    ("Qwen2.5-3B",   str(LLM1_PATH)),
    ("Llama-3.2-3B", str(LLM2_PATH)),
]
MODELS_ALL = [
    ("Qwen2.5-1.5B", str(LLM1_5B_PATH)),
] + MODELS_DEFAULT


# ── Prompt builders ───────────────────────────────────────────────────────────

def _mc_block(question: str, choices: list, answer_letter: str | None = None) -> str:
    """Single multiple-choice block. If answer_letter given, append it as the answer."""
    block = f"Question: {question}\n"
    for i, c in enumerate(choices):
        block += f"{CHOICES[i]}. {c}\n"
    if answer_letter is not None:
        block += f"Answer: {answer_letter}\n"
    return block


def build_zero_shot(question: str, choices: list, **_) -> str:
    return _mc_block(question, choices) + "Answer with just the letter (A, B, C, or D): "


def build_role(question: str, choices: list, subject: str, **_) -> str:
    expert = SUBJECT_NAMES.get(subject, subject.replace("_", " "))
    prefix = f"You are an expert in {expert}. Answer the following multiple-choice question.\n\n"
    return prefix + _mc_block(question, choices) + "Answer with just the letter (A, B, C, or D): "


def build_few_shot(question: str, choices: list, demos: list, **_) -> str:
    """demos: list of (question, choices, answer_idx) tuples."""
    prompt = "Here are some example questions with answers:\n\n"
    for q, ch, ans_idx in demos:
        prompt += _mc_block(q, ch, CHOICES[ans_idx])
        prompt += "\n"
    prompt += "Now answer the following question.\n\n"
    prompt += _mc_block(question, choices)
    prompt += "Answer with just the letter (A, B, C, or D): "
    return prompt


# ── Few-shot CoT demo examples (2 per subject, hand-crafted reasoning chains) ─
# Each entry: (question, [A, B, C, D], reasoning, answer_letter)
COT_DEMOS: dict[str, list] = {
    "high_school_mathematics": [
        (
            "If f(x) = 2x² - 3x + 1, what is f(2)?",
            ["1", "3", "5", "7"],
            "Substitute x = 2: f(2) = 2(2²) - 3(2) + 1 = 2(4) - 6 + 1 = 8 - 6 + 1 = 3.",
            "B",
        ),
        (
            "A bag contains 3 red and 2 blue marbles. What is the probability of drawing a red marble?",
            ["2/5", "3/5", "1/2", "2/3"],
            "Total marbles = 3 + 2 = 5. Red marbles = 3. Probability = 3/5.",
            "B",
        ),
    ],
    "high_school_physics": [
        (
            "An object is dropped from rest. After 2 seconds, what is its speed? (g = 10 m/s²)",
            ["5 m/s", "10 m/s", "20 m/s", "40 m/s"],
            "Using v = u + at with u = 0, a = 10 m/s², t = 2 s: v = 0 + 10 × 2 = 20 m/s.",
            "C",
        ),
        (
            "A 10 kg object experiences a net force of 30 N. What is its acceleration?",
            ["0.3 m/s²", "3 m/s²", "30 m/s²", "300 m/s²"],
            "Newton's second law: a = F/m = 30 N / 10 kg = 3 m/s².",
            "B",
        ),
    ],
    "high_school_biology": [
        (
            "Which organelle is the primary site of ATP production in eukaryotic cells?",
            ["Nucleus", "Ribosome", "Mitochondrion", "Golgi apparatus"],
            "The mitochondrion performs cellular respiration (oxidative phosphorylation) to produce the majority of ATP in eukaryotic cells.",
            "C",
        ),
        (
            "In a cross between two Aa individuals, what fraction of offspring will be homozygous recessive?",
            ["1/4", "1/2", "3/4", "0"],
            "Aa × Aa produces AA : Aa : aa = 1 : 2 : 1. The homozygous recessive (aa) fraction is 1/4.",
            "A",
        ),
    ],
    "high_school_world_history": [
        (
            "Which dynasty established the Silk Road trade network around 130 BCE?",
            ["Roman Empire", "Han Dynasty", "Ottoman Empire", "Mongol Empire"],
            "The Han Dynasty of China opened overland trade routes westward through Central Asia around 130 BCE, forming the basis of what became known as the Silk Road.",
            "B",
        ),
        (
            "The Industrial Revolution originated in which country during the 18th century?",
            ["France", "Germany", "Great Britain", "United States"],
            "The Industrial Revolution began in Great Britain in the mid-18th century, driven by innovations in steam power and textile machinery.",
            "C",
        ),
    ],
    "high_school_us_history": [
        (
            "The Declaration of Independence was primarily drafted by which Founding Father?",
            ["George Washington", "Benjamin Franklin", "John Adams", "Thomas Jefferson"],
            "The Continental Congress appointed a committee in 1776 to draft the Declaration; Thomas Jefferson was chosen as the principal author.",
            "D",
        ),
        (
            "Which constitutional amendment abolished slavery in the United States?",
            ["13th", "14th", "15th", "16th"],
            "The 13th Amendment, ratified in December 1865, formally abolished slavery and involuntary servitude throughout the United States.",
            "A",
        ),
    ],
    "college_mathematics": [
        (
            "What is the derivative of f(x) = x³ - 4x² + 2x - 1?",
            ["3x² - 4x + 2", "3x² - 8x + 2", "x² - 8x + 2", "3x² - 8x + 1"],
            "Applying the power rule term by term: d/dx(x³) = 3x², d/dx(-4x²) = -8x, d/dx(2x) = 2, d/dx(-1) = 0. So f'(x) = 3x² - 8x + 2.",
            "B",
        ),
        (
            "What is the determinant of the 2×2 matrix [[2, 1], [3, 4]]?",
            ["5", "8", "11", "14"],
            "det = (2)(4) - (1)(3) = 8 - 3 = 5.",
            "A",
        ),
    ],
    "college_physics": [
        (
            "A parallel-plate capacitor has plate area A and separation d. If d is doubled while A is unchanged, the capacitance:",
            ["Doubles", "Halves", "Quadruples", "Remains unchanged"],
            "Capacitance C = ε₀A/d. Doubling d gives C' = ε₀A/(2d) = C/2, so capacitance halves.",
            "B",
        ),
        (
            "Which phenomenon explains the bending of light waves around obstacles?",
            ["Reflection", "Refraction", "Diffraction", "Interference"],
            "Diffraction is the spreading of waves around the edges of obstacles or through openings. It explains why light bends around corners.",
            "C",
        ),
    ],
    "college_biology": [
        (
            "Which enzyme unwinds the DNA double helix at the replication fork?",
            ["DNA polymerase", "Helicase", "Ligase", "Primase"],
            "Helicase breaks the hydrogen bonds between complementary base pairs to unwind and separate the two DNA strands, creating the replication fork.",
            "B",
        ),
        (
            "Organisms that convert solar energy into chemical energy via photosynthesis are called:",
            ["Consumers", "Decomposers", "Producers", "Carnivores"],
            "Producers (plants, algae, and some bacteria) use photosynthesis to fix solar energy into organic molecules, forming the base of food webs.",
            "C",
        ),
    ],
    "philosophy": [
        (
            "Kant's categorical imperative requires that one act only on maxims that:",
            ["Maximize overall happiness", "Could be universalized as a law for all rational beings", "Follow natural law", "Benefit oneself most"],
            "Kant's categorical imperative states: act only according to that maxim by which you can at the same time will that it should become a universal law. This is a test of whether the principle of your action could apply to everyone.",
            "B",
        ),
        (
            "Descartes' cogito argument concludes that which of the following cannot be doubted?",
            ["The existence of God", "The external world", "The existence of one's own thinking self", "The truth of mathematics"],
            "Descartes found he could doubt the external world, God, and even mathematical truths under a hypothesis of an evil deceiver. However, the very act of doubting proves that a thinking subject exists: cogito, ergo sum.",
            "C",
        ),
    ],
    "international_law": [
        (
            "The principle 'pacta sunt servanda' in international law means that:",
            ["States may withdraw from any treaty at will", "Treaties must be performed in good faith", "Only powerful states need follow treaties", "Treaties expire after 10 years"],
            "'Pacta sunt servanda' (Latin: agreements must be kept) is a foundational principle codified in the Vienna Convention on the Law of Treaties, requiring parties to honor their treaty obligations in good faith.",
            "B",
        ),
        (
            "The International Court of Justice is the principal judicial organ of which organization?",
            ["NATO", "World Trade Organization", "United Nations", "European Union"],
            "The ICJ was established by the UN Charter in 1945 as the principal judicial organ of the United Nations, settling legal disputes between states and giving advisory opinions.",
            "C",
        ),
    ],
}


def build_cot(question: str, choices: list, subject: str = "", **_) -> str:
    """Few-shot CoT: 2 hand-crafted reasoning examples per subject + target question."""
    demos = COT_DEMOS.get(subject, [])
    prompt = "Here are some example questions solved step by step:\n\n"
    for q, ch, reasoning, ans in demos:
        prompt += _mc_block(q, ch)
        prompt += f"Let me think step by step.\n{reasoning}\nAnswer: {ans}\n\n"
    prompt += "Now answer the following question using step-by-step reasoning.\n\n"
    prompt += _mc_block(question, choices)
    prompt += 'Let me think step by step.\n'
    return prompt


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer_short(text: str) -> str:
    """For zero-shot / role / few-shot (max_tokens=8).
    Uses word-boundary regex to avoid substring false positives
    (e.g. 'A' in 'ANSWER', 'B' in 'BECAUSE').
    """
    text = text.strip().upper()
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    # last-resort: first char if it is a valid choice
    return text[0] if text and text[0] in "ABCD" else "X"


def extract_answer_cot(text: str) -> str:
    """Parse 'Answer: X' from CoT output.
    Fallback 1: scan text after the last 'answer' keyword for A/B/C/D.
    Fallback 2: last standalone A/B/C/D in the whole output.
    """
    upper = text.upper()
    # Primary: explicit 'Answer: X' pattern
    m = re.search(r"ANSWER\s*:\s*([ABCD])", upper)
    if m:
        return m.group(1)
    # Fallback 1: look in the tail after last 'answer' mention
    last_ans = upper.rfind("ANSWER")
    if last_ans != -1:
        tail = upper[last_ans:]
        m2 = re.search(r"\b([ABCD])\b", tail)
        if m2:
            return m2.group(1)
    # Fallback 2: last standalone letter in full output
    letters = re.findall(r"\b([ABCD])\b", upper)
    return letters[-1] if letters else "X"


# ── Core runner ───────────────────────────────────────────────────────────────

def peak_vram_mb() -> float:
    if _CUDA:
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def run_strategy(
    llm,
    strategy: str,
    mmlu_data: dict,
    ctx: int,
) -> tuple[dict, list]:
    """Run one prompt strategy over all subjects.
    Returns (result_dict, examples) where examples holds 1 sample per subject.
    """
    max_prompt_tokens = ctx - 32   # headroom for output

    is_cot    = strategy == "cot"
    max_tok   = 500 if is_cot else 8   # 500 ensures CoT reasoning is never truncated
    extractor = extract_answer_cot if is_cot else extract_answer_short

    scores        = {}
    correct_total = 0
    total         = 0
    skipped       = 0
    truncated     = 0   # CoT: finish_reason == "length"
    examples      = []   # one example per subject for report
    start_all     = time.time()

    for subj, ds in mmlu_data.items():
        # Build few-shot demo pool from first FEW_SHOT_N samples
        demos = [
            (ds[i]["question"], ds[i]["choices"], ds[i]["answer"])
            for i in range(min(FEW_SHOT_N, len(ds)))
        ]
        eval_samples = ds if strategy != "few_shot" else ds.select(range(FEW_SHOT_N, len(ds)))
        if strategy == "cot":
            eval_samples = eval_samples.select(range(min(COT_MAX_PER_SUBJ, len(eval_samples))))

        correct      = 0
        subj_example = None   # capture first valid sample in this subject
        n_eval_total = len(eval_samples)
        print(f"    [{strategy}] {subj}  (0/{n_eval_total})", end="", flush=True)

        for q_idx, sample in enumerate(eval_samples):
            if strategy == "zero_shot":
                prompt = build_zero_shot(sample["question"], sample["choices"])
            elif strategy == "role":
                prompt = build_role(sample["question"], sample["choices"], subject=subj)
            elif strategy == "few_shot":
                prompt = build_few_shot(sample["question"], sample["choices"], demos=demos)
            else:  # cot
                prompt = build_cot(sample["question"], sample["choices"], subject=subj)

            token_ids = llm.tokenize(prompt.encode())
            if len(token_ids) > max_prompt_tokens:
                skipped += 1
                continue

            out      = llm(prompt, max_tokens=max_tok, temperature=0.0)
            raw_text = out["choices"][0]["text"]
            if is_cot and out["choices"][0].get("finish_reason") == "length":
                truncated += 1
            pred     = extractor(raw_text)
            gold     = CHOICES[sample["answer"]]
            if pred == gold:
                correct += 1

            # Per-question progress (overwrite same line)
            done = q_idx + 1
            running_acc = correct / done
            print(f"\r    [{strategy}] {subj}  {done}/{n_eval_total}  acc={running_acc:.3f}", end="", flush=True)

            # Save the very first valid example per subject
            if subj_example is None:
                subj_example = {
                    "subject":   subj,
                    "question":  sample["question"],
                    "choices":   {CHOICES[i]: c for i, c in enumerate(sample["choices"])},
                    "prompt":    prompt,
                    "response":  raw_text.strip(),
                    "predicted": pred,
                    "gold":      gold,
                    "correct":   pred == gold,
                }

        if subj_example:
            examples.append(subj_example)

        n_eval         = len(eval_samples)
        acc            = correct / n_eval if n_eval > 0 else 0.0
        scores[subj]   = acc
        correct_total += correct
        total         += n_eval
        # End the progress line, then print final subject result
        print(f"\r    [{strategy}] {subj}: {acc:.3f} ({correct}/{n_eval})          ")

    elapsed     = time.time() - start_all
    overall_acc = correct_total / total if total > 0 else 0.0

    trunc_rate = f"{truncated}/{total} ({truncated/total*100:.1f}%)" if is_cot and total > 0 else "N/A"
    print(f"  → Overall {overall_acc:.3f}  time={elapsed:.1f}s  skipped={skipped}  truncated={trunc_rate}")

    result = {
        "strategy":        strategy,
        "Overall Acc":     round(overall_acc, 3),
        **{k: round(v, 3) for k, v in scores.items()},
        "Time(s)":         round(elapsed, 1),
        "Skipped":         skipped,
        "Truncated":       truncated if is_cot else 0,
        "Truncate Rate":   round(truncated / total, 3) if is_cot and total > 0 else 0.0,
    }
    return result, examples


def run_model(model_name: str, model_path: str, mmlu_data: dict,
              strategies: list) -> tuple[list, list]:
    """Load model once, run requested strategies, return (rows, examples)."""
    from llama_cpp import Llama

    print(f"\n{'#'*60}")
    print(f"# Model: {model_name}")
    print(f"{'#'*60}")

    if _CUDA:
        torch.cuda.reset_peak_memory_stats()

    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=4,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )

    rows     = []
    all_examples = []
    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        row, examples = run_strategy(llm, strategy, mmlu_data, N_CTX)
        row["model"]         = model_name
        row["n_ctx"]         = N_CTX
        row["Peak VRAM(MB)"] = round(peak_vram_mb(), 1)
        rows.append(row)
        for ex in examples:
            ex["model"]    = model_name
            ex["strategy"] = strategy
        all_examples.extend(examples)

    del llm
    gc.collect()
    if _CUDA:
        torch.cuda.empty_cache()

    return rows, all_examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", action="store_true",
                        help="Include CoT strategy (slow, ~18x per question)")
    parser.add_argument("--all-models", action="store_true",
                        help="Also run Qwen2.5-1.5B (skipped by default)")
    args = parser.parse_args()

    models = MODELS_ALL if args.all_models else MODELS_DEFAULT
    print(f"[INFO] Models: {[m[0] for m in models]}")

    if args.cot:
        strategies = ["cot"]
        print("[INFO] CoT-only mode")
    else:
        strategies = ["zero_shot", "role", "few_shot"]
        print("[INFO] CoT skipped (pass --cot to enable)")

    # Load MMLU once
    print("Loading MMLU subjects...")
    mmlu_data = {}
    for subj in MMLU_SUBJECTS:
        ds = load_from_disk(str(MMLU_BENCH_DIR / f"mmlu_{subj}"))
        mmlu_data[subj] = ds
        print(f"  {subj}: {len(ds)} questions")

    all_rows     = []
    all_examples = []
    for model_name, model_path in models:
        rows, examples = run_model(model_name, model_path, mmlu_data,
                                   strategies=strategies)
        all_rows.extend(rows)
        all_examples.extend(examples)

    # Build DataFrame for this run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_order = (
        ["model", "strategy", "n_ctx", "Overall Acc"]
        + list(MMLU_SUBJECTS)
        + ["Time(s)", "Skipped", "Truncated", "Truncate Rate", "Peak VRAM(MB)"]
    )
    df = pd.DataFrame(all_rows)[col_order]
    df.insert(0, "timestamp", ts)
    df = df.set_index(["timestamp", "model", "strategy"])

    # Incremental save: append to existing CSV if it exists
    out_path = OUT_TABLES / "llm_prompt_strategies.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path, index_col=["timestamp", "model", "strategy"])
        df = pd.concat([existing, df])
    df.to_csv(out_path)

    # Save examples (append to existing JSON)
    ex_path = OUT_TABLES / "llm_prompt_examples.json"
    if ex_path.exists():
        with open(ex_path, "r", encoding="utf-8") as f:
            existing_ex = json.load(f)
    else:
        existing_ex = []
    for ex in all_examples:
        ex["timestamp"] = ts
    with open(ex_path, "w", encoding="utf-8") as f:
        json.dump(existing_ex + all_examples, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Timestamp : {ts}")
    print(f"Saved -> {out_path}  (total {len(df)} rows)")
    print(f"Saved -> {ex_path}  ({len(existing_ex + all_examples)} examples total)")
    print(df.to_string())


if __name__ == "__main__":
    main()
