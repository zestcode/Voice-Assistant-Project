"""
LLM Benchmark — Llama-3.2-3B-Instruct (Q4_K_M) on MMLU 5-subject subset
Hyperparameter sweep: n_ctx in [512, 1024, 2048]
Metrics: Accuracy (A/B/C/D), Wall-clock time (s), Peak VRAM (MB)

Required env packages:
  llama-cpp-python, datasets, pandas, torch

Usage:
  python benchmark_llm_llama.py
"""

import gc
import os
import sys
import time

# Register CUDA DLL directories for Python 3.8+ (PATH is no longer used for DLL resolution)
for _dll_dir in [
    r"D:\Softeware\miniconda\envs\llm\Library\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
]:
    if os.path.isdir(_dll_dir):
        os.add_dll_directory(_dll_dir)

import pandas as pd
from datasets import load_from_disk

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import LLM2_PATH, MMLU_BENCH_DIR, MMLU_SUBJECTS, OUT_TABLES

try:
    import torch
    N_GPU_LAYERS = -1 if torch.cuda.is_available() else 0
    _CUDA = torch.cuda.is_available()
except ImportError:
    N_GPU_LAYERS = -1
    _CUDA = False

CHOICES   = ["A", "B", "C", "D"]
CTX_SIZES = [1024, 2048, 5012]         # ← hyperparameter sweep


def format_mmlu_prompt(question: str, choices: list) -> str:
    prompt = f"Question: {question}\n"
    for i, c in enumerate(choices):
        prompt += f"{CHOICES[i]}. {c}\n"
    prompt += "Answer with just the letter (A, B, C, or D): "
    return prompt


def extract_answer(text: str) -> str:
    text = text.strip().upper()
    for ch in CHOICES:
        if ch in text:
            return ch
    return text[:1] if text else "X"


def peak_vram_mb() -> float:
    """Return peak VRAM since last reset, in MB. Returns 0 if no CUDA."""
    if _CUDA:
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def run_ctx(ctx: int, mmlu_data: dict) -> dict:
    """Load model with given n_ctx, run full MMLU sweep, return result dict."""
    from llama_cpp import Llama

    print(f"\n{'='*60}")
    print(f"n_ctx = {ctx}")
    print(f"{'='*60}")

    if _CUDA:
        torch.cuda.reset_peak_memory_stats()

    print(f"Loading Llama-3.2-3B (n_ctx={ctx}, n_gpu_layers={N_GPU_LAYERS}) ...")
    llm = Llama(
        model_path=str(LLM2_PATH),
        n_ctx=ctx,
        n_threads=4,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )

    scores        = {}
    correct_total = 0
    total         = 0
    start_all     = time.time()

    max_prompt_tokens = ctx - 16   # leave room for output tokens

    for subj, ds in mmlu_data.items():
        correct = 0
        for sample in ds:
            prompt      = format_mmlu_prompt(sample["question"], sample["choices"])
            token_ids   = llm.tokenize(prompt.encode())
            if len(token_ids) > max_prompt_tokens:
                continue   # prompt too long for this n_ctx; skip (counts as wrong)
            out  = llm(prompt, max_tokens=8, temperature=0.0)
            pred = extract_answer(out["choices"][0]["text"])
            gold = CHOICES[sample["answer"]]
            if pred == gold:
                correct += 1
        acc = correct / len(ds)
        scores[subj]   = acc
        correct_total += correct
        total         += len(ds)
        print(f"  {subj}: {acc:.3f} ({correct}/{len(ds)})")

    elapsed     = time.time() - start_all
    overall_acc = correct_total / total
    vram_mb     = peak_vram_mb()

    print(f"\nOverall Acc : {overall_acc:.3f} ({correct_total}/{total})")
    print(f"Time        : {elapsed:.1f}s")
    print(f"Peak VRAM   : {vram_mb:.1f} MB")

    del llm
    gc.collect()
    if _CUDA:
        torch.cuda.empty_cache()

    return {
        "n_ctx":       ctx,
        "Overall Acc": round(overall_acc, 3),
        **{k: round(v, 3) for k, v in scores.items()},
        "Time(s)":     round(elapsed, 1),
        "Peak VRAM(MB)": round(vram_mb, 1),
    }


def main():
    # ── Load MMLU subjects once
    mmlu_data = {}
    total_q   = 0
    for subj in MMLU_SUBJECTS:
        ds = load_from_disk(str(MMLU_BENCH_DIR / f"mmlu_{subj}"))
        mmlu_data[subj] = ds
        total_q += len(ds)
        print(f"  {subj}: {len(ds)} questions")
    print(f"\nTotal MMLU questions: {total_q}")

    # ── Sweep over context lengths
    rows = []
    for ctx in CTX_SIZES:
        rows.append(run_ctx(ctx, mmlu_data))

    # ── Save results
    df = pd.DataFrame(rows).set_index("n_ctx")
    out_path = OUT_TABLES / "llm_llama_benchmark.csv"
    df.to_csv(out_path)
    print(f"\nSaved results to {out_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()
