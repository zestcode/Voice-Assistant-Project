"""
Prepare LLM benchmark assets:
  1. Download 7 new MMLU subjects to data/benchmarks/llm/
  2. Download Qwen2.5-3B-Instruct-Q4_K_M.gguf to models/llm/qwen2.5-3b-q4_k_m/
  3. Patch config.py  (LLM1_PATH + MMLU_SUBJECTS)

Usage:
  python scripts/prepare_llm_benchmark.py

Required packages:
  datasets, huggingface_hub
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASE, MMLU_BENCH_DIR

# ── Target MMLU subjects (full 10-subject list) ───────────────────────────────
NEW_MMLU_SUBJECTS = [
    # high school
    "high_school_mathematics",
    "high_school_physics",
    "high_school_biology",
    "high_school_world_history",
    "high_school_us_history",
    # college / university
    "college_mathematics",
    "college_physics",
    "college_biology",
    "philosophy",
    "international_law",
]

# Subjects already on disk from the old benchmark (skip re-download)
ALREADY_HAVE = {
    "college_physics",
    "high_school_world_history",
    "high_school_biology",
}

# ── Qwen 3B model ─────────────────────────────────────────────────────────────
QWEN3B_REPO     = "Qwen/Qwen2.5-3B-Instruct-GGUF"
QWEN3B_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
QWEN3B_DIR      = BASE / "models" / "llm" / "qwen2.5-3b-q4_k_m"
QWEN3B_PATH     = QWEN3B_DIR / QWEN3B_FILENAME


# ── Step 1 — MMLU subjects ────────────────────────────────────────────────────

def download_mmlu_subjects():
    from datasets import load_dataset

    MMLU_BENCH_DIR.mkdir(parents=True, exist_ok=True)

    to_download = [s for s in NEW_MMLU_SUBJECTS if s not in ALREADY_HAVE]
    # Also skip if folder already exists on disk
    to_download = [
        s for s in to_download
        if not (MMLU_BENCH_DIR / f"mmlu_{s}").exists()
    ]

    if not to_download:
        print("[MMLU] All subjects already on disk, nothing to download.")
        return

    print(f"[MMLU] Downloading {len(to_download)} subjects: {to_download}")
    for subj in to_download:
        print(f"  Downloading mmlu/{subj} ...", end=" ", flush=True)
        ds = load_dataset("cais/mmlu", subj, split="test")
        save_path = MMLU_BENCH_DIR / f"mmlu_{subj}"
        ds.save_to_disk(str(save_path))
        print(f"OK  ({len(ds)} questions → {save_path})")

    print("[MMLU] Done.\n")


# ── Step 2 — Qwen 3B GGUF ────────────────────────────────────────────────────

def download_qwen3b():
    if QWEN3B_PATH.exists():
        print(f"[Qwen3B] Already exists: {QWEN3B_PATH}")
        return

    from huggingface_hub import hf_hub_download

    QWEN3B_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Qwen3B] Downloading {QWEN3B_FILENAME} from {QWEN3B_REPO} ...")
    local = hf_hub_download(
        repo_id=QWEN3B_REPO,
        filename=QWEN3B_FILENAME,
        local_dir=str(QWEN3B_DIR),
    )
    print(f"[Qwen3B] Saved to {local}\n")


# ── Step 3 — Patch config.py ─────────────────────────────────────────────────

def patch_config():
    config_path = Path(__file__).resolve().parent / "config.py"
    content     = config_path.read_text(encoding="utf-8")
    original    = content

    # --- LLM1_PATH ---
    new_llm1 = (
        'LLM1_PATH = BASE / "models" / "llm" / "qwen2.5-3b-q4_k_m" '
        f'/ "{QWEN3B_FILENAME}"'
    )
    content = re.sub(r"LLM1_PATH\s*=.*", new_llm1, content)

    # --- MMLU_SUBJECTS list ---
    subjects_repr = "[\n" + "".join(f'    "{s}",\n' for s in NEW_MMLU_SUBJECTS) + "]"
    content = re.sub(
        r"MMLU_SUBJECTS\s*=\s*\[.*?\]",
        f"MMLU_SUBJECTS = {subjects_repr}",
        content,
        flags=re.DOTALL,
    )

    if content == original:
        print("[config] No changes needed.")
        return

    config_path.write_text(content, encoding="utf-8")
    print("[config] Patched config.py:")
    print(f"  LLM1_PATH  → {QWEN3B_PATH}")
    print(f"  MMLU_SUBJECTS → {NEW_MMLU_SUBJECTS}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1 / 3 — MMLU subjects")
    print("=" * 60)
    download_mmlu_subjects()

    print("=" * 60)
    print("Step 2 / 3 — Qwen2.5-3B-Instruct Q4_K_M")
    print("=" * 60)
    download_qwen3b()

    print("=" * 60)
    print("Step 3 / 3 — Patch config.py")
    print("=" * 60)
    patch_config()

    print("=" * 60)
    print("All done. Run `python scripts/config.py` to verify paths.")
    print("=" * 60)


if __name__ == "__main__":
    main()
