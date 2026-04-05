"""
One-off script: save Qwen2.5-1.5B prompt-strategy results (from terminal output)
into outputs/tables/llm_prompt_strategies.csv, merging with any existing rows.

Usage:
  python scripts/save_qwen15b_results.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MMLU_SUBJECTS, OUT_TABLES

SUBJECTS = list(MMLU_SUBJECTS)

# ── Hard-coded results from terminal output ───────────────────────────────────
ROWS = [
    {
        "model": "Qwen2.5-1.5B", "strategy": "zero_shot", "n_ctx": 2048,
        "Overall Acc": 0.248,
        "high_school_mathematics": 0.244, "high_school_physics": 0.192,
        "high_school_biology": 0.223,     "high_school_world_history": 0.325,
        "high_school_us_history": 0.304,  "college_mathematics": 0.270,
        "college_physics": 0.167,         "college_biology": 0.257,
        "philosophy": 0.212,              "international_law": 0.281,
        "Time(s)": 122.2, "Skipped": 0, "Peak VRAM(MB)": 0.0,
    },
    {
        "model": "Qwen2.5-1.5B", "strategy": "role", "n_ctx": 2048,
        "Overall Acc": 0.276,
        "high_school_mathematics": 0.296, "high_school_physics": 0.219,
        "high_school_biology": 0.268,     "high_school_world_history": 0.300,
        "high_school_us_history": 0.299,  "college_mathematics": 0.240,
        "college_physics": 0.284,         "college_biology": 0.278,
        "philosophy": 0.267,              "international_law": 0.289,
        "Time(s)": 122.4, "Skipped": 0, "Peak VRAM(MB)": 0.0,
    },
    {
        "model": "Qwen2.5-1.5B", "strategy": "few_shot", "n_ctx": 2048,
        "Overall Acc": 0.431,
        "high_school_mathematics": 0.281, "high_school_physics": 0.243,
        "high_school_biology": 0.511,     "high_school_world_history": 0.688,
        "high_school_us_history": 0.612,  "college_mathematics": 0.392,
        "college_physics": 0.293,         "college_biology": 0.468,
        "philosophy": 0.299,              "international_law": 0.424,
        "Time(s)": 129.8, "Skipped": 0, "Peak VRAM(MB)": 0.0,
    },
    {
        "model": "Qwen2.5-1.5B", "strategy": "cot", "n_ctx": 2048,
        "Overall Acc": 0.532,
        "high_school_mathematics": 0.360, "high_school_physics": 0.340,
        "high_school_biology": 0.640,     "high_school_world_history": 0.740,
        "high_school_us_history": 0.520,  "college_mathematics": 0.260,
        "college_physics": 0.360,         "college_biology": 0.700,
        "philosophy": 0.660,              "international_law": 0.740,
        "Time(s)": 432.0, "Skipped": 0, "Peak VRAM(MB)": 0.0,
    },
]

COL_ORDER = (
    ["model", "strategy", "n_ctx", "Overall Acc"]
    + SUBJECTS
    + ["Time(s)", "Skipped", "Peak VRAM(MB)"]
)

out_path = OUT_TABLES / "llm_prompt_strategies.csv"

new_df = pd.DataFrame(ROWS)[COL_ORDER].set_index(["model", "strategy"])

if out_path.exists():
    existing = pd.read_csv(out_path, index_col=["model", "strategy"])
    # Drop any old Qwen2.5-1.5B rows then append fresh ones
    existing = existing[~existing.index.get_level_values("model").isin(["Qwen2.5-1.5B"])]
    combined = pd.concat([existing, new_df])
else:
    combined = new_df

combined.to_csv(out_path)
print(f"Saved {len(new_df)} rows -> {out_path}")
print(combined.to_string())
