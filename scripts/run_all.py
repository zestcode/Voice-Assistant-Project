"""
Run all benchmarks sequentially in the current conda environment.
Failed benchmarks are skipped with a warning; others continue.

For GPT-SoVITS: start the server first in a separate terminal:
  cd code/GPT-SoVITS && python api_v2.py

Usage:
  python scripts/run_all.py
  python scripts/run_all.py --skip gptsovits      # skip specific models
  python scripts/run_all.py --only asr_whisper    # run one model only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

BENCHMARKS = [
    ("asr_whisper",   "benchmark_asr_whisper.py"),
    ("asr_moonshine", "benchmark_asr_moonshine.py"),
    ("llm_qwen",      "benchmark_llm_qwen.py"),
    ("llm_llama",     "benchmark_llm_llama.py"),
    ("tts_f5",        "benchmark_tts_f5.py"),        # F5-TTS voice cloning (gptsovits env)
    ("tts_gptsovits", "benchmark_tts_gptsovits.py"), # GPT-SoVITS via HTTP API (optional)
]


def run_benchmark(name: str, script: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script)],
        cwd=str(SCRIPTS_DIR.parent),
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n[OK] {name} finished in {elapsed:.0f}s")
        return True
    else:
        print(f"\n[FAILED] {name} exited with code {result.returncode} after {elapsed:.0f}s")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="+", default=[], metavar="NAME",
                        help="benchmark names to skip")
    parser.add_argument("--only", nargs="+", default=[], metavar="NAME",
                        help="run only these benchmarks")
    args = parser.parse_args()

    targets = [(n, s) for n, s in BENCHMARKS
               if (not args.only or n in args.only)
               and n not in args.skip]

    if not targets:
        print("No benchmarks selected.")
        return

    print(f"Will run: {[n for n, _ in targets]}")

    results = {}
    for name, script in targets:
        results[name] = run_benchmark(name, script)

    # ── Summary
    print(f"\n{'='*60}")
    print("RUN SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "OK     " if ok else "FAILED "
        print(f"  {status}  {name}")

    # ── Print benchmark results table
    print()
    subprocess.run([sys.executable, str(SCRIPTS_DIR / "summarize.py")],
                   cwd=str(SCRIPTS_DIR.parent))


if __name__ == "__main__":
    main()
