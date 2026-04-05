"""
Overall Benchmark Summary
Loads all per-model CSV results from outputs/tables/ and prints a combined report.

Usage:
  python summarize.py
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUT_TABLES


def load_csv(name: str) -> pd.DataFrame | None:
    path = OUT_TABLES / name
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def main():
    print("=" * 60)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)

    # ── ASR ──────────────────────────────────────────────────────────────────
    print("\n--- ASR (LibriSpeech test-clean) ---")
    asr_frames = [
        load_csv("asr_whisper_benchmark.csv"),
        load_csv("asr_moonshine_benchmark.csv"),
    ]
    asr_df = pd.concat([f for f in asr_frames if f is not None]) if any(f is not None for f in asr_frames) else None
    if asr_df is not None:
        for name, row in asr_df.iterrows():
            print(f"  {name:25s}  WER: {row.get('WER%', float('nan')):6.2f}%  RTFx: {row.get('RTFx', float('nan')):.1f}x")
    else:
        print("  (no results — run benchmark_asr_whisper.py / benchmark_asr_moonshine.py)")

    # ── LLM ──────────────────────────────────────────────────────────────────
    print("\n--- LLM (MMLU 5-subject) ---")
    llm_frames = [
        load_csv("llm_qwen_benchmark.csv"),
        load_csv("llm_llama_benchmark.csv"),
    ]
    llm_df = pd.concat([f for f in llm_frames if f is not None]) if any(f is not None for f in llm_frames) else None
    if llm_df is not None:
        for name, row in llm_df.iterrows():
            print(f"  {name:25s}  Acc: {row.get('Overall Acc', float('nan')) * 100:6.2f}%")
    else:
        print("  (no results — run benchmark_llm_qwen.py / benchmark_llm_llama.py)")

    # ── TTS ──────────────────────────────────────────────────────────────────
    print("\n--- TTS (Round-trip WER) ---")
    tts_frames = [
        load_csv("tts_f5_benchmark.csv"),
        load_csv("tts_gptsovits_benchmark.csv"),
        load_csv("tts_kokoro_benchmark.csv"),   # legacy, may not exist
    ]
    tts_df = pd.concat([f for f in tts_frames if f is not None]) if any(f is not None for f in tts_frames) else None
    if tts_df is not None:
        for name, row in tts_df.iterrows():
            print(f"  {name:25s}  WER: {row.get('Round-trip WER', float('nan')) * 100:6.2f}%  "
                  f"Latency: {row.get('Avg Latency(s)', float('nan')):.2f}s")
    else:
        print("  (no results — run benchmark_tts_kokoro.py / benchmark_tts_gptsovits.py)")

    print("\n" + "=" * 60)
    print(f"All results saved to: {OUT_TABLES}")


if __name__ == "__main__":
    main()
