"""
ASR Benchmark — Faster-Whisper-Small on LibriSpeech test-clean
Metrics: WER (Word Error Rate), RTFx (Inverse Real-Time Factor)

Required env packages:
  faster-whisper, datasets, jiwer, numpy, pandas, torch

Usage:
  python benchmark_asr_whisper.py
"""

import gc
import io
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, load_from_disk
from faster_whisper import WhisperModel
from jiwer import wer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import ASR1_PATH, ASR_BENCH_PATH, OUT_TABLES

# ── settings
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_FP  = "float16" if DEVICE == "cuda" else "int8"
ASR_N_SAMPLES = 200   # set to None for full dataset


def normalize_text(t: str) -> str:
    return t.strip().lower()


def main():
    # ── Load LibriSpeech test-clean
    print(f"Loading LibriSpeech test-clean from {ASR_BENCH_PATH} ...")
    ls_test = load_from_disk(str(ASR_BENCH_PATH))
    n = min(ASR_N_SAMPLES, len(ls_test)) if ASR_N_SAMPLES else len(ls_test)
    ls_subset = ls_test.select(range(n))
    ls_subset = ls_subset.cast_column("audio", Audio(decode=False))
    print(f"Using {len(ls_subset)} / {len(ls_test)} samples")

    # ── Load model
    print(f"\nLoading Faster-Whisper-Small (device={DEVICE}) ...")
    fw_model = WhisperModel(str(ASR1_PATH), device=DEVICE, compute_type=COMPUTE_FP)

    refs, hyps = [], []
    total_audio_dur = 0.0
    total_infer_time = 0.0

    for i, sample in enumerate(ls_subset):
        audio_info = sample["audio"]
        raw = audio_info.get("bytes") or open(audio_info["path"], "rb").read()
        audio_array, sr = sf.read(io.BytesIO(raw), dtype="float32")
        ref_text = normalize_text(sample["text"])
        audio_dur   = len(audio_array) / sr

        start = time.time()
        segments, info = fw_model.transcribe(audio_array, beam_size=5)
        hyp_text = normalize_text(" ".join([seg.text for seg in segments]))
        elapsed  = time.time() - start

        refs.append(ref_text)
        hyps.append(hyp_text)
        total_audio_dur  += audio_dur
        total_infer_time += elapsed

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ls_subset)}] running WER: {wer(refs, hyps):.4f}")

    fw_wer  = wer(refs, hyps)
    fw_rtfx = total_audio_dur / total_infer_time

    print(f"\n=== Faster-Whisper-Small ===")
    print(f"WER:  {fw_wer:.4f} ({fw_wer * 100:.2f}%)")
    print(f"RTFx: {fw_rtfx:.1f}x realtime")
    print(f"Total audio: {total_audio_dur:.1f}s, Inference: {total_infer_time:.1f}s")

    # ── Save results
    result = {"Faster-Whisper-Small": {"WER%": round(fw_wer * 100, 2), "RTFx": round(fw_rtfx, 2)}}
    df = pd.DataFrame(result).T
    out_path = OUT_TABLES / "asr_whisper_benchmark.csv"
    df.to_csv(out_path)
    print(f"\nSaved to {out_path}")

    del fw_model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
