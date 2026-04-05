"""
ASR Benchmark — Moonshine-Base on LibriSpeech test-clean
Metrics: WER (Word Error Rate), RTFx (Inverse Real-Time Factor)

Required env packages:
  transformers, datasets, jiwer, numpy, pandas, torch, soundfile

Usage:
  python benchmark_asr_moonshine.py
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
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import ASR2_PATH, ASR_BENCH_PATH, OUT_TABLES

# ── settings
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.float16 if DEVICE == "cuda" else torch.float32
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
    print(f"\nLoading Moonshine-Base via transformers (device={DEVICE}) ...")
    processor = AutoProcessor.from_pretrained(str(ASR2_PATH))
    ms_model  = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(ASR2_PATH), torch_dtype=DTYPE
    ).to(DEVICE)

    refs, hyps = [], []
    total_audio_dur = 0.0
    total_infer_time = 0.0

    for i, sample in enumerate(ls_subset):
        audio_info = sample["audio"]
        raw = audio_info.get("bytes") or open(audio_info["path"], "rb").read()
        audio_array, sr = sf.read(io.BytesIO(raw), dtype="float32")
        ref_text = normalize_text(sample["text"])
        audio_dur   = len(audio_array) / sr

        start  = time.time()
        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE, dtype=DTYPE) for k, v in inputs.items()}
        with torch.no_grad():
            predicted_ids = ms_model.generate(**inputs, max_new_tokens=256, max_length=None)
        hyp_text = normalize_text(
            processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        )
        elapsed = time.time() - start

        refs.append(ref_text)
        hyps.append(hyp_text)
        total_audio_dur  += audio_dur
        total_infer_time += elapsed

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ls_subset)}] running WER: {wer(refs, hyps):.4f}")

    ms_wer  = wer(refs, hyps)
    ms_rtfx = total_audio_dur / total_infer_time

    print(f"\n=== Moonshine-Base ===")
    print(f"WER:  {ms_wer:.4f} ({ms_wer * 100:.2f}%)")
    print(f"RTFx: {ms_rtfx:.1f}x realtime")
    print(f"Total audio: {total_audio_dur:.1f}s, Inference: {total_infer_time:.1f}s")

    # ── Save results
    result = {"Moonshine-Base": {"WER%": round(ms_wer * 100, 2), "RTFx": round(ms_rtfx, 2)}}
    df = pd.DataFrame(result).T
    out_path = OUT_TABLES / "asr_moonshine_benchmark.csv"
    df.to_csv(out_path)
    print(f"\nSaved to {out_path}")

    del ms_model, processor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
