"""
TTS Benchmark — Kokoro on EmergentTTS-Eval (Round-Trip WER)
Metric: synthesize → Whisper-small transcribe → WER vs original text

Required env packages:
  kokoro>=0.3.4, faster-whisper, datasets, jiwer, soundfile, librosa,
  numpy, pandas, torch

Usage:
  python benchmark_tts_kokoro.py
"""

import gc
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from jiwer import wer
from kokoro import KPipeline

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import ASR1_PATH, TTS_BENCH_PATH, OUT_TABLES, OUT_TTS_KOKORO

# ── settings
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_FP = "float16" if DEVICE == "cuda" else "int8"
TTS_N      = 50

FALLBACK_PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "She sells seashells by the seashore on a sunny Saturday morning.",
    "The temperature is expected to reach thirty-seven point five degrees Celsius.",
    "Can you explain how photosynthesis works in simple terms?",
    "According to the quarterly report, revenue increased by 12.5 percent.",
    "Dr. Elizabeth Johnson presented her findings at the IEEE conference.",
    "Is it true that the speed of light is approximately 299,792 kilometers per second?",
    "The patient should take 500 milligrams of amoxicillin every eight hours.",
    "Please visit our website at www.example.com for more information.",
    "The Renaissance period began in Italy during the 14th century.",
    "What is the difference between machine learning and deep learning?",
    "The orchestra performed Beethoven's Symphony Number 9 in D minor.",
    "Excuse me, could you tell me how to get to the nearest subway station?",
    "The experiment yielded statistically significant results with a p-value of 0.003.",
    "I'm sorry to hear about your situation. Let me see how I can help.",
    "Congratulations on your promotion! You really deserve this recognition.",
    "Warning: this product contains allergens including nuts, soy, and wheat.",
    "The algorithm has a time complexity of O of n log n in the average case.",
    "Tokyo, Paris, New York, and Sydney are among the world's most visited cities.",
]


def load_tts_prompts() -> list:
    prompts = []
    try:
        from datasets import load_from_disk
        tts_ds = load_from_disk(str(TTS_BENCH_PATH))
        if isinstance(tts_ds, dict):
            tts_ds = tts_ds[list(tts_ds.keys())[0]]
        for field in ["text", "sentence", "prompt", "input_text"]:
            if field in tts_ds.column_names:
                prompts = [s[field] for s in tts_ds]
                break
        print(f"Loaded {len(prompts)} prompts from EmergentTTS-Eval")
    except Exception as e:
        print(f"Could not load EmergentTTS-Eval: {e}")
    if not prompts:
        prompts = FALLBACK_PROMPTS
        print(f"Using {len(prompts)} curated fallback prompts")
    prompts = prompts[:TTS_N]
    print(f"Will evaluate on {len(prompts)} prompts")
    return prompts


def normalize_text(t: str) -> str:
    return t.strip().lower()


def compute_roundtrip_wer(whisper_judge, audio_array: np.ndarray, original_text: str, sr: int = 24000):
    if sr != 16000:
        import librosa
        audio_16k = librosa.resample(audio_array.astype(np.float32), orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio_array.astype(np.float32)
    segments, _ = whisper_judge.transcribe(audio_16k, beam_size=5)
    hyp = normalize_text(" ".join([s.text for s in segments]))
    ref = normalize_text(original_text)
    return wer(ref, hyp), hyp


def main():
    prompts = load_tts_prompts()

    # ── Load Whisper judge
    print(f"\nLoading Whisper-small for round-trip WER (device={DEVICE}) ...")
    whisper_judge = WhisperModel(str(ASR1_PATH), device=DEVICE, compute_type=COMPUTE_FP)

    # ── Load Kokoro
    print("Loading Kokoro ...")
    kokoro_pipe = KPipeline(lang_code='a')  # American English

    kokoro_wers      = []
    kokoro_latencies = []
    OUT_TTS_KOKORO.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        start = time.time()
        try:
            segments = list(kokoro_pipe(prompt, voice='af_bella', speed=1.0))
            last = segments[-1]
            if isinstance(last, tuple) and len(last) >= 3:
                audio = last[2]
            elif hasattr(last, 'audio'):
                audio = last.audio
            else:
                audio = np.concatenate([
                    s[2] if isinstance(s, tuple) else s.audio for s in segments
                ])
        except Exception as e:
            print(f"  [{i}] Kokoro failed: {e}")
            continue

        latency  = time.time() - start
        audio_np = np.array(audio, dtype=np.float32)
        sf.write(str(OUT_TTS_KOKORO / f"sample_{i:03d}.wav"), audio_np, 24000)

        sample_wer, _ = compute_roundtrip_wer(whisper_judge, audio_np, prompt, sr=24000)
        kokoro_wers.append(sample_wer)
        kokoro_latencies.append(latency)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] avg WER: {np.mean(kokoro_wers):.4f}, "
                  f"avg latency: {np.mean(kokoro_latencies):.2f}s")

    print(f"\n=== Kokoro ===")
    print(f"Round-trip WER: {np.mean(kokoro_wers):.4f} ({np.mean(kokoro_wers) * 100:.2f}%)")
    print(f"Avg latency:    {np.mean(kokoro_latencies):.2f}s")

    # ── Save results
    row = {
        "Round-trip WER": round(float(np.mean(kokoro_wers)), 4),
        "Avg Latency(s)": round(float(np.mean(kokoro_latencies)), 4),
        "Samples":        len(kokoro_wers),
    }
    df = pd.DataFrame({"Kokoro": row}).T
    out_path = OUT_TABLES / "tts_kokoro_benchmark.csv"
    df.to_csv(out_path)
    print(f"Saved to {out_path}")

    del kokoro_pipe, whisper_judge
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
