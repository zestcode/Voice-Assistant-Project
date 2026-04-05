"""
TTS Benchmark — GPT-SoVITS on EmergentTTS-Eval (Round-Trip WER)
Metric: synthesize → Whisper-small transcribe → WER vs original text

Required env packages:
  faster-whisper, datasets, jiwer, soundfile, librosa, numpy, pandas,
  torch, requests

GPT-SoVITS inference server must be running before executing this script.
Start it from the GPT-SoVITS repo root (in its own env):
  cd code/GPT-SoVITS
  python api_v2.py

Also set REF_AUDIO / REF_TEXT in config.py to your reference voice clip.

Usage:
  python benchmark_tts_gptsovits.py
"""

import gc
import io
import sys
import time

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from jiwer import wer
from resemblyzer import VoiceEncoder, preprocess_wav

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import (
    ASR1_PATH,
    TTS_BENCH_PATH,
    OUT_TABLES, OUT_TTS_GPTSOVITS,
    REF_AUDIO, REF_TEXT,
    GPTSOVITS_API,
)

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


def compute_roundtrip_wer(whisper_judge, audio_array: np.ndarray, original_text: str, sr: int):
    if sr != 16000:
        import librosa
        audio_16k = librosa.resample(audio_array.astype(np.float32), orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio_array.astype(np.float32)
    segments, _ = whisper_judge.transcribe(audio_16k, beam_size=5)
    hyp = normalize_text(" ".join([s.text for s in segments]))
    ref = normalize_text(original_text)
    return wer(ref, hyp), hyp


def _ref_audio_samples() -> int:
    """返回参考音频的样本数，用于从输出中截掉拼接的参考段。"""
    ref_np, ref_sr = sf.read(str(REF_AUDIO))
    return len(ref_np), ref_sr


def synthesize_gptsovits(text: str) -> tuple[np.ndarray, int]:
    """Call the GPT-SoVITS HTTP inference API, strip prepended reference audio."""
    params = {
        "ref_audio_path":    str(REF_AUDIO),
        "prompt_text":       REF_TEXT,
        "prompt_lang":       "en",
        "text":              text,
        "text_lang":         "en",
        "media_type":        "wav",
        "text_split_method": "cut0",   # no splitting — avoids comma-fragmentation
        "batch_size":        1,
    }
    resp = requests.get(GPTSOVITS_API + "/tts", params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"API returned {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()
    audio_np, sr = sf.read(io.BytesIO(resp.content))
    audio_np = audio_np.astype(np.float32)

    # GPT-SoVITS prepends the reference audio to the output by default.
    # Trim it out so round-trip WER only measures the synthesized portion.
    ref_samples, ref_sr = _ref_audio_samples()
    # resample ref_samples count to match output sr if they differ
    trim_samples = int(ref_samples * sr / ref_sr)
    if len(audio_np) > trim_samples:
        audio_np = audio_np[trim_samples:]

    return audio_np, sr


def main():
    # ── Pre-flight checks
    if not REF_AUDIO.exists():
        print(f"ERROR: reference audio not found at {REF_AUDIO}")
        print("Place a ref.wav in data/ref_voice/ and update REF_TEXT in config.py")
        return

    print(f"GPT-SoVITS API: {GPTSOVITS_API}")
    print(f"Ref audio:      {REF_AUDIO}")
    try:
        requests.get(GPTSOVITS_API + "/tts", timeout=3)
    except Exception:
        print(f"ERROR: cannot reach GPT-SoVITS server at {GPTSOVITS_API}")
        print("Start it with:  python api_v2.py")
        return

    prompts = load_tts_prompts()

    # ── Load Whisper judge
    print(f"\nLoading Whisper-small for round-trip WER (device={DEVICE}) ...")
    whisper_judge = WhisperModel(str(ASR1_PATH), device=DEVICE, compute_type=COMPUTE_FP)

    # ── Load speaker encoder and pre-compute reference embedding
    print("Loading speaker encoder (resemblyzer) ...")
    spk_encoder = VoiceEncoder()
    ref_embed = spk_encoder.embed_utterance(preprocess_wav(REF_AUDIO))

    sovits_wers      = []
    sovits_latencies = []
    sovits_spk_sims  = []
    OUT_TTS_GPTSOVITS.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        start = time.time()
        try:
            audio_np, sr = synthesize_gptsovits(prompt)
        except Exception as e:
            print(f"  [{i}] GPT-SoVITS failed: {e}")
            continue
        latency = time.time() - start
        sovits_latencies.append(latency)

        wav_path = OUT_TTS_GPTSOVITS / f"sample_{i:03d}.wav"
        sf.write(str(wav_path), audio_np, sr)

        sample_wer, _ = compute_roundtrip_wer(whisper_judge, audio_np, prompt, sr=sr)
        sovits_wers.append(sample_wer)

        synth_embed = spk_encoder.embed_utterance(preprocess_wav(wav_path))
        spk_sim = float(np.dot(ref_embed, synth_embed) /
                        (np.linalg.norm(ref_embed) * np.linalg.norm(synth_embed)))
        sovits_spk_sims.append(spk_sim)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] avg WER: {np.mean(sovits_wers):.4f}, "
                  f"avg spk_sim: {np.mean(sovits_spk_sims):.4f}, "
                  f"avg latency: {np.mean(sovits_latencies):.2f}s")

    if not sovits_wers:
        print("No results collected — check server logs.")
        return

    print(f"\n=== GPT-SoVITS ===")
    print(f"Round-trip WER:     {np.mean(sovits_wers):.4f} ({np.mean(sovits_wers) * 100:.2f}%)")
    print(f"Speaker Similarity: {np.mean(sovits_spk_sims):.4f}")
    print(f"Avg latency:        {np.mean(sovits_latencies):.2f}s")

    # ── Save results
    row = {
        "Round-trip WER":     round(float(np.mean(sovits_wers)), 4),
        "Speaker Similarity": round(float(np.mean(sovits_spk_sims)), 4),
        "Avg Latency(s)":     round(float(np.mean(sovits_latencies)), 4),
        "Samples":            len(sovits_wers),
    }
    df = pd.DataFrame({"GPT-SoVITS": row}).T
    out_path = OUT_TABLES / "tts_gptsovits_benchmark.csv"
    df.to_csv(out_path)
    print(f"Saved to {out_path}")

    del whisper_judge
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
