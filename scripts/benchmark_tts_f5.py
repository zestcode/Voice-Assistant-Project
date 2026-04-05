"""
TTS Benchmark — F5-TTS (voice cloning) on EmergentTTS-Eval (Round-Trip WER)
Metric: synthesize → Whisper-small transcribe → WER vs original text

Required env: gptsovits
  conda activate gptsovits
  python scripts/benchmark_tts_f5.py
"""

import gc
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel
from jiwer import wer
from resemblyzer import VoiceEncoder, preprocess_wav

# torchaudio 2.11 on Windows requires torchcodec/FFmpeg — patch to use soundfile instead
def _sf_load(path, frame_offset=0, num_frames=-1, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(data.T)
    if frame_offset > 0:
        tensor = tensor[:, frame_offset:]
    if num_frames > 0:
        tensor = tensor[:, :num_frames]
    return tensor, sr

torchaudio.load = _sf_load

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import ASR1_PATH, TTS_BENCH_PATH, OUT_TABLES, OUT_TTS_F5, REF_AUDIO, REF_TEXT

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
        print(f"Using {len(prompts)} fallback prompts")
    return prompts[:TTS_N]


def normalize_text(t: str) -> str:
    return t.strip().lower()


def compute_roundtrip_wer(whisper_judge, audio: np.ndarray, original_text: str, sr: int):
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    segments, _ = whisper_judge.transcribe(audio.astype(np.float32), beam_size=5)
    hyp = normalize_text(" ".join(s.text for s in segments))
    ref = normalize_text(original_text)
    return wer(ref, hyp)


def main():
    prompts = load_tts_prompts()
    print(f"Will evaluate {len(prompts)} prompts\n")

    print(f"Loading Whisper-small for round-trip WER (device={DEVICE}) ...")
    whisper_judge = WhisperModel(str(ASR1_PATH), device=DEVICE, compute_type=COMPUTE_FP)

    # ── Load speaker encoder and pre-compute reference embedding
    print("Loading speaker encoder (resemblyzer) ...")
    spk_encoder = VoiceEncoder()
    ref_embed = spk_encoder.embed_utterance(preprocess_wav(REF_AUDIO))

    from f5_tts.api import F5TTS
    print(f"Loading F5-TTS (device={DEVICE}) ...")
    tts = F5TTS(model="F5TTS_v1_Base", device=DEVICE)
    print(f"Using reference voice: {REF_AUDIO}\n")

    OUT_TTS_F5.mkdir(parents=True, exist_ok=True)

    wers, latencies, spk_sims = [], [], []

    for i, prompt in enumerate(prompts):
        t0 = time.time()
        try:
            wav, sr, _ = tts.infer(
                ref_file=str(REF_AUDIO),
                ref_text=REF_TEXT,
                gen_text=prompt,
                remove_silence=True,
            )
            audio = np.array(wav, dtype=np.float32)
        except Exception as e:
            print(f"  [{i}] F5-TTS failed: {e}")
            continue

        latency = time.time() - t0
        wav_path = OUT_TTS_F5 / f"sample_{i:03d}.wav"
        sf.write(str(wav_path), audio, sr)

        sample_wer = compute_roundtrip_wer(whisper_judge, audio, prompt, sr)
        wers.append(sample_wer)
        latencies.append(latency)

        synth_embed = spk_encoder.embed_utterance(preprocess_wav(wav_path))
        spk_sim = float(np.dot(ref_embed, synth_embed) /
                        (np.linalg.norm(ref_embed) * np.linalg.norm(synth_embed)))
        spk_sims.append(spk_sim)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] avg WER: {np.mean(wers):.4f}  "
                  f"avg spk_sim: {np.mean(spk_sims):.4f}  "
                  f"avg latency: {np.mean(latencies):.2f}s")

    print(f"\n=== F5-TTS ===")
    print(f"Round-trip WER:     {np.mean(wers):.4f} ({np.mean(wers) * 100:.2f}%)")
    print(f"Speaker Similarity: {np.mean(spk_sims):.4f}")
    print(f"Avg latency:        {np.mean(latencies):.2f}s")

    row = {
        "Round-trip WER":     round(float(np.mean(wers)), 4),
        "Speaker Similarity": round(float(np.mean(spk_sims)), 4),
        "Avg Latency(s)":     round(float(np.mean(latencies)), 4),
        "Samples":            len(wers),
    }
    df = pd.DataFrame({"F5-TTS": row}).T
    out_path = OUT_TABLES / "tts_f5_benchmark.csv"
    df.to_csv(out_path)
    print(f"Saved to {out_path}")

    del tts, whisper_judge
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
