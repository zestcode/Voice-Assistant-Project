"""
End-to-End Streaming Benchmark — WebSocket client
Measures TTFT, TTFA, and component latencies across 3 audio length tiers.

Prerequisites:
  python run.py   (starts ASR :8001, LLM :8002, TTS :8003, Orchestrator :7860)

Run in voiceui env:
  pip install websockets datasets soundfile scipy numpy pandas
  python scripts/benchmark_e2e_streaming.py

Outputs:
  outputs/tables/e2e_streaming_benchmark.csv   — per-sample rows
  outputs/tables/e2e_streaming_summary.csv     — mean/std/p50/p95 per tier
"""

import argparse
import asyncio
import base64
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ASR_BENCH_PATH, OUT_TABLES

# ── Config ────────────────────────────────────────────────────────────────────
WS_URL         = "ws://localhost:7860/ws"
SAMPLES_PER_TIER = 20      # samples per length tier
INTER_SAMPLE_S   = 3.0     # seconds to wait between samples
TIMEOUT_S        = 90.0    # max seconds to wait for one pipeline to complete

# Audio length tiers: (tier_name, n_concat, target_duration_s, trim_s)
# trim_s: truncate combined audio to this many seconds (None = no trim)
ALL_TIERS = [
    ("xshort", 1,  1, 0.8),
    ("short",  1,  5, None),
    ("medium", 3, 15, None),
    ("long",   6, 30, None),
]
ALL_TIER_NAMES = [t[0] for t in ALL_TIERS]


# ── Audio helpers ─────────────────────────────────────────────────────────────

def to_wav_bytes(arr: np.ndarray, sr: int) -> bytes:
    """Encode float32 numpy array as 16kHz mono WAV bytes."""
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        num = int(len(arr) * 16000 / sr)
        arr = scipy.signal.resample(arr, num).astype(np.float32)
        sr  = 16000
    buf = io.BytesIO()
    sf.write(buf, arr.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def arr_to_b64(wav_bytes: bytes) -> str:
    """Base64-encode WAV bytes in 32KB chunks (avoids call-stack overflow)."""
    CHUNK = 0x8000
    parts = []
    for i in range(0, len(wav_bytes), CHUNK):
        parts.append(wav_bytes[i:i + CHUNK])
    return base64.b64encode(b"".join(parts)).decode()


def load_librispeech_samples(n_needed: int) -> list[dict]:
    """Load samples from LibriSpeech Arrow file. Returns list of {array, sr, text}."""
    from datasets import load_from_disk, Audio
    ds = load_from_disk(str(ASR_BENCH_PATH))
    # Disable torchcodec-based decoding — use raw bytes + soundfile instead
    ds = ds.cast_column("audio", Audio(decode=False))
    samples = []
    for item in ds:
        audio = item["audio"]
        raw_bytes = audio.get("bytes") or open(audio["path"], "rb").read()
        arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        samples.append({
            "array": arr,
            "sr":    sr,
            "text":  item.get("text", ""),
        })
        if len(samples) >= n_needed:
            break
    print(f"[Bench] Loaded {len(samples)} LibriSpeech samples")
    return samples


def build_tier_inputs(samples: list[dict], n_concat: int, trim_s=None) -> list[dict]:
    """
    Concatenate n_concat consecutive samples (with 0.5s silence gap) into one input.
    If trim_s is set, truncate each combined audio to that many seconds.
    Returns list of {wav_bytes, duration_s, texts}.
    """
    inputs = []
    silence_05s = np.zeros(8000, dtype=np.float32)  # 0.5s at 16kHz

    i = 0
    while i + n_concat <= len(samples) and len(inputs) < SAMPLES_PER_TIER:
        group = samples[i:i + n_concat]
        parts = []
        texts = []
        for j, s in enumerate(group):
            arr = s["array"].copy()
            if s["sr"] != 16000:
                import scipy.signal
                num = int(len(arr) * 16000 / s["sr"])
                arr = scipy.signal.resample(arr, num).astype(np.float32)
            parts.append(arr)
            if j < n_concat - 1:
                parts.append(silence_05s)
            texts.append(s["text"])
        combined = np.concatenate(parts)
        if trim_s is not None:
            combined = combined[:int(trim_s * 16000)]
        duration = len(combined) / 16000
        inputs.append({
            "wav_bytes": to_wav_bytes(combined, 16000),
            "duration_s": round(duration, 2),
            "texts": texts,
        })
        i += n_concat

    return inputs


# ── WebSocket pipeline measurement ────────────────────────────────────────────

async def measure_one(ws, wav_bytes: bytes) -> dict:
    """
    Send one audio sample via WebSocket and collect timing events.
    Returns dict with TTFT, TTFA, server metrics, n_chunks.
    """
    result = {
        "ttft_ms":     float("nan"),
        "ttfa_ms":     float("nan"),
        "asr_s":       float("nan"),
        "llm_s":       float("nan"),
        "tts_s":       float("nan"),
        "total_s":     float("nan"),
        "tok_s":       float("nan"),
        "n_chunks":    0,
        "transcript":  "",
        "error":       None,
    }

    b64 = arr_to_b64(wav_bytes)
    t_send = time.perf_counter()

    await ws.send(json.dumps({"type": "audio", "data": b64}))

    first_token_seen  = False
    first_audio_seen  = False

    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        now = time.perf_counter()
        elapsed_ms = (now - t_send) * 1000

        mtype = msg.get("type")

        if mtype == "transcript":
            result["asr_s"]      = msg.get("asr_s", float("nan"))
            result["transcript"] = msg.get("text", "")

        elif mtype == "token" and not first_token_seen:
            first_token_seen   = True
            result["ttft_ms"]  = round(elapsed_ms, 1)

        elif mtype == "audio_chunk":
            result["n_chunks"] += 1
            if not first_audio_seen and msg.get("idx", 0) == 0:
                first_audio_seen  = True
                result["ttfa_ms"] = round(elapsed_ms, 1)

        elif mtype == "done":
            m = msg.get("metrics", {})
            result["llm_s"]   = m.get("llm_s",   float("nan"))
            result["tts_s"]   = m.get("tts_s",   float("nan"))
            result["total_s"] = m.get("total_s", float("nan"))
            result["tok_s"]   = m.get("tok_s",   float("nan"))
            result["asr_s"]   = m.get("asr_s",   result["asr_s"])
            break

        elif mtype == "error":
            result["error"] = msg.get("msg", "unknown error")
            break

    return result


async def run_tier(tier_name: str, inputs: list[dict], rows: list[dict]):
    """Run all samples for one tier, appending results to rows."""
    print(f"\n── Tier: {tier_name} ({len(inputs)} samples) ──────────────────")

    async with websockets.connect(WS_URL, max_size=64 * 1024 * 1024) as ws:
        for i, inp in enumerate(inputs):
            duration = inp["duration_s"]
            print(f"  [{i+1:02d}/{len(inputs)}] duration={duration:.1f}s  ", end="", flush=True)

            try:
                result = await asyncio.wait_for(
                    measure_one(ws, inp["wav_bytes"]),
                    timeout=TIMEOUT_S,
                )
                row = {
                    "tier":       tier_name,
                    "sample_idx": i,
                    "duration_s": duration,
                    **result,
                }
                rows.append(row)
                print(
                    f"TTFT={result['ttft_ms']:.0f}ms  "
                    f"TTFA={result['ttfa_ms']:.0f}ms  "
                    f"total={result['total_s']:.2f}s  "
                    f"chunks={result['n_chunks']}"
                )
            except asyncio.TimeoutError:
                print("TIMEOUT")
                rows.append({
                    "tier": tier_name, "sample_idx": i, "duration_s": duration,
                    "ttft_ms": float("nan"), "ttfa_ms": float("nan"),
                    "asr_s": float("nan"), "llm_s": float("nan"),
                    "tts_s": float("nan"), "total_s": float("nan"),
                    "tok_s": float("nan"), "n_chunks": 0,
                    "transcript": "", "error": "timeout",
                })
            except Exception as e:
                print(f"ERROR: {e}")
                rows.append({
                    "tier": tier_name, "sample_idx": i, "duration_s": duration,
                    "ttft_ms": float("nan"), "ttfa_ms": float("nan"),
                    "asr_s": float("nan"), "llm_s": float("nan"),
                    "tts_s": float("nan"), "total_s": float("nan"),
                    "tok_s": float("nan"), "n_chunks": 0,
                    "transcript": "", "error": str(e),
                })

            if i < len(inputs) - 1:
                await asyncio.sleep(INTER_SAMPLE_S)


async def main_async(tiers):
    # Load samples — need enough for all selected tiers
    max_concat = max(n for _, n, _, _ in tiers)
    n_needed   = SAMPLES_PER_TIER * max_concat + 10  # +10 buffer
    samples    = load_librispeech_samples(n_needed)

    all_rows: list[dict] = []

    for tier_name, n_concat, target_s, trim_s in tiers:
        inputs = build_tier_inputs(samples, n_concat, trim_s)
        actual_dur = np.mean([x["duration_s"] for x in inputs]) if inputs else 0
        print(f"\n[Bench] Tier '{tier_name}': {len(inputs)} inputs, "
              f"avg duration={actual_dur:.1f}s (target ~{target_s}s)")
        await run_tier(tier_name, inputs, all_rows)

    return all_rows


# ── Statistics & output ───────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["ttft_ms", "ttfa_ms", "asr_s", "llm_s", "tts_s", "total_s", "tok_s", "n_chunks"]
    rows = []
    for tier, group in df.groupby("tier", sort=False):
        row = {
            "tier":        tier,
            "n_samples":   len(group),
            "avg_dur_s":   round(group["duration_s"].mean(), 2),
        }
        for m in metrics:
            vals = group[m].dropna()
            if len(vals) == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"]  = float("nan")
                row[f"{m}_p50"]  = float("nan")
                row[f"{m}_p95"]  = float("nan")
            else:
                row[f"{m}_mean"] = round(vals.mean(), 1)
                row[f"{m}_std"]  = round(vals.std(),  1)
                row[f"{m}_p50"]  = round(np.percentile(vals, 50), 1)
                row[f"{m}_p95"]  = round(np.percentile(vals, 95), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  End-to-End Streaming Benchmark Summary")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(f"\nTier: {row['tier']}  (n={int(row['n_samples'])}, avg_dur={row['avg_dur_s']}s)")
        print(f"  {'Metric':<12}  {'Mean':>8}  {'Std':>8}  {'p50':>8}  {'p95':>8}")
        print(f"  {'-'*52}")
        for m, unit in [
            ("ttft_ms", "ms"), ("ttfa_ms", "ms"),
            ("asr_s", "s"),    ("llm_s", "s"),
            ("tts_s", "s"),    ("total_s", "s"),
            ("tok_s", "tok/s"),("n_chunks", ""),
        ]:
            mean = row.get(f"{m}_mean", float("nan"))
            std  = row.get(f"{m}_std",  float("nan"))
            p50  = row.get(f"{m}_p50",  float("nan"))
            p95  = row.get(f"{m}_p95",  float("nan"))
            label = f"{m} ({unit})" if unit else m
            print(f"  {label:<18}  {mean:>8.1f}  {std:>8.1f}  {p50:>8.1f}  {p95:>8.1f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Streaming Benchmark")
    parser.add_argument(
        "--tiers", nargs="+", choices=ALL_TIER_NAMES, default=ALL_TIER_NAMES,
        metavar="TIER",
        help=f"Which tiers to run. Choices: {ALL_TIER_NAMES}. Default: all.",
    )
    args = parser.parse_args()

    tiers = [t for t in ALL_TIERS if t[0] in args.tiers]
    # Output filename suffix: use tier names joined if not all tiers
    if set(args.tiers) == set(ALL_TIER_NAMES):
        suffix = ""
    else:
        suffix = "_" + "_".join(args.tiers)

    print("[Bench] End-to-End Streaming Benchmark")
    print(f"[Bench] WebSocket: {WS_URL}")
    print(f"[Bench] Samples per tier: {SAMPLES_PER_TIER}")
    print(f"[Bench] Running tiers: {[t[0] for t in tiers]}")

    # Quick connectivity check
    import requests
    try:
        for port, name in [(8001,"ASR"),(8002,"LLM"),(8003,"TTS")]:
            r = requests.get(f"http://localhost:{port}/health", timeout=3)
            print(f"[Bench] {name} :  {r.json().get('status','?')}")
        r = requests.get("http://localhost:7860/health", timeout=10)
        agg = r.json()
        all_ok = all(v.get("status") == "ok" for v in agg.values())
        print(f"[Bench] Orchestrator /health:  {'ok' if all_ok else 'partial'}")
    except Exception as e:
        print(f"[Bench] ERROR: server not reachable — {e}")
        print("[Bench] Start all servers with: python run.py")
        sys.exit(1)

    rows = asyncio.run(main_async(tiers))

    if not rows:
        print("[Bench] No results collected.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    summary = compute_summary(df)
    print_summary(summary)

    # Save outputs — separate files when running a subset of tiers
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    raw_path     = OUT_TABLES / f"e2e_streaming_benchmark{suffix}.csv"
    summary_path = OUT_TABLES / f"e2e_streaming_summary{suffix}.csv"
    df.to_csv(raw_path,     index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\n[Bench] Raw results   → {raw_path}")
    print(f"[Bench] Summary table → {summary_path}")


if __name__ == "__main__":
    main()
