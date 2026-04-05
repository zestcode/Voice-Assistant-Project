"""
Export benchmark input samples as WAV files for playback.

Each tier exports N_SAMPLES samples (default 20, matching benchmark).
- xshort: each LibriSpeech sample trimmed to 0.8s
- short:  1 sample per file (~3-5s)
- medium: 3 samples concatenated per file (~15s)
- long:   6 samples concatenated per file (~30s)

Run in voiceui env:
  python scripts/export_bench_samples.py               # all tiers, 20 samples each
  python scripts/export_bench_samples.py --tiers xshort
  python scripts/export_bench_samples.py --tiers xshort short --n 5
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ASR_BENCH_PATH

OUT = Path(__file__).resolve().parent.parent / "outputs" / "bench_samples"
OUT.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 20  # match benchmark

# (tier_name, n_concat, trim_s)
TIERS = [
    ("xshort", 1, 0.8),
    ("short",  1, None),
    ("medium", 3, None),
    ("long",   6, None),
]
TIER_NAMES = [t[0] for t in TIERS]


def load_samples(n):
    from datasets import load_from_disk, Audio
    ds = load_from_disk(str(ASR_BENCH_PATH))
    ds = ds.cast_column("audio", Audio(decode=False))
    samples = []
    for item in ds:
        audio = item["audio"]
        raw = audio.get("bytes") or open(audio["path"], "rb").read()
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        samples.append({"array": arr, "sr": sr, "text": item.get("text", "")})
        if len(samples) >= n:
            break
    return samples


def resample_16k(arr, sr):
    if sr == 16000:
        return arr
    import scipy.signal
    num = int(len(arr) * 16000 / sr)
    return scipy.signal.resample(arr, num).astype(np.float32)


silence = np.zeros(8000, dtype=np.float32)  # 0.5s at 16kHz


def build_inputs(samples, n_concat, trim_s, n_out):
    """Build n_out audio arrays by sliding a window of n_concat samples."""
    inputs = []
    for i in range(n_out):
        group = samples[i * n_concat: i * n_concat + n_concat]
        if len(group) < n_concat:
            break
        parts = []
        for j, s in enumerate(group):
            arr = resample_16k(s["array"], s["sr"])
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            parts.append(arr)
            if j < n_concat - 1:
                parts.append(silence)
        combined = np.concatenate(parts)
        if trim_s is not None:
            combined = combined[:int(trim_s * 16000)]
        texts = [s["text"] for s in group]
        inputs.append((combined, texts))
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Export benchmark input samples as WAV.")
    parser.add_argument(
        "--tiers", nargs="+", choices=TIER_NAMES, default=TIER_NAMES,
        metavar="TIER",
        help=f"Which tiers to export. Choices: {TIER_NAMES}. Default: all.",
    )
    parser.add_argument(
        "--n", type=int, default=N_SAMPLES,
        help=f"Number of samples per tier (default: {N_SAMPLES}).",
    )
    args = parser.parse_args()

    selected = [(name, nc, trim) for name, nc, trim in TIERS if name in args.tiers]
    max_concat = max(nc for _, nc, _ in selected)
    n_needed = max_concat * args.n + max_concat

    print(f"Loading up to {n_needed} LibriSpeech samples...")
    samples = load_samples(n_needed)

    for tier_name, n_concat, trim_s in selected:
        tier_dir = OUT / tier_name
        tier_dir.mkdir(exist_ok=True)
        inputs = build_inputs(samples, n_concat, trim_s, args.n)
        print(f"[{tier_name}] exporting {len(inputs)} samples...")
        for idx, (audio, texts) in enumerate(inputs):
            duration = len(audio) / 16000
            out_path = tier_dir / f"sample_{idx:03d}.wav"
            sf.write(str(out_path), audio, 16000, subtype="PCM_16")
            text_preview = texts[0][:60]
            print(f"  [{idx:02d}] {duration:.2f}s  {text_preview}")

    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
