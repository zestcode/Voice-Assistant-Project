"""
ASR quick test — run in 'moonshine' conda env
  python tests/test_asr.py

Prints RESULT_JSON: {...} on last stdout line for test.py to capture.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from config import ASR1_PATH, REF_AUDIO, REF_TEXT

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = "float16" if DEVICE == "cuda" else "int8"


def main():
    print(f"[ASR] Device : {DEVICE.upper()}")
    print(f"[ASR] Loading Faster-Whisper-Small ...")
    model = WhisperModel(str(ASR1_PATH), device=DEVICE, compute_type=DTYPE)

    audio, sr = sf.read(str(REF_AUDIO), dtype="float32")
    dur = len(audio) / sr
    print(f"[ASR] Input  : {REF_AUDIO.name}  ({dur:.2f}s audio)")

    # Warm-up (first inference always slower due to CUDA kernel compilation)
    model.transcribe(audio[:sr], beam_size=1)  # 1 second warm-up

    t0 = time.perf_counter()
    segs, _ = model.transcribe(audio, beam_size=5)
    transcript = " ".join(s.text for s in segs).strip()
    latency = time.perf_counter() - t0
    rtfx    = dur / latency

    print(f"[ASR] Output : {transcript}")
    print(f"[ASR] Latency: {latency:.3f}s   RTFx: {rtfx:.1f}x")

    result = {
        "model":      "faster-whisper-small",
        "device":     DEVICE,
        "latency_s":  round(latency, 3),
        "rtfx":       round(rtfx, 1),
        "transcript": transcript,
        "expected":   REF_TEXT,
        "match":      REF_TEXT.lower() in transcript.lower(),
    }
    print(f"RESULT_JSON: {json.dumps(result)}")


if __name__ == "__main__":
    main()
