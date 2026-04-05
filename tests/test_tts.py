"""
TTS quick test — run in 'gptsovits' conda env
  python tests/test_tts.py

Prints RESULT_JSON: {...} on last stdout line for test.py to capture.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from config import REF_AUDIO, REF_TEXT

import numpy as np
import soundfile as sf
import torch
import torchaudio

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
GEN_TEXT = "This is a voice cloning test. The system synthesizes speech that matches the reference speaker."
OUT_WAV  = ROOT / "outputs" / "tts_audio" / "quick_test_f5.wav"

# torchaudio 2.11 hardcodes torchcodec which requires FFmpeg DLLs on Windows.
# Patch load() to use soundfile before f5_tts imports torchaudio internally.
def _sf_load(path, frame_offset=0, num_frames=-1, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(data.T)           # (channels, samples)
    if frame_offset > 0:
        tensor = tensor[:, frame_offset:]
    if num_frames > 0:
        tensor = tensor[:, :num_frames]
    return tensor, sr

torchaudio.load = _sf_load


def main():
    print(f"[TTS] Device : {DEVICE.upper()}")
    print(f"[TTS] Loading F5-TTS ...")
    from f5_tts.api import F5TTS
    tts = F5TTS(model="F5TTS_v1_Base", device=DEVICE)

    print(f"[TTS] Reference : {REF_AUDIO.name}")
    print(f"[TTS] Generating: {GEN_TEXT}")

    t0 = time.perf_counter()
    wav, sr, _ = tts.infer(
        ref_file=str(REF_AUDIO),
        ref_text=REF_TEXT,
        gen_text=GEN_TEXT,
        remove_silence=True,
    )
    latency = time.perf_counter() - t0

    audio = np.array(wav, dtype=np.float32)
    dur   = len(audio) / sr
    rtf   = latency / dur   # <1.0 means faster than real-time

    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT_WAV), audio, sr)

    print(f"[TTS] Output  : {dur:.2f}s audio  sr={sr}")
    print(f"[TTS] Latency : {latency:.3f}s   RTF: {rtf:.2f}x  ({'faster' if rtf < 1 else 'slower'} than real-time)")
    print(f"[TTS] Saved   : {OUT_WAV}")

    result = {
        "model":      "f5-tts",
        "device":     DEVICE,
        "latency_s":  round(latency, 3),
        "audio_dur_s": round(dur, 3),
        "rtf":        round(rtf, 3),
        "sample_rate": sr,
        "output_wav": str(OUT_WAV),
    }
    print(f"RESULT_JSON: {json.dumps(result)}")


if __name__ == "__main__":
    main()
