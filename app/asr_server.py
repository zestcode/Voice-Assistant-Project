"""
ASR microservice — run in 'moonshine' conda env
  conda activate moonshine
  python app/asr_server.py

POST /transcribe  multipart: file=<wav bytes>  → {"text": str, "latency": float}
GET  /health                                   → {"status": "ok"}
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

BASE     = Path(__file__).resolve().parent.parent
# large-v3-turbo: same accuracy as large-v3, 3× faster, 1.5GB
# Falls back to local path if already downloaded; otherwise downloads automatically
ASR_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = "int8_float16" if DEVICE == "cuda" else "int8"
PORT     = 8001

app   = FastAPI(title="ASR Server")
model: WhisperModel | None = None


@app.on_event("startup")
def load_model():
    global model
    print(f"[ASR] Loading faster-whisper-large-v3-turbo on {DEVICE.upper()} ...")
    print("[ASR] First run auto-downloads ~1.5 GB from HuggingFace ...")
    model = WhisperModel(ASR_MODEL, device=DEVICE, compute_type=DTYPE)
    # Warm-up: compile CUDA kernels so first real request is fast
    _silence = np.zeros(16000, dtype=np.float32)
    list(model.transcribe(_silence, beam_size=1)[0])
    print("[ASR] Ready")


@app.get("/health")
def health():
    return {"status": "ok", "model": "faster-whisper-large-v3-turbo", "device": DEVICE}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    raw  = await audio.read()
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)           # stereo → mono
    if arr.max() > 1.0:
        arr = arr / 32768.0              # int16 range guard

    # Reject audio that is too short to contain real speech
    duration = len(arr) / 16000
    if duration < 0.5:
        return {"text": "", "latency": 0.0}

    t0 = time.perf_counter()
    segs, info = model.transcribe(
        arr,
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        no_speech_threshold=0.6,          # discard segment if no-speech prob > 0.6
        log_prob_threshold=-1.0,          # discard segment if avg log-prob < -1.0
        compression_ratio_threshold=2.4,  # discard segment if compression ratio > 2.4 (repetitive hallucination)
        condition_on_previous_text=False, # prevents hallucinations from compounding across segments
    )
    # Filter out segments where the model itself is uncertain
    parts = []
    for s in segs:
        if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0:
            parts.append(s.text)
    text    = " ".join(parts).strip()
    latency = round(time.perf_counter() - t0, 3)

    return {"text": text, "latency": latency}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", timeout_keep_alive=75)
