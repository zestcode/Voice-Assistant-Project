"""
TTS microservice (F5-TTS voice cloning) — run in 'gptsovits' conda env
  conda activate gptsovits
  python app/tts_server.py

POST /synthesize  multipart: text=str, ref_text=str, ref_audio=<wav bytes (optional)>
                  → WAV bytes  +  headers: X-Latency, X-SampleRate
GET  /health      → {"status": "ok"}
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn

# torchaudio 2.11 requires torchcodec (FFmpeg DLLs) on Windows — patch before f5_tts loads
def _sf_load(path, frame_offset=0, num_frames=-1, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(data.T)
    if frame_offset > 0:
        tensor = tensor[:, frame_offset:]
    if num_frames > 0:
        tensor = tensor[:, :num_frames]
    return tensor, sr

torchaudio.load = _sf_load
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, JSONResponse

BASE      = Path(__file__).resolve().parent.parent
REF_AUDIO = BASE / "data" / "ref_voice" / "ref.wav"
REF_TEXT  = "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
PORT      = 8003

app = FastAPI(title="TTS Server")
tts = None


@app.on_event("startup")
def load_model():
    global tts
    from f5_tts.api import F5TTS
    print(f"[TTS] Loading F5-TTS on {DEVICE.upper()} ...")
    print("[TTS] First run auto-downloads ~1.3 GB from HuggingFace ...")
    tts = F5TTS(model="F5TTS_v1_Base", device=DEVICE)
    # Warm-up: compile CUDA kernels with a short dummy inference
    if REF_AUDIO.exists():
        try:
            tts.infer(
                ref_file=str(REF_AUDIO),
                ref_text=REF_TEXT,
                gen_text="Warmup.",
                nfe_step=4,
                remove_silence=False,
            )
        except Exception:
            pass
    print("[TTS] Ready")


@app.get("/health")
def health():
    return {"status": "ok", "model": "f5-tts", "device": DEVICE}


@app.post("/synthesize")
async def synthesize(
    text:      str           = Form(...),
    ref_text:  str           = Form(REF_TEXT),
    ref_audio: UploadFile | None = File(default=None),
):
    # Resolve reference audio
    if ref_audio is not None:
        raw_bytes = await ref_audio.read()
        ref_path  = io.BytesIO(raw_bytes)   # f5-tts accepts file-like objects
    else:
        ref_path = str(REF_AUDIO)

    t0 = time.perf_counter()
    wav, sr, _ = tts.infer(
        ref_file=ref_path,
        ref_text=ref_text,
        gen_text=text,
        nfe_step=16,          # 16 vs default 32 — halves diffusion steps, minimal quality loss
        remove_silence=False, # skip VAD post-processing
    )
    latency = round(time.perf_counter() - t0, 3)

    # Encode output as WAV bytes
    buf = io.BytesIO()
    sf.write(buf, np.array(wav, dtype=np.float32), sr, format="WAV")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"X-Latency": str(latency), "X-SampleRate": str(sr)},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", timeout_keep_alive=75)
