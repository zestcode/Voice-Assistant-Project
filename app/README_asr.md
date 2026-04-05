# ASR Microservice

**Model:** `faster-whisper-large-v3-turbo` (auto-downloaded from HuggingFace, ~1.5 GB)
**Conda env:** `moonshine`
**Port:** `8001`

---

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `POST` | `/transcribe` | multipart: `file=<wav bytes>` | `{"text": str, "latency": float}` |
| `GET` | `/health` | — | `{"status": "ok", "model": "...", "device": "cuda/cpu"}` |

---

## Environment Setup

```bash
conda activate moonshine
pip install faster-whisper soundfile numpy torch
```

> **First run** downloads `deepdml/faster-whisper-large-v3-turbo-ct2` (~1.5 GB) from
> HuggingFace into the default cache (`~/.cache/huggingface/`).
> Subsequent starts load from cache with no download.

---

## Run

```bash
conda activate moonshine
cd <project_root>
python app/asr_server.py
```

Expected startup output:
```
[ASR] Loading faster-whisper-large-v3-turbo on CUDA ...
[ASR] First run auto-downloads ~1.5 GB from HuggingFace ...
[ASR] Ready
INFO:     Uvicorn running on http://0.0.0.0:8001
```

---

## Implementation Notes

- **Compute type:** `int8_float16` on CUDA, `int8` on CPU
- **VAD filter** (`vad_filter=True`): silences shorter than 300 ms are skipped
- **Hallucination guards:** segments with `no_speech_prob > 0.6` or `avg_logprob < -1.0` are discarded
- **Silence guard:** recordings shorter than 0.5 s return `{"text": ""}` without calling the model
- **Warm-up:** a silent dummy inference runs at startup to compile CUDA kernels before the first real request

---

## Quick Test

```bash
# Health check
curl http://localhost:8001/health

# Transcribe a WAV file
curl -X POST http://localhost:8001/transcribe \
     -F "audio=@data/ref_voice/ref.wav" \
     -s | python -m json.tool
```
