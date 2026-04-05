# TTS Microservice

**Model:** `F5-TTS` (F5TTS\_v1\_Base, auto-downloaded from HuggingFace, ~1.3 GB)
**Conda env:** `gptsovits`
**Port:** `8003`

---

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `POST` | `/synthesize` | multipart form: `text` (str), `ref_text` (str, optional), `ref_audio` (WAV file, optional) | WAV bytes + headers `X-Latency`, `X-SampleRate` |
| `GET` | `/health` | — | `{"status": "ok", "model": "f5-tts", "device": "cuda/cpu"}` |

### `/synthesize` fields

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Text to synthesize |
| `ref_text` | string | no | Transcript of the reference audio clip |
| `ref_audio` | WAV file | no | Reference speaker audio (3–15 s). If omitted, uses `data/ref_voice/ref.wav` |

---

## Environment Setup

```bash
conda activate gptsovits
pip install f5-tts soundfile numpy torch torchaudio fastapi uvicorn
```

> **First run** downloads `F5TTS_v1_Base` weights (~1.3 GB) from HuggingFace into
> `~/.cache/huggingface/`. Subsequent starts load from cache.

### Windows / torchaudio fix

On Windows, `torchaudio >= 2.11` requires `torchcodec` (FFmpeg DLLs) which may not be present.
The server patches `torchaudio.load` at startup to use `soundfile` instead — no manual action needed.

---

## Reference Voice

Default reference audio: `data/ref_voice/ref.wav`
Default transcript: `"CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"` (set in `tts_server.py`)

To use a custom voice at runtime, upload a WAV file via the `/synthesize` endpoint's `ref_audio` field
(or through the Gradio UI voice clone panel).

---

## Run

```bash
conda activate gptsovits
cd <project_root>
python app/tts_server.py
```

Expected startup output:
```
[TTS] Loading F5-TTS on CUDA ...
[TTS] First run auto-downloads ~1.3 GB from HuggingFace ...
[TTS] Ready
INFO:     Uvicorn running on http://0.0.0.0:8003
```

---

## Implementation Notes

- **nfe_step:** `16` (half of the default 32 diffusion steps) — halves synthesis time with minimal quality loss
- **remove_silence:** `False` — VAD post-processing skipped to reduce latency
- **Warm-up:** a short dummy inference runs at startup to compile CUDA kernels
- **Latency:** ~1.3 s/sample on RTX-class GPU (RTF ≈ 0.65); dominates ~88% of end-to-end pipeline latency

---

## Quick Test

```bash
# Health check
curl http://localhost:8003/health

# Synthesize with default reference voice
curl -X POST http://localhost:8003/synthesize \
     -F "text=Hello, this is a test of the voice cloning system." \
     -o test_output.wav

# Synthesize with custom reference audio
curl -X POST http://localhost:8003/synthesize \
     -F "text=Hello, this is a test." \
     -F "ref_text=CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS" \
     -F "ref_audio=@data/ref_voice/ref.wav" \
     -o test_output.wav
```
