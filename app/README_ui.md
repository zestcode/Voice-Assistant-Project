# Voice AI Assistant — Gradio UI

**File:** `app/voice_assistant.py`
**Port:** `7860`  →  open `http://localhost:7860` in browser

The UI orchestrates all three microservices. It does **not** load any models itself —
all inference happens inside the three backend servers.

---

## Prerequisites

All three microservices must be running before launching the UI.

| Service | Env | Command | Port |
|---|---|---|---|
| ASR | `moonshine` | `python app/asr_server.py` | 8001 |
| LLM | `llm` | `python app/llm_server.py` | 8002 |
| TTS | `gptsovits` | `python app/tts_server.py` | 8003 |

---

## Environment Setup

The UI can run in **any** conda environment — it only needs:

```bash
pip install gradio requests numpy soundfile scipy
```

Tested with `gradio >= 4.0`.

---

## Run

```bash
# Four separate terminals, one per service + UI

# Terminal 1
conda activate moonshine
python app/asr_server.py

# Terminal 2
conda activate llm
python app/llm_server.py

# Terminal 3
conda activate gptsovits
python app/tts_server.py

# Terminal 4  (any env with gradio)
python app/voice_assistant.py
```

Then open `http://localhost:7860` in a browser.

---

## UI Features

### Main Pipeline

1. **Record** audio from the microphone
2. Click **Process**
3. The UI displays:
   - **ASR transcript** — what you said
   - **LLM response** — the assistant's reply
   - **TTS audio** — synthesized speech (autoplays)
   - **Latency breakdown** — per-stage and total time

### Voice Clone Panel

- Upload any **3–15 s WAV** to replace the default reference voice
- Optionally type the transcript of the reference clip in the text box
- If left empty, defaults to `data/ref_voice/ref.wav`

### Server Status

- **Refresh server status** button pings all three `/health` endpoints
- Shows `OK` / `OFFLINE` per service

### README / Report Export

- **Generate README / Report** button outputs a Markdown session summary
  with average latency, per-turn log, and architecture overview
- Saves to `outputs/voice_assistant_report.md`

---

## Pipeline Data Flow

```
Microphone (WAV, 16kHz mono)
    │
    ▼
POST /transcribe  →  ASR :8001  (faster-whisper-large-v3-turbo)
    │  transcript (str)
    ▼
POST /generate    →  LLM :8002  (Qwen2.5-1.5B Q4_K_M)
    │  response (str)
    ▼
POST /synthesize  →  TTS :8003  (F5-TTS voice cloning)
    │  WAV bytes
    ▼
Gradio audio player (autoplay)
```

---

## Latency Display (example)

| Stage | Model | Latency |
|---|---|---|
| ASR | Faster-Whisper-Large-V3-Turbo | ~0.20 s |
| LLM | Qwen2.5-1.5B Q4\_K\_M | ~0.28 s |
| TTS | F5-TTS (voice clone) | ~1.30 s |
| **Total** | — | **~1.80 s** |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ASR OFFLINE` in status | asr_server.py not started | `conda activate moonshine && python app/asr_server.py` |
| `LLM OFFLINE` | llm_server.py not started | `conda activate llm && python app/llm_server.py` |
| `TTS OFFLINE` | tts_server.py not started | `conda activate gptsovits && python app/tts_server.py` |
| `(no speech detected)` after recording | Audio too short or too quiet | Speak for at least 1 s; check mic gain |
| TTS outputs wrong voice | ref_audio field empty, default ref.wav used | Upload a custom reference WAV |
| Gradio audio playback broken | Brotli middleware bug in Gradio 6.x | Already patched in `voice_assistant.py` — no action needed |
