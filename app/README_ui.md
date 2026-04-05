# Voice AI Assistant — Web UI

**Primary entry point:** `app/orchestrator.py` + `app/static/index.html`
**Conda env:** `voiceui`
**Port:** `7860`  →  open `http://localhost:7860` in browser

The orchestrator serves `index.html` as a static file and exposes a WebSocket endpoint at
`ws://localhost:7860/ws`. The browser communicates with the orchestrator over a single persistent
WebSocket; the orchestrator calls the three backend services (ASR, LLM, TTS) over HTTP/SSE.
No models are loaded by the orchestrator — all inference happens in the backend servers.

> **Legacy:** `app/voice_assistant.py` is a Gradio-based UI retained for reference.
> It uses the blocking HTTP `/pipeline` endpoint and does not support streaming or interrupt.
> The WebSocket UI (`orchestrator.py` + `index.html`) supersedes it for all active use.

---

## Prerequisites

All three microservices must be running before launching the orchestrator.

| Service | Env | Command | Port |
|---|---|---|---|
| ASR | `moonshine` | `python app/asr_server.py` | 8001 |
| LLM | `llm` | `python app/llm_server.py` | 8002 |
| TTS | `gptsovits` | `python app/tts_server.py` | 8003 |

---

## Environment Setup

```bash
conda activate voiceui
pip install fastapi uvicorn requests numpy soundfile scipy websockets
```

---

## Run

```bash
# Four separate terminals

# Terminal 1
conda activate moonshine && python app/asr_server.py

# Terminal 2
conda activate llm && python app/llm_server.py

# Terminal 3
conda activate gptsovits && python app/tts_server.py

# Terminal 4
conda activate voiceui && python app/orchestrator.py
```

Or use the one-click launcher:
```bash
app/start_servers.bat
```

Then open `http://localhost:7860` in a browser (Chrome 66+, Firefox 76+, Edge 79+).

---

## UI Features

### Recording Modes

- **Push-to-talk (manual):** Press and hold the record button; release to submit. The full buffer is
  sent to ASR. More robust against background noise; preferred in noisy environments.
- **Auto-record (VAD):** Microphone stays open. Voice activity detection (RMS energy threshold +
  400 ms hold-time debounce) triggers recording automatically. Silence frames and recordings shorter
  than 0.5 s are discarded before forwarding to ASR, suppressing hallucinated transcripts from
  ambient noise or inter-word gaps.

### Real-Time Pipeline Display

Each turn streams four stages back to the browser:

1. **Transcript** — displayed immediately after ASR completes, before LLM starts
2. **LLM response** — tokens appear word-by-word as the model generates
3. **Audio playback** — first sentence plays while the LLM is still generating sentences 2, 3, …
   (sentence-level streaming via TTS chunking)
4. **Latency breakdown** — ASR / LLM / TTS / Total shown after the `done` message

### Interrupt

A **Stop** button (and automatic VAD trigger during playback) sends `{"type": "interrupt"}` over
the WebSocket. The orchestrator cancels the current pipeline task; the LLM and TTS services
discard queued work. The mic is immediately re-enabled for the next turn.

### Voice Cloning

Upload a 3–15 s WAV clip to replace the default reference voice. The clip is sent to the TTS
service as the speaker-conditioning reference for zero-shot voice cloning.

---

## Pipeline Data Flow

```
Browser ──WS── Orchestrator ──HTTP──  ASR :8001  (Faster-Whisper-Large-V3-Turbo)
                    │        ──SSE──  LLM :8002  (Qwen2.5-3B-Instruct Q4_K_M)
                    │        ──HTTP── TTS :8003  (F5-TTS voice cloning)
                    │
                    ▼  messages pushed to browser:
  {type:"transcript"}          — after ASR completes
  {type:"token"}               — per LLM token (real-time display)
  {type:"audio_chunk", idx:N}  — per-sentence TTS result (base64 WAV)
  {type:"done"}                — pipeline complete + per-stage metrics
  {type:"interrupted"}         — interrupt confirmed
  {type:"error"}               — stage failure
```

---

## Typical Latency (34-turn session log, CUDA)

| Stage | Mean | Median | Range |
|---|---|---|---|
| ASR | 0.63 s | 0.34 s | 0.15–4.37 s |
| LLM | 0.11 s | 0.10 s | 0.04–0.27 s |
| TTS (per sentence chunk) | 0.94 s | 0.80 s | 0.47–3.18 s |
| **Inference sum** | **1.80 s** | **1.61 s** | **0.77–4.98 s** |

Time-to-first-audio (TTFA) from the streaming benchmark: ~3.0 s for short inputs (~0.8 s audio),
~3.9 s for full-sentence inputs (~8 s audio).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Browser shows "WebSocket error" | Orchestrator not running | `conda activate voiceui && python app/orchestrator.py` |
| `ASR OFFLINE` | asr_server.py not started | `conda activate moonshine && python app/asr_server.py` |
| `LLM OFFLINE` | llm_server.py not started | `conda activate llm && python app/llm_server.py` |
| `TTS OFFLINE` | tts_server.py not started | `conda activate gptsovits && python app/tts_server.py` |
| `(no speech detected)` | Audio too short / too quiet | Speak for at least 0.5 s; check mic gain |
| Hallucinated transcript in VAD mode | Silence below energy threshold not filtered | Check RMS threshold in `asr_server.py`; use push-to-talk mode as workaround |
| TTS outputs wrong voice | ref_audio not uploaded; default `ref.wav` used | Upload a custom reference WAV via the voice clone panel |
| Audio chunks play out of order | Browser `audioQueue` idx desync after interrupt | Reload the page to reset queue state |
