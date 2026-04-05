"""
Voice AI Assistant — Gradio UI
Orchestrates three microservices via HTTP:
  ASR  → localhost:8001  (moonshine env)
  LLM  → localhost:8002  (llm env)
  TTS  → localhost:8003  (gptsovits env)

Run this file in ANY env that has: gradio requests numpy soundfile
  pip install gradio requests numpy soundfile
  python app/voice_assistant.py
"""
import io
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import gradio as gr

# Gradio 6.x brotli middleware miscalculates Content-Length → patch to passthrough
try:
    import gradio.brotli_middleware as _bm
    class _NoBrotli:
        def __init__(self, app, *a, **kw): self.app = app
        async def __call__(self, scope, receive, send): await self.app(scope, receive, send)
    _bm.BrotliMiddleware = _NoBrotli
except Exception:
    pass

ASR_URL = "http://localhost:8001"
LLM_URL = "http://localhost:8002"
TTS_URL = "http://localhost:8003"

BASE      = Path(__file__).resolve().parent.parent
REF_AUDIO = BASE / "data" / "ref_voice" / "ref.wav"
REF_TEXT  = "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"
OUT_DIR   = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_session_log: list[dict] = []

TIMEOUT_ASR = 30
TIMEOUT_LLM = 60
TIMEOUT_TTS = 60


# ── Helpers

def _audio_to_wav_bytes(audio_tuple) -> bytes:
    """Convert Gradio numpy audio tuple → 16kHz mono float32 WAV bytes."""
    sr, arr = audio_tuple

    # Gradio returns int16 from microphone — normalise to float32 [-1, 1]
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        arr = arr.astype(np.float32) / 2147483648.0
    else:
        arr = arr.astype(np.float32)

    # Stereo → mono
    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    # Resample to 16kHz (Whisper works best at 16kHz)
    if sr != 16000:
        import scipy.signal
        num_samples = int(len(arr) * 16000 / sr)
        arr = scipy.signal.resample(arr, num_samples).astype(np.float32)
        sr = 16000

    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _check_servers() -> str:
    status = []
    for name, url in [("ASR", ASR_URL), ("LLM", LLM_URL), ("TTS", TTS_URL)]:
        try:
            requests.get(f"{url}/health", timeout=2)
            status.append(f"**{name}** OK")
        except Exception:
            status.append(f"**{name}** OFFLINE")
    return " | ".join(status)


# ── Pipeline stages

def call_asr(wav_bytes: bytes) -> tuple[str, float]:
    r = requests.post(
        f"{ASR_URL}/transcribe",
        files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
        timeout=TIMEOUT_ASR,
    )
    r.raise_for_status()
    d = r.json()
    return d["text"], d["latency"]


def call_llm(text: str) -> tuple[str, float, float]:
    r = requests.post(
        f"{LLM_URL}/generate",
        json={"prompt": text},
        timeout=TIMEOUT_LLM,
    )
    r.raise_for_status()
    d = r.json()
    return d["response"], d["latency"], d["tok_s"]


def call_tts(text: str, ref_audio_path: str | None, ref_text: str) -> tuple[np.ndarray, int, float]:
    data   = {"text": text, "ref_text": ref_text}
    files  = {}
    if ref_audio_path:
        files["ref_audio"] = ("ref.wav", open(ref_audio_path, "rb"), "audio/wav")

    r = requests.post(
        f"{TTS_URL}/synthesize",
        data=data,
        files=files if files else None,
        timeout=TIMEOUT_TTS,
    )
    r.raise_for_status()

    latency = float(r.headers.get("X-Latency", 0))
    sr      = int(r.headers.get("X-SampleRate", 24000))
    arr, _  = sf.read(io.BytesIO(r.content), dtype="float32")
    arr_i16 = (arr * 32767).clip(-32768, 32767).astype(np.int16)
    return arr_i16, sr, latency


# ── Main pipeline

def pipeline(audio_input, ref_audio_upload, ref_text_input):
    if audio_input is None:
        return "No audio recorded.", "", None, "*Record something first.*", _check_servers()

    try:
        wav_bytes = _audio_to_wav_bytes(audio_input)
    except Exception as e:
        return f"Audio error: {e}", "", None, "", _check_servers()

    # Resolve reference voice
    ref_path = ref_audio_upload if ref_audio_upload else None
    ref_txt  = ref_text_input.strip() if ref_text_input and ref_text_input.strip() else REF_TEXT

    # ASR
    try:
        transcript, asr_lat = call_asr(wav_bytes)
    except Exception as e:
        return f"ASR error: {e}", "", None, "", _check_servers()

    print(f"[Pipeline] ASR transcript: '{transcript}'")

    if not transcript:
        return "(no speech detected)", "", None, "", _check_servers()

    # LLM
    try:
        response, llm_lat, llm_tput = call_llm(transcript)
    except Exception as e:
        return transcript, f"LLM error: {e}", None, "", _check_servers()

    print(f"[Pipeline] LLM response:   '{response}'")

    # TTS
    try:
        audio_arr, sr, tts_lat = call_tts(response, ref_path, ref_txt)
    except Exception as e:
        return transcript, response, None, f"TTS error: {e}", _check_servers()

    total = asr_lat + llm_lat + tts_lat
    entry = {
        "time":    time.strftime("%H:%M:%S"),
        "input":   transcript[:50] + ("…" if len(transcript) > 50 else ""),
        "asr_s":   round(asr_lat,  3),
        "llm_s":   round(llm_lat,  3),
        "tts_s":   round(tts_lat,  3),
        "total_s": round(total,    3),
        "tok_s":   round(llm_tput, 1),
    }
    _session_log.append(entry)

    metrics = f"""### Latency Breakdown
| Stage | Model | Latency |
|-------|-------|---------|
| ASR | Faster-Whisper-Small | **{asr_lat} s** |
| LLM | Qwen2.5-1.5B Q4_K_M | **{llm_lat} s** |
| TTS | F5-TTS (voice clone) | **{tts_lat} s** |
| **Total** | — | **{total:.3f} s** |

LLM: **{llm_tput} tok/s**"""

    return transcript, response, (sr, audio_arr), metrics, _check_servers()


# ── README export

def generate_readme() -> str:
    if not _session_log:
        return "*Run at least one session first.*"

    n   = len(_session_log)
    avg = lambda k: round(float(np.mean([m[k] for m in _session_log])), 3)
    rows = "\n".join(
        f"| {m['time']} | {m['input']} | {m['asr_s']} | "
        f"{m['llm_s']} | {m['tts_s']} | {m['total_s']} | {m['tok_s']} |"
        for m in _session_log
    )
    md = f"""# Voice AI Assistant — README & Performance Report

## Architecture
```
Microphone → [ASR :8001] → [LLM :8002] → [TTS :8003] → Speaker
              moonshine       llm           gptsovits
```

## Models
| Stage | Model | Conda Env | Port |
|-------|-------|-----------|------|
| ASR | Faster-Whisper-Small (462 MB, FP16) | moonshine | 8001 |
| LLM | Qwen2.5-1.5B Q4_K_M (1.1 GB GGUF) | llm | 8002 |
| TTS | F5-TTS voice cloning (~1.3 GB) | gptsovits | 8003 |

## Average Performance ({n} session{'s' if n > 1 else ''})
| Stage | Avg Latency |
|-------|-------------|
| ASR | {avg('asr_s')} s |
| LLM | {avg('llm_s')} s |
| TTS | {avg('tts_s')} s |
| **Total** | **{avg('total_s')} s** |

LLM avg throughput: {round(float(np.mean([m['tok_s'] for m in _session_log])), 1)} tok/s

## Session Log
| Time | Input | ASR(s) | LLM(s) | TTS(s) | Total(s) | tok/s |
|------|-------|--------|--------|--------|----------|-------|
{rows}

## How to Run
```bash
# Terminal 1
conda activate moonshine && python app/asr_server.py

# Terminal 2
conda activate llm && python app/llm_server.py

# Terminal 3
conda activate gptsovits && python app/tts_server.py

# Terminal 4 (any env with gradio requests numpy soundfile)
python app/voice_assistant.py
```

## Voice Cloning
Upload any 3–15 s WAV in the UI to clone that voice.
Default voice: `data/ref_voice/ref.wav`
"""
    out = OUT_DIR / "voice_assistant_report.md"
    out.write_text(md, encoding="utf-8")
    return md + f"\n\n*Saved → `{out}`*"


# ── Gradio UI

with gr.Blocks(title="Voice AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Voice AI Assistant\n"
        "`moonshine:ASR` → `llm:Qwen2.5` → `gptsovits:F5-TTS`  |  Three-env microservice pipeline\n\n"
        "> Start the three servers first (see README). Models load on server startup."
    )

    server_status = gr.Markdown(_check_servers())
    refresh_btn   = gr.Button("Refresh server status", size="sm")
    refresh_btn.click(fn=_check_servers, outputs=[server_status])

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in   = gr.Audio(sources=["microphone"], type="numpy",
                                  label="Microphone input")
            run_btn    = gr.Button("Process", variant="primary", size="lg")
            metrics_md = gr.Markdown("*Metrics appear here after first run.*")

        with gr.Column(scale=1):
            transcript_box = gr.Textbox(label="ASR transcript", lines=2,
                                        placeholder="What you said…")
            response_box   = gr.Textbox(label="LLM response",   lines=4,
                                        placeholder="Assistant reply…")
            audio_out      = gr.Audio(label="TTS output", autoplay=True)

    gr.Markdown("### Voice Clone Settings")
    gr.Markdown(
        f"Default: `data/ref_voice/ref.wav`  \n"
        "Upload any 3–15 s WAV to clone a different voice."
    )
    with gr.Row():
        ref_upload   = gr.Audio(sources=["upload"], type="filepath",
                                label="Reference voice WAV (optional)")
        ref_text_box = gr.Textbox(label="Reference audio transcript (optional)",
                                  placeholder="Spoken words in the reference WAV…",
                                  lines=2)

    run_btn.click(
        fn=pipeline,
        inputs=[audio_in, ref_upload, ref_text_box],
        outputs=[transcript_box, response_box, audio_out, metrics_md, server_status],
    )

    gr.Markdown("---")
    readme_btn = gr.Button("Generate README / Report", variant="secondary")
    readme_out = gr.Markdown()
    readme_btn.click(fn=generate_readme, outputs=[readme_out])


if __name__ == "__main__":
    print("[UI] Starting Gradio at http://localhost:7860")
    demo.launch(server_port=7860, share=False, show_error=True)
