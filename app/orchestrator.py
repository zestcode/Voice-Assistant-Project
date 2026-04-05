"""
Voice AI Orchestrator — FastAPI + WebSocket
Run in 'voiceui' conda env:
  pip install fastapi uvicorn requests numpy soundfile scipy websockets
  python app/orchestrator.py

Endpoints:
  GET  /           → serves index.html
  GET  /health     → status of all three microservices
  WS   /ws         → streaming pipeline: audio→ASR→LLM(stream)→TTS(per sentence)→audio chunks
  POST /pipeline   → legacy blocking pipeline (used by tests)
  GET  /audio/{id} → serve generated WAV file (legacy pipeline only)
  GET  /report     → session statistics
"""

import asyncio
import base64
import io
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT    = Path(__file__).resolve().parent.parent
STATIC  = Path(__file__).resolve().parent / "static"
STATIC.mkdir(exist_ok=True)
OUT_DIR = ROOT / "outputs" / "audio_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASR_URL = "http://localhost:8001"
LLM_URL = "http://localhost:8002"
TTS_URL = "http://localhost:8003"

# Persistent session — reuses TCP connections, eliminates ~2s/call handshake overhead
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=6, pool_maxsize=6)
_session.mount("http://", _adapter)

# Thread pool for blocking HTTP calls inside async handlers
_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="orch")

REF_AUDIO = ROOT / "data" / "ref_voice" / "ref.wav"
REF_TEXT  = "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"

_session_log: list[dict] = []

app = FastAPI(title="Voice AI Orchestrator")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    status = {}
    for name, url in [("ASR", ASR_URL), ("LLM", LLM_URL), ("TTS", TTS_URL)]:
        try:
            r = _session.get(f"{url}/health", timeout=2)
            status[name] = r.json()
        except Exception as e:
            status[name] = {"status": "offline", "error": str(e)}
    return status


# ── Serve UI ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h2>index.html not found in app/static/</h2>", status_code=404)
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/audio/{audio_id}")
def serve_audio(audio_id: str):
    path = OUT_DIR / f"{audio_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(str(path), media_type="audio/wav")


# ── Audio conversion ──────────────────────────────────────────────────────────

def _to_wav_bytes(raw: bytes) -> bytes:
    """Return 16kHz mono WAV bytes. Pure Python — no subprocess."""
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        num = int(len(arr) * 16000 / sr)
        arr = scipy.signal.resample(arr, num).astype(np.float32)
        sr = 16000
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ── Sentence splitting ────────────────────────────────────────────────────────

_SENT_SPLIT  = re.compile(r'(?<=[.!?])\s+')
_COMMA_SPLIT = re.compile(r',\s*')
TTS_MAX_CHARS = 120  # F5-TTS fails with tensor mismatch above ~150 chars


def _split_long_text(text: str) -> list[str]:
    """
    Split text that exceeds TTS_MAX_CHARS into shorter chunks.
    Splits preferentially at commas; falls back to hard word-boundary split.
    """
    if len(text) <= TTS_MAX_CHARS:
        return [text]

    # Try splitting at commas
    parts = _COMMA_SPLIT.split(text)
    chunks = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = (current + ", " + part).lstrip(", ") if current else part
        if len(candidate) <= TTS_MAX_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If single part still too long, hard-split by words
            if len(part) > TTS_MAX_CHARS:
                words = part.split()
                sub = ""
                for w in words:
                    trial = (sub + " " + w).strip()
                    if len(trial) <= TTS_MAX_CHARS:
                        sub = trial
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = w
                if sub:
                    current = sub
                else:
                    current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:TTS_MAX_CHARS]]


def _flush_sentences(buf: str, min_words: int = 5) -> tuple[list[str], str]:
    """Pull complete sentences off the front of buf; require >= min_words per sentence.
    Long sentences are further split to stay within TTS_MAX_CHARS."""
    sentences = []
    while True:
        m = _SENT_SPLIT.search(buf)
        if not m:
            break
        candidate = buf[:m.start() + 1].strip()
        if len(candidate.split()) >= min_words:
            sentences.extend(_split_long_text(candidate))
            buf = buf[m.end():]
        else:
            break   # too short — wait for more tokens
    return sentences, buf


# ── Sync HTTP helpers (run in executor) ──────────────────────────────────────

def _call_asr_sync(wav_bytes: bytes) -> dict:
    r = _session.post(
        f"{ASR_URL}/transcribe",
        files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _call_tts_sync(text: str, ref_bytes: bytes | None, ref_txt: str) -> bytes:
    tts_files = {}
    if ref_bytes:
        tts_files["ref_audio"] = ("ref.wav", ref_bytes, "audio/wav")
    r = _session.post(
        f"{TTS_URL}/synthesize",
        data={"text": text, "ref_text": ref_txt},
        files=tts_files if tts_files else None,
        timeout=60,
    )
    r.raise_for_status()
    return r.content


def _consume_llm_stream(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
):
    """Thread: consume SSE stream from LLM server, push tokens into asyncio queue."""
    try:
        with _session.post(
            f"{LLM_URL}/generate_stream",
            json={"prompt": prompt},
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    token = json.loads(payload).get("token", "")
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[WS] LLM stream error: {e}")
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel


# ── Async helpers ─────────────────────────────────────────────────────────────

async def _stream_llm_tokens(prompt: str):
    """Async generator: yields LLM tokens one at a time via thread-bridged queue."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop.run_in_executor(_executor, _consume_llm_stream, prompt, queue, loop)
    while True:
        token = await queue.get()
        if token is None:
            break
        yield token


async def _tts_and_send(
    ws: WebSocket,
    text: str,
    idx: int,
    ref_bytes: bytes | None,
    ref_txt: str,
):
    """Synthesize one sentence and push audio_chunk to browser."""
    loop = asyncio.get_event_loop()
    try:
        wav = await loop.run_in_executor(
            _executor,
            lambda: _call_tts_sync(text, ref_bytes, ref_txt),
        )
        audio_b64 = base64.b64encode(wav).decode()
        await ws.send_json({"type": "audio_chunk", "idx": idx, "data": audio_b64})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[WS] TTS[{idx}] error: {e}")
        await ws.send_json({"type": "error", "msg": f"TTS[{idx}] failed: {e}"})


# ── WebSocket streaming pipeline ──────────────────────────────────────────────

async def _run_ws_pipeline(
    ws: WebSocket,
    wav_bytes: bytes,
    ref_bytes: bytes | None,
    ref_txt: str,
):
    loop = asyncio.get_event_loop()
    t_total = time.perf_counter()

    # ── ASR
    try:
        asr_data = await loop.run_in_executor(_executor, _call_asr_sync, wav_bytes)
    except Exception as e:
        await ws.send_json({"type": "error", "msg": f"ASR failed: {e}"})
        return

    transcript = asr_data["text"]
    asr_lat    = asr_data["latency"]
    print(f"[WS] ASR: '{transcript}'  ({asr_lat}s)")
    await ws.send_json({"type": "transcript", "text": transcript, "asr_s": asr_lat})

    if not transcript.strip():
        await ws.send_json({"type": "done"})
        return

    # ── LLM stream + sentence-level TTS (parallel)
    full_response = ""
    sentence_buf  = ""
    tts_tasks: list[asyncio.Task] = []
    chunk_idx = 0
    llm_start = time.perf_counter()

    try:
        async for token in _stream_llm_tokens(transcript):
            full_response += token
            sentence_buf  += token
            await ws.send_json({"type": "token", "text": token})

            sentences, sentence_buf = _flush_sentences(sentence_buf)
            for sent in sentences:
                task = asyncio.create_task(
                    _tts_and_send(ws, sent, chunk_idx, ref_bytes, ref_txt)
                )
                tts_tasks.append(task)
                chunk_idx += 1

        llm_lat = round(time.perf_counter() - llm_start, 3)

        # Flush any remainder that didn't end with punctuation
        if sentence_buf.strip():
            for remainder in _split_long_text(sentence_buf.strip()):
                task = asyncio.create_task(
                    _tts_and_send(ws, remainder, chunk_idx, ref_bytes, ref_txt)
                )
                tts_tasks.append(task)
                chunk_idx += 1

        tts_start = time.perf_counter()
        await asyncio.gather(*tts_tasks, return_exceptions=True)
        tts_lat = round(time.perf_counter() - tts_start, 3)

    except asyncio.CancelledError:
        for t in tts_tasks:
            t.cancel()
        return

    total = round(time.perf_counter() - t_total, 3)
    tok_s = round(len(full_response.split()) / llm_lat, 1) if llm_lat > 0 else 0
    print(f"[WS] done  asr={asr_lat}s  llm={llm_lat}s  tts={tts_lat}s  total={total}s")

    entry = {
        "timestamp":  datetime.now().isoformat(),
        "transcript": transcript,
        "response":   full_response,
        "asr_s":      round(asr_lat, 3),
        "llm_s":      llm_lat,
        "tts_s":      tts_lat,
        "total_s":    total,
        "tok_s":      tok_s,
    }
    _session_log.append(entry)
    _save_log(entry)

    await ws.send_json({"type": "done", "response": full_response, "metrics": entry})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_task: asyncio.Task | None = None

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype == "audio":
                # Cancel any in-flight pipeline before starting a new one
                if current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass

                wav_bytes = base64.b64decode(msg["data"])
                ref_bytes = base64.b64decode(msg["ref_audio"]) if msg.get("ref_audio") else None
                ref_txt   = msg.get("ref_text", "").strip() or REF_TEXT

                current_task = asyncio.create_task(
                    _run_ws_pipeline(websocket, wav_bytes, ref_bytes, ref_txt)
                )

            elif mtype == "interrupt":
                if current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass
                await websocket.send_json({"type": "interrupted"})

    except WebSocketDisconnect:
        if current_task and not current_task.done():
            current_task.cancel()
    except Exception as e:
        print(f"[WS] connection error: {e}")
        if current_task and not current_task.done():
            current_task.cancel()


# ── Legacy blocking pipeline (used by test.py) ────────────────────────────────

@app.post("/pipeline")
async def pipeline(
    audio:     UploadFile = File(...),
    ref_audio: UploadFile = File(default=None),
    ref_text:  str        = Form(default=""),
):
    t_total = time.perf_counter()
    raw_bytes = await audio.read()
    try:
        wav_bytes = _to_wav_bytes(raw_bytes)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Audio conversion failed: {e}"})

    try:
        r = _session.post(f"{ASR_URL}/transcribe",
                          files={"audio": ("audio.wav", wav_bytes, "audio/wav")}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"ASR failed: {e}"})

    asr_data   = r.json()
    transcript = asr_data["text"]
    asr_lat    = asr_data["latency"]
    if not transcript.strip():
        return JSONResponse({"transcript": "", "response": "", "audio_id": None,
                             "metrics": {"asr_s": asr_lat, "llm_s": 0, "tts_s": 0,
                                         "total_s": asr_lat, "tok_s": 0}})

    try:
        r = _session.post(f"{LLM_URL}/generate", json={"prompt": transcript}, timeout=60)
        r.raise_for_status()
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"LLM failed: {e}"})

    llm_data = r.json()
    response = llm_data["response"]
    llm_lat  = llm_data["latency"]
    tok_s    = llm_data["tok_s"]

    ref_bytes = await ref_audio.read() if ref_audio else None
    ref_txt   = ref_text.strip() or REF_TEXT
    tts_files = {"ref_audio": ("ref.wav", ref_bytes, "audio/wav")} if ref_bytes else {}

    try:
        r = _session.post(f"{TTS_URL}/synthesize",
                          data={"text": response, "ref_text": ref_txt},
                          files=tts_files if tts_files else None, timeout=60)
        r.raise_for_status()
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"TTS failed: {e}"})

    tts_lat  = float(r.headers.get("X-Latency", 0))
    total    = round(time.perf_counter() - t_total, 3)
    audio_id = str(uuid.uuid4())[:8]
    (OUT_DIR / f"{audio_id}.wav").write_bytes(r.content)

    entry = {"timestamp": datetime.now().isoformat(), "transcript": transcript,
             "response": response, "asr_s": round(asr_lat, 3), "llm_s": round(llm_lat, 3),
             "tts_s": round(tts_lat, 3), "total_s": total, "tok_s": round(tok_s, 1)}
    _session_log.append(entry)
    _save_log(entry)
    return {"transcript": transcript, "response": response, "audio_id": audio_id, "metrics": entry}


# ── Session log & report ──────────────────────────────────────────────────────

def _save_log(entry: dict):
    path = ROOT / "outputs" / "tables" / "session_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(entry)
    path.write_text(json.dumps(history, indent=2))


@app.get("/report")
def get_report():
    if not _session_log:
        return {"error": "No sessions yet"}
    n   = len(_session_log)
    avg = lambda k: round(float(np.mean([m[k] for m in _session_log])), 3)
    return {
        "sessions":    n,
        "avg_asr_s":   avg("asr_s"),
        "avg_llm_s":   avg("llm_s"),
        "avg_tts_s":   avg("tts_s"),
        "avg_total_s": avg("total_s"),
        "avg_tok_s":   round(float(np.mean([m["tok_s"] for m in _session_log])), 1),
        "log":         _session_log,
    }


if __name__ == "__main__":
    print("[Orchestrator] Starting at http://localhost:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning", timeout_keep_alive=75)
