# orchestrator.py — Design Explanation

## What problem this solves
The original HTTP `/pipeline` endpoint is a single synchronous request: browser waits for
ASR+LLM+TTS to all complete before receiving anything. This causes ~1.6s perceived latency and
makes interrupt impossible (no channel back to server mid-request).

The WebSocket `/ws` endpoint solves both problems:
1. **Streaming**: browser receives transcript immediately after ASR, tokens as LLM generates,
   and audio chunks sentence-by-sentence as TTS completes — first audio plays ~0.6s after speech ends.
2. **Interrupt**: browser sends `{"type":"interrupt"}` at any time; server cancels the pipeline task.

## Architecture / data flow

```
Browser ──WS── Orchestrator ──HTTP── ASR :8001
                    │        ──SSE── LLM :8002
                    │        ──HTTP── TTS :8003
                    │
                    ▼ messages pushed to browser:
  {type:"transcript"}  — after ASR
  {type:"token"}       — per LLM token (real-time display)
  {type:"audio_chunk", idx:N} — per sentence TTS result (base64 WAV)
  {type:"done"}        — pipeline complete + metrics
  {type:"interrupted"} — confirmed interrupt
  {type:"error"}       — any stage failure
```

## Key design decisions

**Why asyncio.Task per connection, not per request?**
Each WebSocket connection has one coroutine (`ws_endpoint`) that owns `current_task`. A new audio
message cancels the previous task before starting a new one. This handles rapid re-sends without
leaking tasks.

**Why ThreadPoolExecutor for HTTP calls?**
`requests` is blocking I/O. Running it directly in an `async def` would block uvicorn's event loop,
freezing all other connections. `run_in_executor` offloads blocking calls to threads.

**Why asyncio.Queue to bridge the LLM SSE stream?**
The SSE consumer runs in a thread (because `requests.stream` is blocking). The WebSocket sender is
async. An `asyncio.Queue` is the canonical bridge: thread calls `loop.call_soon_threadsafe(queue.put_nowait, token)`,
async code does `await queue.get()`. No polling, no busy-wait.

**Why sentence-level TTS (not word-level, not full-response)?**
- Word-level: TTS quality degrades on single words (no prosody context)
- Full-response: waits for entire LLM output (~150 tokens), adding 0.2s+ delay
- Sentence-level: balanced — good prosody, first audio arrives after first sentence (~5-8 words)

**Why `asyncio.gather(*tts_tasks)` instead of playing them sequentially?**
TTS for sentence 2 starts while sentence 1 is synthesizing. Since GPU time is the bottleneck,
parallel synthesis reduces total TTS time. The browser plays chunks in `idx` order regardless of
arrival order, so out-of-order delivery is safe.

**Why keep HTTP `/pipeline` endpoint?**
`test.py` and any non-WS clients use it. It's 40 lines and maintenance-free — no reason to remove.

**Sentence splitting logic (`_flush_sentences`):**
Uses `re.compile(r'(?<=[.!?])\s+')` — splits only on `.!?` *followed by whitespace* (not mid-word
periods). Requires ≥5 words to avoid splitting on "Dr. Smith" or "e.g. test". Short final fragments
are flushed as-is after LLM stream ends.

## Dependencies and assumptions
- `voiceui` conda env: `fastapi uvicorn requests numpy soundfile scipy websockets`
- ASR :8001, LLM :8002, TTS :8003 must be running before ws requests arrive
- LLM server must have `/generate_stream` SSE endpoint (added in this session)
- Browser sends audio as base64-encoded 16kHz mono WAV in `{"type":"audio","data":"..."}` message

## Edge cases
- Empty transcript (silence): sends `{type:"done"}` immediately, no LLM/TTS called
- TTS failure on one sentence: sends `{type:"error"}`, other sentences continue
- WebSocket disconnect mid-pipeline: `WebSocketDisconnect` cancels `current_task`
- CancelledError in `run_in_executor`: thread keeps running but result is discarded (acceptable)
- Very short LLM response (<5 words, no sentence end): entire response flushed as one TTS call

## Context for Future Edits

**What must stay true for this file to keep working:**
- [ ] `_consume_llm_stream` reads `data["token"]` — must match llm_server SSE field name
- [ ] Audio chunks sent with `{"type":"audio_chunk","idx":N,"data":base64}` — browser queues on `idx`
- [ ] `_executor` ThreadPoolExecutor shared across all connections — don't create per-request
- [ ] `_session` (requests.Session) shared — gives TCP connection reuse across WS requests
- [ ] Port 7860, conda env `voiceui`
- [ ] `REF_TEXT` and `REF_AUDIO` must match tts_server defaults

**Likely next changes:**
- Add streaming ASR (chunk audio while recording, not after stop)
- Add conversation history (pass prior turns to LLM)
- Add model-switching endpoint (change ASR/LLM/TTS model at runtime)
- Move `_executor` worker count to config

**Do not change without understanding:**
- `loop.call_soon_threadsafe(queue.put_nowait, token)`: the thread doesn't own the event loop.
  Using `queue.put_nowait` directly from a thread (without `call_soon_threadsafe`) is NOT thread-safe.
- `asyncio.Task.cancel()` on a task blocked in `run_in_executor`: the executor thread is NOT
  cancelled — it finishes naturally. Only the awaiting coroutine gets `CancelledError` at its
  next `await` point. This is intentional and harmless.
- `_flush_sentences` min_words=5: lowering this causes TTS to fire on fragments like "Yes." or
  "OK." which the TTS model handles poorly (silent/distorted output).
