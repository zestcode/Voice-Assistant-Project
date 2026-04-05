# llm_server.py — Design Explanation

## What problem this solves
The original `/generate` endpoint returns the full LLM response as one JSON blob after generation
completes (~0.2s). This forces the orchestrator to wait for the entire response before starting TTS,
adding a full sentence's worth of delay. The new `/generate_stream` endpoint streams tokens via
Server-Sent Events (SSE) so the orchestrator can start synthesizing speech the moment the first
complete sentence arrives.

## Architecture / data flow
```
Client (orchestrator thread) ──POST /generate_stream──> llm_server
                                                          │
                              llama_cpp stream=True       │
                              emits one token at a time   │
                                                          ▼
                              data: {"token": "Hello"}  ──┐
                              data: {"token": " there"} ──┤──> iter_lines() in orchestrator
                              data: [DONE]               ──┘
```

The orchestrator consumes this inside a `ThreadPoolExecutor` thread (since `requests.stream` is
blocking I/O), bridging to an `asyncio.Queue` so the async WebSocket handler can `yield` tokens.

## Key design decisions

**Why SSE over WebSocket for this endpoint?**
SSE is a one-directional stream over a plain HTTP response. The LLM server doesn't need to receive
anything mid-generation, so SSE is simpler. The orchestrator-to-browser path uses WebSocket because
it needs bidirectional (interrupt signal).

**Why POST not GET for SSE?**
The prompt is arbitrary-length text. GET querystrings have URL-length limits and expose prompts in
server logs. The browser's `EventSource` API only supports GET, but we consume programmatically via
`requests.post(stream=True)` so POST is fine.

**Why keep `/generate` alongside `/generate_stream`?**
`/generate` is used by `test.py` and `benchmark_*.py`. Removing it would break tests.

**Generator function vs async generator?**
`llama_cpp.Llama.create_chat_completion(stream=True)` is a synchronous generator — it blocks the
calling thread. `StreamingResponse` in FastAPI accepts a *sync* generator and runs it in a thread
pool automatically. Using an async generator here would require wrapping with `asyncio.to_thread`,
adding complexity for no benefit.

## Dependencies and assumptions
- `llama_cpp` must be installed in the `llm` conda env
- Model file at `models/llm/qwen2.5-1.5b-q4_k_m/*.gguf`
- uvicorn + fastapi in `llm` env
- Token format: `{"token": "<text>"}` — orchestrator parses this field

## Edge cases
- Empty delta from llama_cpp (role-only chunks): skipped via `if delta`
- Caller disconnects mid-stream: generator is garbage-collected, llama_cpp stops naturally
- Model not loaded (startup failure): both endpoints return 500 with traceback from FastAPI

## Context for Future Edits

**What must stay true for this file to keep working:**
- [ ] SSE format: lines must be `data: <json>\n\n` and terminate with `data: [DONE]\n\n`
- [ ] Token field name is `"token"` — orchestrator's `_consume_llm_stream` reads exactly this key
- [ ] `/generate` endpoint must remain for test.py compatibility
- [ ] Port 8002, conda env `llm`
- [ ] `n_gpu_layers=-1` stays unless VRAM issues arise

**Likely next changes:**
- Add system prompt configurability via request body
- Switch to larger model (Qwen2.5-3B or 7B) — only change `LLM_PATH`
- Add conversation history / multi-turn support

**Do not change without understanding:**
- The `StreamingResponse` with a sync generator: FastAPI/starlette runs it in a thread automatically.
  Switching to `async def token_generator()` breaks this — you'd need `asyncio.to_thread`.
- `X-Accel-Buffering: no` header: prevents nginx/proxy from buffering the SSE stream into chunks,
  which would make streaming appear as batch output.
