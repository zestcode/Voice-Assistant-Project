# index.html — Design Explanation

## What problem this solves
The original fetch-based UI was a fire-and-forget POST: record → send → wait ~1.6s → receive everything.
No streaming display, no interrupt, and the browser had to re-establish HTTP connection each call.

The new WebSocket UI enables:
1. **Real-time transcript** — displayed immediately after ASR, before LLM starts
2. **Token streaming** — LLM response builds word-by-word in the UI
3. **Sentence-level audio queue** — first TTS chunk plays while LLM is still generating sentences 2, 3…
4. **Interrupt** — button + automatic VAD during playback sends `{type:"interrupt"}` to server
5. **Persistent connection** — single WebSocket, no reconnection overhead per request

## Architecture / data flow

```
Mic stream ──────────────────────────────────────────────────┐
     │                                                        │ (VAD reads during playback)
     ▼                                                        ▼
MediaRecorder ──stop──> blobToWav16k() ──> WS.send(audio)   AnalyserNode
                                                │
                        WS.onmessage ◄──────────┘
                             │
                    type switch:
                    "transcript"  → update transcript box
                    "token"       → append to response box
                    "audio_chunk" → enqueueChunk(idx, data)
                    "done"        → update metrics, setState idle
                    "interrupted" → resetAudioQueue, setState idle
                    "error"       → show error
                             │
                    AudioContext Queue
                    audioQueue[idx] = ArrayBuffer
                    plays idx=0,1,2… in order
                    drainQueue() called on each chunk arrival + each onended
```

## Key design decisions

**Why AudioContext queue instead of <audio> element?**
`<audio>` can play one src at a time. Chaining multiple WAV blobs sequentially requires `ended`
event + `src` swap, which has a ~100ms gap between sentences (browser re-initializes decoder).
`AudioContext.createBufferSource()` queues decoded PCM directly to the audio graph, allowing
gapless or near-gapless playback. Each chunk gets its own `AudioContext` instance to avoid
state management complexity.

**Why indexed queue (audioQueue[idx]) instead of FIFO array?**
TTS tasks run in parallel on the server — sentence 2 may finish before sentence 1 (unlikely but
possible if sentence 1 is long). An index map guarantees in-order playback regardless of arrival
order. `drainQueue()` only advances when `audioQueue[nextPlayIdx]` exists.

**VAD strategy — AnalyserNode energy threshold:**
A permanent mic stream is requested on page load (reused for recording). During playback,
`animateWave()` already calls `analyser.getByteTimeDomainData()` per animation frame. VAD taps
into this: if RMS energy > threshold for 400ms continuously, interrupt fires. The 400ms debounce
prevents background noise triggering on a single loud frame.

**Why persistent mic stream (not per-recording)?**
Requesting `getUserMedia` each time adds ~100-300ms permission + hardware init latency on some
platforms. One stream, opened once on page load, is reused for both recording and VAD.

**Why base64 for audio over WebSocket?**
FastAPI's WebSocket `.send_json()` only accepts JSON strings, not binary frames easily. Base64
encodes WAV bytes into a JSON field. The overhead is ~33% size increase, acceptable for ~50KB WAV
chunks. Alternative: send binary frames with a 1-byte type prefix — avoided as it adds protocol
complexity without meaningful latency improvement on localhost.

## Dependencies and assumptions
- WebSocket at `ws://localhost:7860/ws` — orchestrator must be running
- Browser must support: `MediaRecorder`, `AudioContext`, `OfflineAudioContext`, `WebSocket`
- All modern browsers (Chrome 66+, Firefox 76+, Edge 79+) support these
- Server sends `audio_chunk` messages with `idx` field starting at 0, monotonically increasing

## Edge cases
- WS disconnect: `ws.onclose` resets state, shows reconnect button
- Audio decode failure (malformed chunk): caught, skips that chunk, drainQueue continues
- VAD false positive (cough/background noise): 400ms hold-time filters most transients
- `nextPlayIdx` desync after interrupt: `resetAudioQueue()` resets to 0 for next pipeline

## Context for Future Edits

**What must stay true for this file to keep working:**
- [ ] WS message format: `{type, ...fields}` — must match orchestrator send_json calls exactly
- [ ] `audio_chunk.idx` field — queue depends on this being 0-based sequential integers
- [ ] `audio_chunk.data` field — base64-encoded WAV bytes
- [ ] VAD uses `vadAnalyser` from persistent mic stream — do not close this stream
- [ ] `blobToWav16k()` sends 16kHz mono WAV — ASR server expects this format
- [ ] `resetAudioQueue()` must be called on every new pipeline start

**Likely next changes:**
- Streaming ASR: send audio chunks while still recording (requires chunked MediaRecorder)
- Language selector: pass `lang` field in WS audio message
- Voice clone: currently sends ref_audio as base64 in WS message, could be pre-uploaded separately
- VAD threshold slider: expose `VAD_THRESHOLD` as a UI control

**Do not change without understanding:**
- `drainQueue()` is re-entrant safe because `isPlaying` flag prevents double-play. But it must be
  called in exactly two places: on chunk arrival AND in `source.onended`. Missing either breaks queue.
- `audioQueue[nextPlayIdx]` uses numeric keys in a plain object — JavaScript coerces these to strings
  internally but comparison works. Do not switch to a `Map` without updating `drainQueue`.
- The `playingChain` promise is replaced by `currentSource.onended` callback pattern — they are
  mutually exclusive approaches. Do not mix them.
