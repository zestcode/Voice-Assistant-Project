# benchmark_e2e_streaming.py — Design Explanation

## What problem this solves
The existing `session_log.json` only captures server-side latencies (asr_s, llm_s, tts_s).
It cannot measure:
- **TTFT** (Time to First Token) — when the user first sees LLM text streaming
- **TTFA** (Time to First Audio) — when the first TTS chunk arrives at the client
- How these metrics **scale with input audio length** (5s vs 15s vs 30s)

This script fills that gap by acting as a programmatic WebSocket client, replaying real
LibriSpeech audio at three length tiers and recording precise client-side timestamps.

## Architecture / data flow
```
LibriSpeech Arrow file
  → load 60 samples via datasets
  → group into 3 tiers by concatenation:
      SHORT  (~5s):  1 sample
      MEDIUM (~15s): 3 samples concatenated
      LONG   (~30s): 6 samples concatenated
  → each tier: 20 WAV inputs

For each WAV:
  t_send = perf_counter()
  ws.send({type:"audio", data:base64_wav})
  receive loop:
    "transcript"  → record t_transcript, asr_s from message
    "token" (1st) → record t_first_token  (TTFT = t_first_token - t_send)
    "audio_chunk" idx=0 → record t_first_audio (TTFA = t_first_audio - t_send)
    "done"        → record t_done, extract server metrics, count n_chunks
  wait 2s between samples (let pipeline drain)

→ outputs/tables/e2e_streaming_benchmark.csv  (per-sample rows)
→ outputs/tables/e2e_streaming_summary.csv    (mean/std/p50/p95 per tier)
→ printed summary table
```

## Key design decisions

**Why WebSocket client (websockets lib) instead of HTTP /pipeline?**
The `/pipeline` endpoint is synchronous and returns everything at once. It cannot expose
TTFT or TTFA — those only exist in the streaming WS protocol where the server pushes
transcript → tokens → audio_chunk messages individually.

**Why use LibriSpeech instead of synthesized audio?**
LibriSpeech is the standard ASR evaluation corpus already used in this project. Using it
as pipeline input makes the benchmark consistent with the standalone ASR benchmark and
tests the system on real human speech variation (not clean TTS output).

**Why concatenate samples for MEDIUM/LONG tiers?**
Most individual LibriSpeech test-clean utterances are 3-15s. Concatenating with 0.5s
silence gap between them produces realistic longer input without recording new audio.
The 0.5s gap also tests the ASR VAD filter's ability to handle mid-utterance pauses.

**Why 20 samples per tier?**
Enough for stable p95 estimates without running for too long. 60 total samples at
~5s average processing time = ~5 minutes total runtime.

**Why asyncio + websockets.connect?**
The WebSocket protocol requires async I/O. `websockets` is already in voiceui env.
Using `asyncio.wait_for` with a 60s timeout prevents the benchmark from hanging if
a sample fails.

**Silence between samples (2s wait)**
The server pipeline is stateful per WebSocket connection. Sending the next sample
before the previous TTS finishes playing would queue up on the server and inflate
latency measurements. 2s gap ensures the server is idle before the next sample.

## Dependencies and assumptions
- Three microservices must be running: ASR :8001, LLM :8002, TTS :8003
- Orchestrator must be running at :7860 (provides /ws endpoint)
- Run in voiceui env: `pip install websockets datasets soundfile scipy numpy pandas`
- LibriSpeech Arrow file at `data/benchmarks/asr/librispeech_clean_test/`
- Audio field name in dataset is "audio" with sub-keys "array" and "sampling_rate"

## Edge cases
- Sample timeout (60s): logged as failed, excluded from statistics
- Empty transcript from ASR: logged with asr_s but TTFT/TTFA set to NaN
- n_chunks=0 (TTS failed): logged, excluded from TTFA stats
- Connection drop: script reconnects and retries current sample once

## Context for Future Edits

**What must stay true for this file to keep working:**
- [ ] WS message format matches orchestrator.py: {type, asr_s in transcript msg, idx in audio_chunk}
- [ ] orchestrator sends `done` with `metrics` dict containing asr_s/llm_s/tts_s/total_s/tok_s
- [ ] WebSocket endpoint at ws://localhost:7860/ws
- [ ] LibriSpeech dataset has "audio" column with "array" (float32) and "sampling_rate" fields
- [ ] voiceui env has: websockets, datasets, soundfile, scipy, numpy, pandas

**Likely next changes:**
- Add more tiers (e.g., 60s) or finer granularity
- Add WER measurement (ASR output vs LibriSpeech reference text)
- Parametrize server URLs for testing different model configurations
- Add GPU memory sampling during pipeline execution

**Do not change without understanding:**
- The `await asyncio.wait_for(receive_one_pipeline(...), 60)` pattern: if removed,
  a hung TTS call will block the entire benchmark indefinitely.
- `asyncio.sleep(2)` between samples: removing this causes server-side pipeline
  overlap, inflating all latency measurements for subsequent samples.
- Base64 encoding in 32KB chunks: same fix as index.html — large WAV files
  exceed JS/Python string limits if encoded in one call.
