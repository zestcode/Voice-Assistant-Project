"""
LLM microservice — run in 'llm' conda env
  conda activate llm
  python app/llm_server.py

POST /generate         JSON: {"prompt": str}  → {"response": str, "latency": float, "tok_s": float}
POST /generate_stream  JSON: {"prompt": str}  → SSE stream: data: {"token": str} … data: [DONE]
GET  /health                                  → {"status": "ok"}
"""
import json
import os
import sys
import time
from pathlib import Path

# Windows DLL fix for llama-cpp-python
for _dll in [
    r"D:\Softeware\miniconda\envs\llm\Library\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
]:
    if os.path.isdir(_dll):
        os.add_dll_directory(_dll)

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

BASE     = Path(__file__).resolve().parent.parent
LLM_PATH = BASE / "models" / "llm" / "qwen2.5-1.5b-q4_k_m" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
N_GPU    = -1  # let llama-cpp-python auto-detect GPU; set to 0 to force CPU
PORT     = 8002

app = FastAPI(title="LLM Server")
llm = None


class PromptRequest(BaseModel):
    prompt: str


@app.on_event("startup")
def load_model():
    global llm
    from llama_cpp import Llama
    print(f"[LLM] Loading Qwen2.5-1.5B Q4_K_M (n_gpu_layers={N_GPU}) ...")
    llm = Llama(
        model_path=str(LLM_PATH),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=N_GPU,
        verbose=False,
    )
    # Warm-up: prime CUDA kernels
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )
    print("[LLM] Ready")


@app.get("/health")
def health():
    return {"status": "ok", "model": "qwen2.5-1.5b-q4_k_m"}


@app.post("/generate")
def generate(req: PromptRequest):
    t0  = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system",
             "content": "You are a helpful voice assistant. Give clear and complete answers."},
            {"role": "user", "content": req.prompt},
        ],
        max_tokens=400,
        temperature=0.7,
    )
    lat    = round(time.perf_counter() - t0, 3)
    text   = out["choices"][0]["message"]["content"].strip()
    tokens = out["usage"]["completion_tokens"]
    return {"response": text, "latency": lat, "tok_s": round(tokens / lat, 1)}


_SYS_MSG = "You are a helpful voice assistant. Give clear and complete answers."


@app.post("/generate_stream")
def generate_stream(req: PromptRequest):
    """SSE endpoint: streams tokens as `data: {"token": "..."}` lines, ends with `data: [DONE]`."""
    def token_generator():
        for chunk in llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYS_MSG},
                {"role": "user",   "content": req.prompt},
            ],
            max_tokens=400,
            temperature=0.7,
            stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                yield f"data: {json.dumps({'token': delta})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", timeout_keep_alive=75)
