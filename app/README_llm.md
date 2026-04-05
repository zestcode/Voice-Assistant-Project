# LLM Microservice

**Model:** `Qwen2.5-3B-Instruct` Q4\_K\_M GGUF (~2.0 GB)
**Conda env:** `llm`
**Port:** `8002`

---

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `POST` | `/generate` | `{"prompt": str}` | `{"response": str, "latency": float, "tok_s": float}` |
| `POST` | `/generate_stream` | `{"prompt": str}` | SSE stream: `data: {"token": "..."}` … `data: [DONE]` |
| `GET` | `/health` | — | `{"status": "ok", "model": "qwen2.5-3b-q4_k_m"}` |

---

## Environment Setup

`llama-cpp-python` must be compiled with CUDA support.
Prebuilt wheels often mismatch the local CUDA toolkit — build from source if needed:

```bash
conda activate llm

# Set build flags to target your CUDA toolkit (adjust path if different)
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe"
set FORCE_CMAKE=1

pip install llama-cpp-python --no-binary :all: --upgrade
pip install fastapi uvicorn pydantic
```

A pre-built wheel for CUDA 13.0 / Blackwell is included in the repository root:
```
llama_cpp_python-0.3.16+cuda13.0.sm100.blackwell-cp311-cp311-win_amd64.whl
```
Install it directly if it matches your system: `pip install <wheel_file>`.

Verify GPU is used after install:
```bash
python -c "from llama_cpp import Llama; print('OK')"
```

---

## Model File

Place the GGUF file at:
```
models/llm/qwen2.5-3b-instruct-q4_k_m.gguf
```

Download from HuggingFace:
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF',
    filename='qwen2.5-3b-instruct-q4_k_m.gguf',
    local_dir='models/llm'
)
"
```

---

## Run

```bash
conda activate llm
cd <project_root>
python app/llm_server.py
```

Expected startup output:
```
[LLM] Loading Qwen2.5-3B Q4_K_M (n_gpu_layers=-1) ...
[LLM] Ready
INFO:     Uvicorn running on http://0.0.0.0:8002
```

---

## Implementation Notes

- **Inference config:** `n_ctx=2048`, `n_threads=4`, `n_gpu_layers=-1` (all layers on GPU)
- **System prompt:** `"You are a concise voice assistant. Reply in 1-3 sentences max."`
- **max_tokens:** `400` for conversational turns, **temperature:** `0.7`
- **Throughput:** ~114 tok/s median on CUDA (RTX-class GPU, range 40–207 tok/s across session turns)
- **Streaming:** `/generate_stream` sends tokens via SSE as they are generated; the orchestrator uses this to dispatch each completed sentence to TTS in parallel with ongoing generation

---

## Quick Test

```bash
# Health check
curl http://localhost:8002/health

# Generate a reply (blocking)
curl -X POST http://localhost:8002/generate \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"What is machine learning?\"}" \
     -s | python -m json.tool

# Streaming response (SSE)
curl -X POST http://localhost:8002/generate_stream \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"What is machine learning?\"}" \
     -N
```
