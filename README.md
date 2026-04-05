# Voice AI Assistant

End-to-end local speech-to-speech pipeline: **ASR → LLM → TTS**.
All inference runs fully on-device — no cloud API calls.

```
Microphone → [ASR :8001] → [LLM :8002] → [TTS :8003] → Speaker
              moonshine        llm          gptsovits
```

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA RTX-class, **12 GB VRAM** minimum |
| OS | Windows 11 (tested); Linux should work with minor path changes |
| CUDA Toolkit | 13.0 (for `llama-cpp-python` build) |
| Disk | ~8 GB total (models + benchmark data) |
| RAM | 16 GB recommended |

---

## Benchmark Results (Summary)

### ASR — LibriSpeech test-clean, 200 samples

| Model | WER (%) | RTFx | Role |
|---|---|---|---|
| Moonshine-Base | 15.39 | 24.64 | Candidate |
| Faster-Whisper-Small | 15.98 | 29.14 | **Selected for pipeline** |
| Faster-Whisper-Large-V3-Turbo | 6.83 | 20.93 | Accuracy upper bound |

### LLM — MMLU (5 subjects)

| Model | Avg. Acc. | Time (s) | Role |
|---|---|---|---|
| Qwen2.5-1.5B Q4\_K\_M | 24.4% | 54.6 | **Selected for pipeline** |
| Llama-3.2-3B Q4\_K\_M | 23.3% | 68.8 | Comparison |

### TTS — EmergentTTS-Eval, 20 samples

| Model | RT-WER (%) | Spk Sim | Latency (s) | Role |
|---|---|---|---|---|
| Kokoro | 5.52 | N/A | 0.15 | Intelligibility baseline (no cloning) |
| GPT-SoVITS | 46.49 | 0.83 | 3.75 | Voice cloning candidate |
| F5-TTS | 18.77 | 0.90 | 1.30 | **Selected for pipeline** |

### End-to-End Latency (34 session turns, streaming pipeline)

| Stage | Mean (s) | Median (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| ASR | 0.63 | 0.34 | 0.15 | 4.37 |
| LLM | 0.11 | 0.10 | 0.04 | 0.27 |
| TTS (per sentence chunk) | 0.94 | 0.80 | 0.47 | 3.18 |
| **Total** | **1.80** | **1.61** | **0.77** | **4.98** |

Sentence-level streaming reduces time-to-first-audio by **~53%** vs. non-streaming baseline.

---

## Project Structure

```
voice_ai_project/
├── run.py                          # One-command launcher (starts all 4 processes)
├── app/
│   ├── asr_server.py               # FastAPI ASR  service  → port 8001
│   ├── llm_server.py               # FastAPI LLM  service  → port 8002
│   ├── tts_server.py               # FastAPI TTS  service  → port 8003
│   ├── orchestrator.py             # Pipeline coordinator (WebSocket + SSE + HTTP)
│   ├── voice_assistant.py          # Gradio UI
│   ├── README_asr.md
│   ├── README_llm.md
│   ├── README_tts.md
│   └── README_ui.md
├── scripts/
│   ├── config.py                   # Central paths and constants
│   ├── benchmark_asr_*.py          # ASR benchmarks (3 scripts)
│   ├── benchmark_llm_*.py          # LLM benchmarks (3 scripts)
│   ├── benchmark_tts_*.py          # TTS benchmarks (3 scripts)
│   ├── benchmark_e2e_streaming.py  # End-to-end streaming latency
│   └── README.md                   # Benchmark usage guide
├── models/
│   ├── asr/   (faster-whisper-small, moonshine-base)
│   ├── llm/   (qwen2.5-1.5b, llama-3.2-3b, qwen2.5-3b — GGUF)
│   └── tts/   (kokoro, gpt-sovits, f5-tts weights cached by HF)
├── data/
│   ├── benchmarks/  (librispeech, mmlu, emergenttts_eval — HF Arrow)
│   └── ref_voice/ref.wav           # Default TTS speaker reference
├── outputs/tables/                 # CSV benchmark results
├── notebooks/EECE7398_HW2_benchmark.ipynb
└── llama_cpp_python-0.3.16+cuda13.0.sm100.blackwell-cp311-cp311-win_amd64.whl
```

---

## Conda Environments

Four environments are used to avoid dependency conflicts:

| Env name | Used by | Key packages |
|---|---|---|
| `moonshine` | ASR server, ASR benchmarks | `faster-whisper`, `transformers`, `datasets`, `jiwer` |
| `llm` | LLM server, LLM benchmarks | `llama-cpp-python` (CUDA build), `datasets` |
| `gptsovits` | TTS server, TTS benchmarks | `f5-tts`, `faster-whisper`, `resemblyzer`, `torchaudio` |
| `voiceui` | Gradio UI (`run.py` launcher) | `gradio`, `requests`, `numpy`, `soundfile`, `scipy` |

### Create environments

```bash
# moonshine
conda create -n moonshine python=3.11 -y
conda activate moonshine
pip install faster-whisper transformers datasets jiwer soundfile librosa numpy pandas torch torchvision torchaudio

# llm
conda create -n llm python=3.11 -y
conda activate llm
# Install pre-built wheel (CUDA 13.0 / Blackwell, included in repo root)
pip install llama_cpp_python-0.3.16+cuda13.0.sm100.blackwell-cp311-cp311-win_amd64.whl
pip install fastapi uvicorn pydantic datasets pandas

# gptsovits
conda create -n gptsovits python=3.11 -y
conda activate gptsovits
pip install f5-tts faster-whisper resemblyzer jiwer soundfile torchaudio fastapi uvicorn

# voiceui
conda create -n voiceui python=3.11 -y
conda activate voiceui
pip install gradio requests numpy soundfile scipy
```

> **llama-cpp-python from source** (if the pre-built wheel does not match your CUDA version):
> ```bash
> conda activate llm
> set CMAKE_ARGS=-DGGML_CUDA=on
> set FORCE_CMAKE=1
> pip install llama-cpp-python --no-binary :all: --upgrade
> ```

---

## Model Setup

### ASR models (local files)

```bash
# Faster-Whisper-Small — download via faster-whisper (auto on first run, or manual):
conda activate moonshine
python -c "from faster_whisper import WhisperModel; WhisperModel('small', download_root='models/asr/faster-whisper-small')"

# Moonshine-Base — download via HuggingFace:
python -c "from huggingface_hub import snapshot_download; snapshot_download('UsefulSensors/moonshine', local_dir='models/asr/moonshine-base')"
```

### LLM models (GGUF files)

Place files at the paths configured in `scripts/config.py`:

```
models/llm/qwen2.5-1.5b-q4_k_m/qwen2.5-1.5b-instruct-q4_k_m.gguf
models/llm/llama-3.2-3b-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf
models/llm/qwen2.5-3b-q4_k_m/qwen2.5-3b-instruct-q4_k_m.gguf
```

Download from HuggingFace (example for Qwen2.5-1.5B):
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF',
                'qwen2.5-1.5b-instruct-q4_k_m.gguf',
                local_dir='models/llm/qwen2.5-1.5b-q4_k_m')
"
```

### TTS models

F5-TTS and Faster-Whisper-Large-V3-Turbo (used in ASR server) are **auto-downloaded** from HuggingFace on first server start (~1.3 GB and ~1.5 GB respectively).

GPT-SoVITS pretrained weights should be placed under `models/tts/gpt-sovits/` as configured in `code/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml`.

### Reference voice

Place a 3–15 second WAV file at:
```
data/ref_voice/ref.wav
```
Update `REF_TEXT` in `scripts/config.py` with the transcript of the clip.

---

## Quick Start — One Command

```bash
# Edit CONDA_BASE in run.py first if your miniconda is not at D:\Softeware\miniconda
conda activate voiceui
python run.py
```

`run.py` will:
1. Start the ASR server (`moonshine` env, port 8001)
2. Start the LLM server (`llm` env, port 8002)
3. Start the TTS server (`gptsovits` env, port 8003)
4. Poll `/health` on all three until ready (timeout: 480 s)
5. Launch the Gradio UI at **http://localhost:7860**
6. On `Ctrl+C`: gracefully shut down all servers

---

## Manual Start — Four Terminals

```bash
# Terminal 1 — ASR
conda activate moonshine
python app/asr_server.py

# Terminal 2 — LLM
conda activate llm
python app/llm_server.py

# Terminal 3 — TTS
conda activate gptsovits
python app/tts_server.py

# Terminal 4 — UI
conda activate voiceui
python app/voice_assistant.py
# Open http://localhost:7860
```

---

## Running Benchmarks

See [scripts/README.md](scripts/README.md) for full details.

```bash
# Download all benchmark datasets first (run once)
conda activate moonshine
python -c "exec(open('scripts/config.py').read())"   # verify paths
# then follow scripts/README.md data preparation section

# ASR benchmarks
conda activate moonshine
python scripts/benchmark_asr_whisper.py
python scripts/benchmark_asr_moonshine.py
python scripts/benchmark_asr_whisper_large.py   # downloads ~1.5 GB on first run

# LLM benchmarks
conda activate llm
python scripts/benchmark_llm_qwen.py
python scripts/benchmark_llm_llama.py
python scripts/benchmark_llm_prompt_strategies.py

# TTS benchmarks (requires GPT-SoVITS server for gptsovits script)
conda activate gptsovits
python scripts/benchmark_tts_f5.py
python scripts/benchmark_tts_gptsovits.py       # start code/GPT-SoVITS/api_v2.py first
```

Results are written to `outputs/tables/`.

---

## Service Documentation

| Service | README |
|---|---|
| ASR (port 8001) | [app/README_asr.md](app/README_asr.md) |
| LLM (port 8002) | [app/README_llm.md](app/README_llm.md) |
| TTS (port 8003) | [app/README_tts.md](app/README_tts.md) |
| Gradio UI | [app/README_ui.md](app/README_ui.md) |
| Benchmarks | [scripts/README.md](scripts/README.md) |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `run.py` exits: `CONDA_BASE not found` | Edit `CONDA_BASE` at top of `run.py` to your miniconda path |
| LLM server starts on CPU only | Rebuild `llama-cpp-python` with `GGML_CUDA=on` (see README_llm.md) |
| TTS server exits silently on Windows | Fixed: `workers` param removed from `uvicorn.run()` in `api_v2.py` |
| ASR returns hallucinated text | Audio too short or quiet; built-in 0.5 s guard + VAD filter handles this |
| F5-TTS torchaudio error on Windows | Fixed: `torchaudio.load` patched to use `soundfile` in `tts_server.py` |
| Port already in use | `netstat -ano | findstr :800x` to find the PID, then kill it |
