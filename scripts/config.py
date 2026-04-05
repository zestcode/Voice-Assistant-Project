"""
Local path configuration — replaces the Colab Drive paths in the notebook.
All other scripts import from here.
"""
import os
from pathlib import Path

# ── Project root (two levels up from this file: scripts/ → project root)
BASE = Path(__file__).resolve().parent.parent

# ── Model paths (local, no copy-to-runtime needed)
ASR1_PATH = BASE / "models" / "asr" / "faster-whisper-small"
ASR2_PATH = BASE / "models" / "asr" / "moonshine-base"
ASR_LARGE_TURBO_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"  # app integration model

LLM1_PATH    = BASE / "models" / "llm" / "qwen2.5-3b-q4_k_m"  / "qwen2.5-3b-instruct-q4_k_m.gguf"
LLM1_5B_PATH = BASE / "models" / "llm" / "qwen2.5-1.5b-q4_k_m" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM2_PATH    = BASE / "models" / "llm" / "llama-3.2-3b-q4_k_m" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

TTS1_PATH  = BASE / "models" / "tts" / "kokoro"   # kept for Kokoro benchmark (legacy)
TTS_F5_PATH = BASE / "models" / "tts" / "f5-tts"  # F5-TTS (weights auto-downloaded to HF cache)

# ── Benchmark dataset paths (already downloaded to data/benchmarks/)
BENCH_BASE     = BASE / "data" / "benchmarks"
ASR_BENCH_PATH = BENCH_BASE / "asr" / "librispeech_clean_test"
MMLU_BENCH_DIR = BENCH_BASE / "llm"
TTS_BENCH_PATH = BENCH_BASE / "tts" / "emergenttts_eval"

MMLU_SUBJECTS = [
    "high_school_mathematics",
    "high_school_physics",
    "high_school_biology",
    "high_school_world_history",
    "high_school_us_history",
    "college_mathematics",
    "college_physics",
    "college_biology",
    "philosophy",
    "international_law",
]

# ── Output paths
OUT_BASE          = BASE / "outputs"
OUT_TABLES        = OUT_BASE / "tables"
OUT_TTS_KOKORO    = OUT_BASE / "tts_audio" / "kokoro_bench"
OUT_TTS_GPTSOVITS = OUT_BASE / "tts_audio" / "gptsovits_bench"
OUT_TTS_F5        = OUT_BASE / "tts_audio" / "f5_bench"

# Reference voice for GPT-SoVITS (voice cloning)
REF_AUDIO = BASE / "data" / "ref_voice" / "ref.wav"
REF_TEXT  = "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"

# GPT-SoVITS code repo (cloned to code/)
GPTSOVITS_CODE = BASE / "code" / "GPT-SoVITS"

# GPT-SoVITS HTTP API (start server separately before running TTS benchmark)
# cd code/GPT-SoVITS && python api_v2.py
GPTSOVITS_API = "http://127.0.0.1:9880"

# ── Create output directories
for _d in [OUT_TABLES, OUT_TTS_KOKORO, OUT_TTS_GPTSOVITS, OUT_TTS_F5]:
    _d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("=== Local path config ===")
    print(f"BASE:      {BASE}")
    print(f"ASR1:      {ASR1_PATH}  {'OK' if ASR1_PATH.exists() else 'MISSING'}")
    print(f"ASR2:      {ASR2_PATH}  {'OK' if ASR2_PATH.exists() else 'MISSING'}")
    print(f"LLM1:      {LLM1_PATH}  {'OK' if LLM1_PATH.exists() else 'MISSING'}")
    print(f"LLM2:      {LLM2_PATH}  {'OK' if LLM2_PATH.exists() else 'MISSING'}")
    print(f"TTS1:      {TTS1_PATH}  {'OK' if TTS1_PATH.exists() else 'MISSING'}")
    print(f"ASR bench: {ASR_BENCH_PATH}  {'OK' if ASR_BENCH_PATH.exists() else 'MISSING'}")
    print(f"TTS bench: {TTS_BENCH_PATH}  {'OK' if TTS_BENCH_PATH.exists() else 'MISSING'}")
