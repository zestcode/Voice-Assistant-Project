# Benchmark Scripts

Standalone evaluation scripts for ASR, LLM, and TTS modules.
All scripts read paths from `config.py` and write CSV results to `outputs/tables/`.

---

## Directory Layout

```
scripts/
├── config.py                          # Central path config (edit BASE if needed)
├── benchmark_asr_whisper.py           # Faster-Whisper-Small  on LibriSpeech
├── benchmark_asr_moonshine.py         # Moonshine-Base        on LibriSpeech
├── benchmark_asr_whisper_large.py     # Faster-Whisper-Large-V3-Turbo on LibriSpeech
├── benchmark_llm_qwen.py              # Qwen2.5-1.5B MMLU
├── benchmark_llm_llama.py             # Llama-3.2-3B  MMLU
├── benchmark_llm_prompt_strategies.py # Zero-shot / role / few-shot / CoT comparison
├── benchmark_tts_gptsovits.py         # GPT-SoVITS  round-trip WER + speaker similarity
└── benchmark_tts_f5.py                # F5-TTS      round-trip WER + speaker similarity
```

---

## Conda Environments

| Script group | Conda env | Key packages |
|---|---|---|
| ASR (whisper, large) | `moonshine` | `faster-whisper`, `datasets`, `jiwer`, `soundfile`, `librosa`, `torch` |
| ASR (moonshine) | `moonshine` | `transformers`, `datasets`, `jiwer`, `soundfile`, `torch` |
| LLM | `llm` | `llama-cpp-python` (CUDA build), `datasets`, `pandas` |
| TTS (gptsovits, f5) | `gptsovits` | `f5-tts`, `faster-whisper`, `resemblyzer`, `jiwer`, `soundfile`, `torchaudio` |

---

## 1 — Data Preparation

Run once before any benchmark. Downloads all datasets to `data/benchmarks/`.

```bash
# Uses any env with `datasets` installed (e.g. moonshine)
conda activate moonshine
python -c "
from datasets import load_dataset
# LibriSpeech test-clean
ds = load_dataset('librispeech_asr', 'clean', split='test')
ds.save_to_disk('data/benchmarks/asr/librispeech_clean_test')
# MMLU subjects
for subj in ['college_computer_science','college_physics','high_school_biology',
             'high_school_world_history','professional_medicine',
             'high_school_mathematics','high_school_physics','high_school_us_history',
             'college_mathematics','college_biology','philosophy','international_law']:
    load_dataset('cais/mmlu', subj, split='test').save_to_disk(f'data/benchmarks/llm/mmlu_{subj}')
# EmergentTTS-Eval
load_dataset('bosonai/EmergentTTS-Eval').save_to_disk('data/benchmarks/tts/emergenttts_eval')
"
```

---

## 2 — ASR Benchmarks

All three scripts use LibriSpeech test-clean (200 samples by default).
**Metrics:** WER (%), RTFx

### Faster-Whisper-Small

```bash
conda activate moonshine
cd <project_root>
python scripts/benchmark_asr_whisper.py
# Output: outputs/tables/asr_whisper_benchmark.csv
```

### Moonshine-Base

```bash
conda activate moonshine
python scripts/benchmark_asr_moonshine.py
# Output: outputs/tables/asr_moonshine_benchmark.csv
```

### Faster-Whisper-Large-V3-Turbo  *(app integration model)*

Downloads ~1.5 GB from HuggingFace on first run.

```bash
conda activate moonshine
python scripts/benchmark_asr_whisper_large.py
# Output: outputs/tables/asr_whisper_large_benchmark.csv
```

---

## 3 — LLM Benchmarks

All scripts use MMLU (50 samples/subject by default).
**Metric:** Accuracy (%)

### Qwen2.5-1.5B

```bash
conda activate llm
python scripts/benchmark_llm_qwen.py
# Output: outputs/tables/llm_qwen_benchmark.csv
```

### Llama-3.2-3B

```bash
conda activate llm
python scripts/benchmark_llm_llama.py
# Output: outputs/tables/llm_llama_benchmark.csv
```

### Prompt Strategy Comparison (zero-shot / role / few-shot / CoT)

Evaluates Qwen2.5-1.5B, Qwen2.5-3B, Llama-3.2-3B across all four strategies.

```bash
conda activate llm
python scripts/benchmark_llm_prompt_strategies.py
# Output: outputs/tables/llm_prompt_strategies.csv
```

---

## 4 — TTS Benchmarks

Both scripts use EmergentTTS-Eval (20 samples).
**Metrics:** Round-trip WER (%), Speaker Similarity (cosine), Avg Latency (s)

Round-trip WER pipeline:
```
text → TTS synthesize → Faster-Whisper-Small transcribe → WER(hypothesis, original_text)
```

Speaker similarity:
```
resemblyzer.VoiceEncoder embed(ref.wav)  vs  embed(synthesized.wav)  → cosine similarity
```

### GPT-SoVITS

GPT-SoVITS inference server must be running first:

```bash
# Terminal A — start GPT-SoVITS API server
conda activate gptsovits
cd code/GPT-SoVITS
python api_v2.py

# Terminal B — run benchmark
conda activate gptsovits
cd <project_root>
python scripts/benchmark_tts_gptsovits.py
# Output: outputs/tables/tts_gptsovits_benchmark.csv
```

### F5-TTS

```bash
conda activate gptsovits
python scripts/benchmark_tts_f5.py
# Output: outputs/tables/tts_f5_benchmark.csv
```

**Reference voice** for both TTS scripts: `data/ref_voice/ref.wav`
Transcript is set in `config.py → REF_TEXT`.

---

## 5 — Installing resemblyzer (first time)

```bash
conda activate gptsovits
pip install resemblyzer
# Downloads pretrained GE2E speaker encoder (~17 MB) on first run
```

---

## Output Files

| File | Content |
|---|---|
| `outputs/tables/asr_whisper_benchmark.csv` | Faster-Whisper-Small WER + RTFx |
| `outputs/tables/asr_moonshine_benchmark.csv` | Moonshine-Base WER + RTFx |
| `outputs/tables/asr_whisper_large_benchmark.csv` | Large-V3-Turbo WER + RTFx |
| `outputs/tables/llm_qwen_benchmark.csv` | Qwen MMLU accuracy per subject |
| `outputs/tables/llm_llama_benchmark.csv` | Llama MMLU accuracy per subject |
| `outputs/tables/llm_prompt_strategies.csv` | All models × all strategies |
| `outputs/tables/tts_gptsovits_benchmark.csv` | RT-WER, Speaker Sim, Latency |
| `outputs/tables/tts_f5_benchmark.csv` | RT-WER, Speaker Sim, Latency |
| `outputs/tts_audio/gptsovits_bench/` | Per-sample synthesized WAVs |
| `outputs/tts_audio/f5_bench/` | Per-sample synthesized WAVs |
