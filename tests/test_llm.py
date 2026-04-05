"""
LLM quick test — run in 'llm' conda env
  python tests/test_llm.py

Prints RESULT_JSON: {...} on last stdout line for test.py to capture.
"""
import json
import os
import sys
import time
from pathlib import Path

# Windows DLL fix for llama-cpp-python CUDA
for _d in [
    r"D:\Softeware\miniconda\envs\llm\Library\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
]:
    if os.path.isdir(_d):
        os.add_dll_directory(_d)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from config import LLM1_PATH

PROMPT = "What is the difference between supervised and unsupervised learning? Two sentences max."


def main():
    print("[LLM] Loading Qwen2.5-1.5B Q4_K_M ...")
    from llama_cpp import Llama
    llm = Llama(model_path=str(LLM1_PATH), n_ctx=512, n_threads=4, n_gpu_layers=-1, verbose=False)

    # Warm-up with minimal prompt
    llm("Hi", max_tokens=1)

    print(f"[LLM] Prompt  : {PROMPT}")
    t0  = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user",   "content": PROMPT},
        ],
        max_tokens=80,
        temperature=0.0,
    )
    latency = time.perf_counter() - t0
    text    = out["choices"][0]["message"]["content"].strip()
    tokens  = out["usage"]["completion_tokens"]
    tok_s   = tokens / latency

    print(f"[LLM] Response: {text}")
    print(f"[LLM] Latency : {latency:.3f}s   {tokens} tokens   {tok_s:.1f} tok/s")

    result = {
        "model":     "qwen2.5-1.5b-q4_k_m",
        "latency_s": round(latency, 3),
        "tokens":    tokens,
        "tok_s":     round(tok_s, 1),
        "response":  text,
    }
    print(f"RESULT_JSON: {json.dumps(result)}")


if __name__ == "__main__":
    main()
