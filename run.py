"""
Voice AI Assistant — Single Integration Launcher
Usage (in ANY conda env that has Python + requests):
  python run.py

What it does:
  1. Starts ASR server  (moonshine env,  :8001) as a subprocess
  2. Starts LLM server  (llm env,        :8002) as a subprocess
  3. Starts TTS server  (gptsovits env,  :8003) as a subprocess
  4. Polls /health until all three are ready (or times out)
  5. Launches Gradio UI at http://localhost:7860
  6. On Ctrl+C: kills all three servers cleanly
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
APP      = ROOT / "app"

CONDA_BASE = Path(r"D:\Softeware\miniconda")   # adjust if your miniconda is elsewhere

SERVERS = [
    {
        "name":   "ASR",
        "env":    "moonshine",
        "script": APP / "asr_server.py",
        "port":   8001,
        "url":    "http://localhost:8001/health",
    },
    {
        "name":   "LLM",
        "env":    "llm",
        "script": APP / "llm_server.py",
        "port":   8002,
        "url":    "http://localhost:8002/health",
    },
    {
        "name":   "TTS",
        "env":    "gptsovits",
        "script": APP / "tts_server.py",
        "port":   8003,
        "url":    "http://localhost:8003/health",
    },
]

UI_SCRIPT     = APP / "orchestrator.py"
STARTUP_TIMEOUT = 480   # seconds to wait for all servers to be healthy
POLL_INTERVAL   = 3     # seconds between health checks

# ── Helpers ───────────────────────────────────────────────────────────────────

def conda_python(env_name: str) -> str:
    """Return path to python.exe inside a conda env."""
    return str(CONDA_BASE / "envs" / env_name / "python.exe")


def is_healthy(url: str) -> bool:
    try:
        r = requests.get(url, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


CUDA_DLL_DIRS = [
    r"D:\Softeware\miniconda\envs\llm\Library\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
]

def start_server(srv: dict) -> subprocess.Popen:
    python = conda_python(srv["env"])
    if not Path(python).exists():
        print(f"  [WARN] Python not found at {python}")
        print(f"         Adjust CONDA_BASE in run.py (currently: {CONDA_BASE})")
        sys.exit(1)

    # Inject CUDA DLL paths into PATH so llama_cpp can find llama.dll
    env = os.environ.copy()
    extra = ";".join(d for d in CUDA_DLL_DIRS if os.path.isdir(d))
    if extra:
        env["PATH"] = extra + ";" + env.get("PATH", "")

    proc = subprocess.Popen(
        [python, str(srv["script"])],
        cwd=str(ROOT),
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    return proc


def wait_for_servers(servers_info: list[tuple[dict, subprocess.Popen]]) -> bool:
    print(f"\nWaiting up to {STARTUP_TIMEOUT}s for servers to be ready ...")
    deadline = time.time() + STARTUP_TIMEOUT
    pending  = {s["name"]: s["url"] for s, _ in servers_info}

    while pending and time.time() < deadline:
        for name in list(pending):
            if is_healthy(pending[name]):
                print(f"  [{name}] READY")
                del pending[name]
        if pending:
            time.sleep(POLL_INTERVAL)

    if pending:
        print(f"\n[ERROR] Timed out waiting for: {list(pending.keys())}")
        return False
    return True


def kill_servers(procs: list[subprocess.Popen]):
    print("\nShutting down servers ...")
    for p in procs:
        if p.poll() is None:
            if sys.platform == "win32":
                p.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                p.terminate()
    time.sleep(1)
    for p in procs:
        if p.poll() is None:
            p.kill()
    print("All servers stopped.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Voice AI Assistant — Integration Launcher")
    print("  ASR (moonshine:8001) | LLM (llm:8002) | TTS (gptsovits:8003)")
    print("=" * 60)

    # Check conda base exists
    if not CONDA_BASE.exists():
        print(f"\n[ERROR] CONDA_BASE not found: {CONDA_BASE}")
        print("  Edit the CONDA_BASE variable at the top of run.py")
        sys.exit(1)

    # Start all three servers
    procs = []
    server_pairs = []
    for srv in SERVERS:
        print(f"  Starting {srv['name']} server (env={srv['env']}, port={srv['port']}) ...")
        p = start_server(srv)
        procs.append(p)
        server_pairs.append((srv, p))

    # Wait for all to be healthy
    all_ready = wait_for_servers(server_pairs)
    if not all_ready:
        kill_servers(procs)
        sys.exit(1)

    print("\nAll servers ready! Launching Gradio UI ...")
    print("Open http://localhost:7860 in your browser\n")
    print("Press Ctrl+C to stop everything.\n")

    # Launch Gradio UI — dedicated voiceui env (gradio + requests only, no ML deps)
    ui_python = conda_python("voiceui")
    ui_proc = subprocess.Popen([ui_python, str(UI_SCRIPT)], cwd=str(ROOT))
    procs.append(ui_proc)

    try:
        ui_proc.wait()   # block until UI exits (Ctrl+C or browser close)
    except KeyboardInterrupt:
        pass
    finally:
        kill_servers(procs)


if __name__ == "__main__":
    main()
