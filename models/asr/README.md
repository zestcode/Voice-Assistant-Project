# ASR Models

This directory stores the ASR model weights. Git does not track the actual model files.
Download each model and place it in the subdirectory shown below.

## Directory Layout

```
models/asr/
├── faster-whisper-small/     # CTranslate2 weights (~462 MB)
└── moonshine-base/           # ONNX weights (~100 MB)
```

## Download Instructions

### faster-whisper-small

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Systran/faster-whisper-small',
    local_dir='models/asr/faster-whisper-small'
)
"
```

Or with the CLI:
```bash
huggingface-cli download Systran/faster-whisper-small \
    --local-dir models/asr/faster-whisper-small
```

### moonshine-base

```bash
huggingface-cli download UsefulSensors/moonshine-base \
    --local-dir models/asr/moonshine-base
```

### faster-whisper-large-v3-turbo (optional, accuracy reference only)

This model is loaded at runtime via its HuggingFace ID and does **not** need to be
stored locally. The benchmark script uses:
```python
ASR_LARGE_TURBO_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
```

## Verification

After downloading, run:
```bash
python scripts/config.py
```
and confirm `ASR1` and `ASR2` both show `OK`.
