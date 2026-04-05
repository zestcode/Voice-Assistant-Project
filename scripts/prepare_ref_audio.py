"""
从 LibriSpeech 测试集中提取第一条样本作为 GPT-SoVITS 的参考音频。
提取完成后自动更新 config.py 中的 REF_TEXT。

Usage:
  python scripts/prepare_ref_audio.py
"""

import re
import sys
from pathlib import Path

import io
import soundfile as sf
from datasets import load_from_disk, Audio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ASR_BENCH_PATH, REF_AUDIO

def main():
    print(f"Loading LibriSpeech from {ASR_BENCH_PATH} ...")
    # decode=False: 只取原始 bytes，不经过 datasets 的音频解码器
    ds = load_from_disk(str(ASR_BENCH_PATH)).cast_column("audio", Audio(decode=False))
    sample = ds[0]

    audio_bytes = sample["audio"]["bytes"]
    text        = sample["text"]

    # 用 soundfile 直接解码
    array, sr = sf.read(io.BytesIO(audio_bytes))

    REF_AUDIO.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(REF_AUDIO), array, sr)
    print(f"Saved ref audio -> {REF_AUDIO}")
    print(f"Ref text: {text}")

    # 自动更新 config.py 中的 REF_TEXT
    config_path = Path(__file__).resolve().parent / "config.py"
    content     = config_path.read_text(encoding="utf-8")
    new_content = re.sub(
        r'REF_TEXT\s*=\s*".*?"',
        f'REF_TEXT  = "{text}"',
        content,
    )
    config_path.write_text(new_content, encoding="utf-8")
    print(f"Updated REF_TEXT in config.py")

if __name__ == "__main__":
    main()
