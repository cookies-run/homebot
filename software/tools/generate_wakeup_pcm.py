#!/usr/bin/env python3
"""用当前配置的 TTS 合成"我在"提示音

替换 cache/wozai.pcm，使唤醒提示音与对话 TTS 音色一致。
"""
import os
import sys
import wave
import io

# 将 src 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import httpx
from configs.config import get_config


def main():
    config = get_config().tts
    api_key = config.api_key
    api_url = config.api_url or "https://api.minimax.chat/v1"
    model = config.model or "speech-2.8-hd"
    voice_id = config.voice_type or "Chinese (Mandarin)_Cute_Spirit"

    if not api_key:
        print("错误: 未配置 MiniMax API Key")
        sys.exit(1)

    url = f"{api_url.rstrip('/')}/t2a_v2"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "text": "我在",
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1,
            "vol": 1,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 16000,
            "format": "wav",
            "channel": 1,
        },
    }

    print(f"正在用 MiniMax 合成唤醒提示音...")
    print(f"  model: {model}")
    print(f"  voice: {voice_id}")

    resp = httpx.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hex_audio = data.get("data", {}).get("audio", "")
    if not hex_audio:
        print("错误: MiniMax 返回空音频")
        sys.exit(1)

    wav_bytes = bytes.fromhex(hex_audio)

    # 解析 WAV，提取裸 PCM
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        pcm_data = wf.readframes(nframes)

    print(f"  WAV: {nchannels}ch, {sampwidth * 8}bit, {framerate}Hz, {nframes}frames")

    # 保存裸 PCM
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    pcm_path = os.path.join(cache_dir, "wozai.pcm")

    with open(pcm_path, "wb") as f:
        f.write(pcm_data)

    print(f"已保存: {pcm_path} ({len(pcm_data)} 字节)")


if __name__ == "__main__":
    main()
