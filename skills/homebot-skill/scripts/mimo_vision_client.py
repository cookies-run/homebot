#!/usr/bin/env python3
"""
Xiaomi MiMo VLM 客户端（OpenAI 兼容协议）

用于替代或回退 MiniMax VLM，当 MiniMax 达到 Token Plan 用量上限时
自动切换到 MiMo 视觉模型。

环境变量:
    MIMO_API_KEY   - API 密钥 (必填)
    MIMO_API_HOST  - API 地址 (默认: https://token-plan-cn.xiaomimimo.com/v1)
    MIMO_MODEL     - 模型名称 (默认: mimo-v2-omni)
"""

import os
import sys
import base64
import argparse
import requests
from typing import List, Dict, Any, Optional


def encode_image(image_path: str) -> str:
    """将本地图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_image_content(image_path: str) -> Dict[str, Any]:
    """创建 OpenAI 兼容格式的图片内容对象"""
    if image_path.startswith(("http://", "https://")):
        return {
            "type": "image_url",
            "image_url": {"url": image_path},
        }

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_types.get(ext, "image/jpeg")
    b64 = encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }


def analyze_images(
    image_paths: List[str],
    prompt: str = "请描述这张图片的内容",
    api_key: Optional[str] = None,
    api_host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 60,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    调用 Xiaomi MiMo 视觉模型分析图片内容

    Args:
        image_paths: 图片文件路径列表（MiMo OpenAI 兼容端点多图可用）
        prompt: 对图片的提问或指令
        api_key: API 密钥，默认从 MIMO_API_KEY 环境变量读取
        api_host: API 地址，默认从 MIMO_API_HOST 环境变量读取
        model: 模型名称，默认从 MIMO_MODEL 环境变量读取
        timeout: 请求超时秒数
        max_tokens: 最大输出 token 数
        temperature: 采样温度

    Returns:
        VLM 返回的文本内容

    Raises:
        ValueError: API Key 未配置
        RuntimeError: API 返回错误
    """
    api_key = api_key or os.getenv("MIMO_API_KEY")
    if not api_key:
        raise ValueError(
            "MIMO_API_KEY 未配置！请设置环境变量:\n"
            "  export MIMO_API_KEY=your_api_key"
        )

    api_host = api_host or os.getenv("MIMO_API_HOST", "https://token-plan-cn.xiaomimimo.com/v1")
    api_host = api_host.rstrip("/")

    model = model or os.getenv("MIMO_MODEL", "mimo-v2-omni")

    if not image_paths:
        raise ValueError("至少需要一张图片")

    content = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append(create_image_content(path))

    url = f"{api_host}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"MiMo API 请求失败: {e}")

    data = resp.json()
    if data.get("error"):
        err = data["error"]
        raise RuntimeError(
            f"MiMo API 错误 [{err.get('code', 'unknown')}]: {err.get('message', 'unknown error')}"
        )

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("MiMo API 返回空 choices")

    message = choices[0].get("message", {})
    return message.get("content", "")


def main():
    parser = argparse.ArgumentParser(
        description="Xiaomi MiMo VLM 图像理解工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python mimo_vision_client.py image.jpg
  python mimo_vision_client.py image.jpg -p "图中有几个人？"
  python mimo_vision_client.py image1.jpg image2.png -p "比较这两张图"
        """
    )
    parser.add_argument("images", nargs="+", help="图片文件路径或 URL")
    parser.add_argument("-p", "--prompt", default="请描述这张图片的内容", help="分析提示词")
    parser.add_argument("--api-key", help="API 密钥")
    parser.add_argument("--api-host", help="API 地址")
    parser.add_argument("-m", "--model", help="模型名称")
    args = parser.parse_args()

    try:
        result = analyze_images(
            image_paths=args.images,
            prompt=args.prompt,
            api_key=args.api_key,
            api_host=args.api_host,
            model=args.model,
        )
        print(result)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
