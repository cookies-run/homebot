#!/usr/bin/env python3
"""
MiniMax 视觉客户端
支持：
  1. /v1/coding_plan/vlm 端点（原生 MiniMax VLM）
  2. /v1/chat/completions 端点（MiniMax-M2.7 原生多模态模型，OpenAI 兼容格式）

环境变量:
    MINIMAX_API_KEY  - API 密钥 (必填)
    MINIMAX_API_HOST - API 地址 (默认: https://api.minimaxi.com)
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


def create_image_url(image_path: str) -> str:
    """创建 MiniMax 支持的 image_url 格式"""
    if image_path.startswith(("http://", "https://")):
        return image_path

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
    return f"data:{mime_type};base64,{b64}"


def _get_api_key_and_host(api_key: Optional[str], api_host: Optional[str]) -> tuple[str, str]:
    """获取并规范化 API key 和 host"""
    api_key = api_key or os.getenv("MINIMAX_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY 或 LLM_API_KEY 未配置！请设置环境变量:\n"
            "  export MINIMAX_API_KEY=your_api_key"
        )

    if not api_host:
        api_host = os.getenv("MINIMAX_API_HOST")
    if not api_host:
        llm_url = os.getenv("LLM_API_URL", "")
        if llm_url:
            from urllib.parse import urlparse
            parsed = urlparse(llm_url)
            api_host = f"{parsed.scheme}://{parsed.netloc}"
    if not api_host:
        api_host = "https://api.minimaxi.com"
    api_host = api_host.rstrip("/")

    return api_key, api_host


def analyze_images(
    image_paths: List[str],
    prompt: str = "请描述这张图片的内容",
    api_key: Optional[str] = None,
    api_host: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """
    调用 MiniMax /v1/coding_plan/vlm 端点分析图片内容

    Args:
        image_paths: 图片文件路径列表（目前 MiniMax VLM 只支持单图，取第一张）
        prompt: 对图片的提问或指令
        api_key: API 密钥，默认从 MINIMAX_API_KEY 环境变量读取
        api_host: API 地址，默认从 MINIMAX_API_HOST 环境变量读取
        timeout: 请求超时秒数

    Returns:
        VLM 返回的文本内容
    """
    api_key, api_host = _get_api_key_and_host(api_key, api_host)

    if not image_paths:
        raise ValueError("至少需要一张图片")

    image_url = create_image_url(image_paths[0])

    url = f"{api_host}/v1/coding_plan/vlm"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "image_url": image_url,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"MiniMax API 请求失败: {e}")

    data = resp.json()
    base_resp = data.get("base_resp", {})
    status_code = base_resp.get("status_code", -1)
    if status_code != 0:
        status_msg = base_resp.get("status_msg", "unknown error")
        raise RuntimeError(f"MiniMax API 错误 [{status_code}]: {status_msg}")

    return data.get("content", "")


def analyze_images_m3(
    image_paths: List[str],
    prompt: str = "请描述这张图片的内容",
    api_key: Optional[str] = None,
    api_host: Optional[str] = None,
    timeout: int = 60,
    model: str = "MiniMax-M2.7",
) -> str:
    """
    调用 MiniMax 多模态模型分析图片。

    M2.7 不支持 /v1/chat/completions 的 image_url 格式，需要走原生 /v1/coding_plan/vlm 端点；
    M3 及后续支持 OpenAI 兼容 image_url 的模型走 /v1/chat/completions 端点。

    Args:
        image_paths: 图片文件路径列表（M2.7 仅使用第一张）
        prompt: 对图片的提问或指令
        api_key: API 密钥
        api_host: API 地址
        timeout: 请求超时秒数
        model: 模型 ID，默认 "MiniMax-M2.7"

    Returns:
        模型返回的文本内容
    """
    # M2.7 不支持 /v1/chat/completions 的 image_url 格式，需要走原生 VLM 端点
    if "M2" in model or "m2" in model:
        return analyze_images(
            image_paths=image_paths,
            prompt=prompt,
            api_key=api_key,
            api_host=api_host,
            timeout=timeout,
        )

    api_key, api_host = _get_api_key_and_host(api_key, api_host)

    if not image_paths:
        raise ValueError("至少需要一张图片")

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": create_image_url(path)},
        })

    url = f"{api_host}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"MiniMax Chat Completions API 请求失败: {e}")

    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"MiniMax Chat Completions API 错误: {data['error']}")

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("MiniMax Chat Completions API 返回为空 choices")

    message = choices[0].get("message", {})
    return message.get("content", "")


def main():
    parser = argparse.ArgumentParser(
        description="MiniMax VLM 图像理解工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python minimax_vision_client.py image.jpg
  python minimax_vision_client.py image.jpg -p "图中有几个人？"
  python minimax_vision_client.py https://example.com/image.jpg
        """
    )
    parser.add_argument("image", help="图片文件路径或 URL")
    parser.add_argument("-p", "--prompt", default="请描述这张图片的内容", help="分析提示词")
    parser.add_argument("--api-key", help="API 密钥")
    parser.add_argument("--api-host", help="API 地址")
    args = parser.parse_args()

    try:
        result = analyze_images(
            image_paths=[args.image],
            prompt=args.prompt,
            api_key=args.api_key,
            api_host=args.api_host,
        )
        print(result)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
