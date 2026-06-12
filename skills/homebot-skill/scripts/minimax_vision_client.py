#!/usr/bin/env python3
"""
MiniMax VLM 客户端
调用 MiniMax /v1/coding_plan/vlm 端点进行图像理解

注意：MiniMax 视觉 API 与标准聊天 API 是两套独立系统，
必须使用 /v1/coding_plan/vlm 端点，标准端点会静默忽略图片。

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


def analyze_images(
    image_paths: List[str],
    prompt: str = "请描述这张图片的内容",
    api_key: Optional[str] = None,
    api_host: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """
    调用 MiniMax VLM 分析图片内容

    Args:
        image_paths: 图片文件路径列表（目前 MiniMax VLM 只支持单图，取第一张）
        prompt: 对图片的提问或指令
        api_key: API 密钥，默认从 MINIMAX_API_KEY 环境变量读取
        api_host: API 地址，默认从 MINIMAX_API_HOST 环境变量读取
        timeout: 请求超时秒数

    Returns:
        VLM 返回的文本内容

    Raises:
        ValueError: API Key 未配置
        RuntimeError: API 返回错误
    """
    api_key = api_key or os.getenv("MINIMAX_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY 或 LLM_API_KEY 未配置！请设置环境变量:\n"
            "  export MINIMAX_API_KEY=your_api_key"
        )

    # 优先 MINIMAX_API_HOST，其次从 LLM_API_URL 提取 base host，默认国内端点
    if not api_host:
        api_host = os.getenv("MINIMAX_API_HOST")
    if not api_host:
        llm_url = os.getenv("LLM_API_URL", "")
        if llm_url:
            # 去掉 /v1 等后缀，保留 https://host 部分
            from urllib.parse import urlparse
            parsed = urlparse(llm_url)
            api_host = f"{parsed.scheme}://{parsed.netloc}"
    if not api_host:
        api_host = "https://api.minimaxi.com"
    api_host = api_host.rstrip("/")

    if not image_paths:
        raise ValueError("至少需要一张图片")

    # MiniMax VLM 目前只支持单图分析
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
