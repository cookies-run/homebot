#!/usr/bin/env python3
"""
cctq.ai 视觉客户端（OpenAI 兼容协议，gpt-5.5 原生多模态）

作为抓取/视觉流程的首选 VLM；调用失败时由调用方（grab_optimized）按优先级
回退到 MiniMax / 火山引擎。

环境变量:
    CCTQ_API_KEY / LLM_API_KEY     - API 密钥 (必填)
    CCTQ_API_URL / LLM_API_URL     - API 地址 (默认: https://www.cctq.ai/v1)
    CCTQ_VISION_MODEL / LLM_MODEL  - 模型名称 (默认: gpt-5.5)
"""

import os
import sys
import base64
import argparse
from typing import List, Dict, Any, Optional

DEFAULT_API_URL = "https://www.cctq.ai/v1"
DEFAULT_MODEL = "gpt-5.5"


def encode_image(image_path: str) -> str:
    """将本地图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_image_url(image_path: str) -> str:
    """创建 OpenAI 兼容的 image_url（本地图片转 data URI）"""
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
    return f"data:{mime_type};base64,{encode_image(image_path)}"


def _resolve_config(
    api_key: Optional[str], api_url: Optional[str], model: Optional[str]
) -> tuple[str, str, str]:
    """解析 API key / host / model，支持 CCTQ_* 与通用 LLM_* 两套环境变量"""
    api_key = api_key or os.getenv("CCTQ_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "CCTQ_API_KEY 或 LLM_API_KEY 未配置！请在 .env.local 中设置 LLM_API_KEY"
        )
    api_url = (
        api_url
        or os.getenv("CCTQ_API_URL")
        or os.getenv("LLM_API_URL")
        or DEFAULT_API_URL
    ).rstrip("/")
    model = model or os.getenv("CCTQ_VISION_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    return api_key, api_url, model


def analyze_images(
    image_paths: List[str],
    prompt: str = "请描述这张图片的内容",
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 512,
    timeout: int = 60,
) -> str:
    """
    调用 cctq.ai gpt-5.5 多模态接口分析图片内容（OpenAI /chat/completions 协议）。

    Args:
        image_paths: 图片文件路径列表（gpt-5.5 支持多图）
        prompt: 对图片的提问或指令
        api_key: API 密钥，默认从 CCTQ_API_KEY / LLM_API_KEY 读取
        api_url: API 地址，默认从 CCTQ_API_URL / LLM_API_URL 读取
        model: 模型名称，默认 gpt-5.5
        max_tokens: 最大输出 token 数
        timeout: 请求超时秒数

    Returns:
        模型返回的文本内容
    """
    api_key, api_url, model = _resolve_config(api_key, api_url, model)

    if not image_paths:
        raise ValueError("至少需要一张图片")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_url, timeout=timeout)

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": create_image_url(path)},
        })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0.1,
    )

    choices = response.choices
    if not choices:
        raise RuntimeError("cctq.ai 返回为空 choices")
    return choices[0].message.content or ""


def main():
    parser = argparse.ArgumentParser(
        description="cctq.ai (gpt-5.5) VLM 图像理解工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cctq_vision_client.py image.jpg
  python cctq_vision_client.py image.jpg -p "图中有几个人？"
        """
    )
    parser.add_argument("image", help="图片文件路径或 URL")
    parser.add_argument("-p", "--prompt", default="请描述这张图片的内容", help="分析提示词")
    parser.add_argument("--api-key", help="API 密钥")
    parser.add_argument("--api-url", help="API 地址")
    parser.add_argument("--model", help="模型名称")
    args = parser.parse_args()

    try:
        result = analyze_images(
            image_paths=[args.image],
            prompt=args.prompt,
            api_key=args.api_key,
            api_url=args.api_url,
            model=args.model,
        )
        print(result)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
