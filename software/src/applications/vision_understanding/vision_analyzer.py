"""视觉分析器 - 获取视频帧并调用 LLM 分析画面内容

核心功能：
1. 订阅 VisionService 发布的图像帧
2. 保存最新帧为临时图片
3. 调用火山引擎 Ark LLM 进行图片内容理解
4. 返回画面描述
"""

import os
import sys
import base64
import tempfile
import time
from typing import Optional, Dict, Any
from pathlib import Path

# 添加项目根目录到路径
_current_dir = Path(__file__).parent
_project_root = _current_dir.parents[2]  # software/src -> software -> homebot
sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from services.vision_service.vision import VisionSubscriber
from common.logging import get_logger
from configs.ai_config import get_ai_credentials

logger = get_logger(__name__)

# 默认火山引擎配置
DEFAULT_ARK_MODEL = "doubao-vision-lite-250225"
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# 默认 MiniMax 配置
DEFAULT_MINIMAX_API_HOST = "https://api.minimaxi.com"

# 默认 OpenAI 配置
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# 默认 cctq.ai (gpt-5.5) 配置
DEFAULT_CCTQ_MODEL = "gpt-5.5"
DEFAULT_CCTQ_BASE_URL = "https://www.cctq.ai/v1"


class VisionAnalyzer:
    """视觉分析器 - 捕获视频帧并进行 AI 分析

    支持多提供商：cctq(gpt-5.5) / MiniMax / 火山Ark / OpenAI，通过 VISION_PROVIDER
    环境变量切换。provider=cctq 时以 gpt-5.5 为主，失败自动回退 MiniMax。
    """

    def __init__(
        self,
        video_addr: str = "tcp://localhost:5560",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_ms: int = 5000
    ):
        """初始化视觉分析器

        Args:
            video_addr: VisionService PUB 地址
            api_key: 兼容参数，根据 provider 解释
            model: 模型 ID
            base_url: API 基础 URL
            timeout_ms: 视频帧获取超时（毫秒）
        """
        self.video_addr = video_addr
        self.timeout_ms = timeout_ms

        # 提供商选择
        self.provider = os.getenv("VISION_PROVIDER", "cctq")
        secrets = get_ai_credentials()

        # ---------- MiniMax 配置 ----------
        self.minimax_api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        if not self.minimax_api_key:
            self.minimax_api_key = secrets.llm.api_key  # 复用 LLM Key
        self.minimax_api_host = base_url or os.getenv("MINIMAX_API_HOST", "")
        if not self.minimax_api_host and secrets.llm.api_url:
            # 从 LLM_API_URL 提取 host，例如 https://api.minimax.chat/v1 -> https://api.minimax.chat
            from urllib.parse import urlparse
            parsed = urlparse(secrets.llm.api_url)
            self.minimax_api_host = f"{parsed.scheme}://{parsed.netloc}"
        if not self.minimax_api_host:
            self.minimax_api_host = DEFAULT_MINIMAX_API_HOST
        self.minimax_api_host = self.minimax_api_host.rstrip("/")

        # ---------- 火山 Ark 配置 ----------
        self.ark_api_key = api_key or os.getenv("ARK_API_KEY", "")
        if not self.ark_api_key and secrets.vision.api_key:
            self.ark_api_key = secrets.vision.api_key
        if not self.ark_api_key and secrets.tts.access_token:
            self.ark_api_key = secrets.tts.access_token

        self.ark_model = model or os.getenv("ARK_MODEL_ID", DEFAULT_ARK_MODEL)
        self.ark_base_url = base_url or os.getenv("ARK_BASE_URL", DEFAULT_ARK_BASE_URL)
        if secrets.vision.api_url:
            self.ark_base_url = secrets.vision.api_url

        # ---------- OpenAI 配置 ----------
        self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.openai_api_key and secrets.vision.api_key:
            self.openai_api_key = secrets.vision.api_key
        if not self.openai_api_key and secrets.llm.api_key:
            self.openai_api_key = secrets.llm.api_key

        self.openai_base_url = base_url or os.getenv("OPENAI_API_URL", "")
        if not self.openai_base_url and secrets.vision.api_url:
            self.openai_base_url = secrets.vision.api_url
        if not self.openai_base_url:
            self.openai_base_url = DEFAULT_OPENAI_BASE_URL

        # ---------- cctq.ai (gpt-5.5) 配置 ----------
        self.cctq_api_key = (
            api_key
            or os.getenv("CCTQ_API_KEY", "")
            or os.getenv("LLM_API_KEY", "")
            or secrets.llm.api_key
        )
        self.cctq_base_url = (
            base_url
            or os.getenv("CCTQ_API_URL", "")
            or os.getenv("LLM_API_URL", "")
            or DEFAULT_CCTQ_BASE_URL
        )
        self.cctq_model = (
            model
            or os.getenv("CCTQ_VISION_MODEL", "")
            or os.getenv("LLM_MODEL", "")
            or DEFAULT_CCTQ_MODEL
        )

        # 统一模型名称
        if self.provider == "minimax":
            self.model = model or ""
        elif self.provider == "cctq":
            self.model = self.cctq_model
        elif self.provider == "openai":
            self.model = model or os.getenv("OPENAI_MODEL", "")
            if not self.model and secrets.vision.model:
                self.model = secrets.vision.model
            if not self.model:
                self.model = DEFAULT_OPENAI_MODEL
        else:
            self.model = self.ark_model

        # 视频订阅器
        self._subscriber: Optional[VisionSubscriber] = None

        # 临时目录
        self._temp_dir = tempfile.mkdtemp(prefix="homebot_vision_")

        logger.info(
            f"VisionAnalyzer initialized, provider={self.provider}, "
            f"video_addr={video_addr}"
        )
    
    def _ensure_subscriber(self) -> VisionSubscriber:
        """确保视频订阅器已启动
        
        Returns:
            VisionSubscriber 实例
        """
        if self._subscriber is None:
            self._subscriber = VisionSubscriber(self.video_addr)
            self._subscriber.start()
            logger.info(f"VisionSubscriber started, connecting to {self.video_addr}")
            # 等待订阅器接收第一帧
            time.sleep(0.5)
        return self._subscriber
    
    def capture_frame(self, output_path: Optional[str] = None) -> Optional[str]:
        """捕获最新视频帧并保存为图片
        
        Args:
            output_path: 图片保存路径，默认使用临时目录
            
        Returns:
            保存的图片路径，失败返回 None
        """
        subscriber = self._ensure_subscriber()
        
        # 尝试多次获取帧
        max_retries = 3
        for attempt in range(max_retries):
            frame_id, frame = subscriber.read_frame()
            if frame is not None:
                break
            logger.warning(f"Frame not ready, retry {attempt + 1}/{max_retries}")
            time.sleep(0.1)
        else:
            logger.error("Failed to get frame from video service")
            return None
        
        # 确定保存路径
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self._temp_dir, f"capture_{timestamp}.jpg")
        
        # 保存图片
        try:
            cv2.imwrite(output_path, frame)
            logger.info(f"Frame saved to {output_path}, frame_id={frame_id}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return None
    
    def encode_image(self, image_path: str) -> str:
        """将图片文件编码为 base64 字符串
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            base64 编码字符串
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def analyze(
        self,
        image_path: str,
        prompt: str = "请描述这张图片的内容",
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """分析图片内容

        根据配置的 provider 自动选择 MiniMax 或火山 Ark。

        Args:
            image_path: 图片文件路径
            prompt: 对图片的提问或指令
            max_tokens: 最大输出 token 数（Ark 专用）

        Returns:
            包含状态和结果的字典
        """
        if not os.path.exists(image_path):
            return {
                "status": "error",
                "message": f"图片文件不存在: {image_path}"
            }

        if self.provider == "cctq":
            result = self._analyze_cctq(image_path, prompt, max_tokens)
            if result.get("status") != "success":
                logger.warning(f"cctq(gpt-5.5) 视觉分析失败，回退 MiniMax: {result.get('message')}")
                return self._analyze_minimax(image_path, prompt)
            return result
        elif self.provider == "minimax":
            return self._analyze_minimax(image_path, prompt)
        elif self.provider == "openai":
            return self._analyze_openai(image_path, prompt, max_tokens)
        else:
            return self._analyze_ark(image_path, prompt, max_tokens)

    def _create_minimax_image_url(self, image_path: str) -> str:
        """创建 MiniMax 支持的 image_url 格式"""
        if image_path.startswith(("http://", "https://")):
            return image_path
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")
        b64 = self.encode_image(image_path)
        return f"data:{mime_type};base64,{b64}"

    def _analyze_minimax(
        self,
        image_path: str,
        prompt: str
    ) -> Dict[str, Any]:
        """调用 MiniMax VLM 分析图片

        使用原生 /v1/coding_plan/vlm 端点。
        """
        if not self.minimax_api_key:
            return {
                "status": "error",
                "message": "MiniMax API Key 未配置，请在 .env.local 中设置 LLM_API_KEY 或 MINIMAX_API_KEY"
            }

        try:
            import requests
        except ImportError:
            return {
                "status": "error",
                "message": "未安装 requests"
            }

        try:
            image_url = self._create_minimax_image_url(image_path)
            url = f"{self.minimax_api_host}/v1/coding_plan/vlm"
            headers = {
                "Authorization": f"Bearer {self.minimax_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "prompt": prompt,
                "image_url": image_url,
            }

            logger.info(f"Sending analysis request to MiniMax VLM, host={self.minimax_api_host}")
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            base_resp = data.get("base_resp", {})
            status_code = base_resp.get("status_code", -1)
            if status_code != 0:
                status_msg = base_resp.get("status_msg", "unknown error")
                raise RuntimeError(f"MiniMax API 错误 [{status_code}]: {status_msg}")

            result_text = data.get("content", "")
            logger.info("MiniMax VLM analysis completed successfully")

            return {
                "status": "success",
                "description": result_text,
                "image_path": image_path
            }

        except Exception as e:
            logger.error(f"MiniMax VLM analysis failed: {e}")
            return {
                "status": "error",
                "message": f"图像分析失败: {e}"
            }

    def _analyze_ark(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """调用火山引擎 Ark LLM 分析图片

        Args:
            image_path: 图片文件路径
            prompt: 对图片的提问或指令
            max_tokens: 最大输出 token 数

        Returns:
            包含状态和结果的字典
        """
        if not self.ark_api_key:
            return {
                "status": "error",
                "message": "Ark API Key 未配置，请设置 ARK_API_KEY 环境变量，或在 .env.local 中配置 VISION_API_KEY 或 VOLCANO_ACCESS_TOKEN"
            }

        try:
            # 尝试导入火山引擎 SDK
            try:
                from volcenginesdkarkruntime import Ark
            except ImportError:
                return {
                    "status": "error",
                    "message": "未安装 volcenginesdkarkruntime，请运行: pip install volcenginesdkarkruntime"
                }

            # 初始化客户端
            client = Ark(base_url=self.ark_base_url, api_key=self.ark_api_key)

            # 编码图片
            base64_image = self.encode_image(image_path)

            # 构建消息内容
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]

            # 发送请求
            logger.info(f"Sending analysis request to Ark LLM, model={self.ark_model}")
            response = client.chat.completions.create(
                model=self.ark_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                stream=False
            )

            result_text = response.choices[0].message.content
            logger.info("Ark analysis completed successfully")

            return {
                "status": "success",
                "description": result_text,
                "image_path": image_path
            }

        except Exception as e:
            logger.error(f"Ark analysis failed: {e}")
            return {
                "status": "error",
                "message": f"图像分析失败: {e}"
            }

    def _analyze_openai_compatible(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int,
        api_key: str,
        base_url: str,
        model: str,
        label: str,
    ) -> Dict[str, Any]:
        """调用 OpenAI 兼容接口（OpenAI 官方 / cctq.ai 等）分析图片

        Args:
            image_path: 图片文件路径
            prompt: 对图片的提问或指令
            max_tokens: 最大输出 token 数
            api_key: API 密钥
            base_url: API 基础 URL
            model: 模型名称
            label: 日志/错误信息中显示的提供商名

        Returns:
            包含状态和结果的字典
        """
        if not api_key:
            return {
                "status": "error",
                "message": f"{label} API Key 未配置，请在 .env.local 中配置对应密钥"
            }

        try:
            from services.llm_service.llm_client import OpenAIOfficialClient

            client = OpenAIOfficialClient(api_key=api_key, base_url=base_url)

            base64_image = self.encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            logger.info(f"Sending analysis request to {label}, model={model}")
            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.9,
            )

            result_text = response["choices"][0]["message"]["content"]
            logger.info(f"{label} analysis completed successfully")

            return {
                "status": "success",
                "description": result_text,
                "image_path": image_path
            }

        except Exception as e:
            logger.error(f"{label} analysis failed: {e}")
            return {
                "status": "error",
                "message": f"图像分析失败: {e}"
            }

    def _analyze_cctq(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """调用 cctq.ai gpt-5.5 多模态模型分析图片"""
        return self._analyze_openai_compatible(
            image_path, prompt, max_tokens,
            api_key=self.cctq_api_key,
            base_url=self.cctq_base_url,
            model=self.cctq_model,
            label="cctq(gpt-5.5)",
        )

    def _analyze_openai(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """调用 OpenAI GPT-4V/GPT-4o 分析图片"""
        return self._analyze_openai_compatible(
            image_path, prompt, max_tokens,
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            model=self.model,
            label="OpenAI",
        )

    def capture_and_analyze(
        self,
        prompt: str = "请描述这张图片的内容",
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """一键捕获并分析画面
        
        Args:
            prompt: 对图片的提问或指令
            max_tokens: 最大输出 token 数
            
        Returns:
            包含状态和结果的字典
        """
        # 捕获帧
        image_path = self.capture_frame()
        if image_path is None:
            return {
                "status": "error",
                "message": "无法获取视频帧，请检查视觉服务是否已启动"
            }
        
        # 分析图片
        return self.analyze(image_path, prompt, max_tokens)
    
    def close(self):
        """关闭资源"""
        if self._subscriber:
            self._subscriber.stop()
            self._subscriber = None
            logger.info("VisionSubscriber stopped")
        
        # 清理临时文件
        try:
            import shutil
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
                logger.info(f"Temp directory cleaned: {self._temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean temp directory: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
        return False


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HomeBot 视觉理解工具 - 捕获画面并调用 LLM 分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（描述画面内容）
  python -m applications.vision_understanding

  # 自定义提问
  python -m applications.vision_understanding -p "图中有几个人？"

  # 指定视频服务地址
  python -m applications.vision_understanding --video-addr tcp://192.168.1.100:5560

  # 使用特定模型
  python -m applications.vision_understanding --model doubao-vision-pro-250226
        """
    )
    
    parser.add_argument(
        "-p", "--prompt",
        default="请描述这张图片的内容",
        help="对图片的提问或指令 (默认: 请描述这张图片的内容)"
    )
    
    parser.add_argument(
        "--video-addr",
        default="tcp://localhost:5560",
        help="VisionService PUB 地址 (默认: tcp://localhost:5560)"
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help="模型 ID (默认从 ARK_MODEL_ID 环境变量获取)"
    )
    
    parser.add_argument(
        "--api-key",
        default=None,
        help="API Key (默认从 ARK_API_KEY 环境变量获取)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="图片保存路径 (默认使用临时目录)"
    )
    
    parser.add_argument(
        "--save-image",
        action="store_true",
        help="保存捕获的图片到当前目录"
    )
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = VisionAnalyzer(
        video_addr=args.video_addr,
        api_key=args.api_key,
        model=args.model
    )
    
    try:
        # 捕获帧
        print(f"[INFO] 正在连接视频服务 {args.video_addr}...")
        image_path = analyzer.capture_frame(args.output)
        if image_path is None:
            print("[ERROR] 无法获取视频帧，请检查视觉服务是否已启动")
            sys.exit(1)
        
        print(f"[OK] 图像捕获成功: {image_path}")
        
        # 如果需要保存到当前目录
        if args.save_image and args.output is None:
            saved_path = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            import shutil
            shutil.copy(image_path, saved_path)
            print(f"[OK] 图片已保存到: {saved_path}")
        
        # 分析图片
        print(f"[INFO] 正在分析图片，提示词: {args.prompt}")
        print(f"[INFO] 使用模型: {analyzer.model}")
        print()
        
        result = analyzer.analyze(image_path, args.prompt)
        
        if result["status"] == "success":
            print("=" * 50)
            print("分析结果:")
            print("=" * 50)
            print(result["description"])
            print("=" * 50)
        else:
            print(f"[ERROR] {result['message']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
