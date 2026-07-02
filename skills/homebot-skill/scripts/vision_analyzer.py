#!/usr/bin/env python3
"""
统一视觉分析适配器（VLM 路由层）。

把 MiniMax / cctq / 火山等底层视觉客户端封装成统一接口，业务技能不再自己写
HTTP 请求或处理各 provider 的响应字段差异。

用法：
    from vision_analyzer import VisionAnalyzer
    analyzer = VisionAnalyzer()
    text, provider = analyzer.analyze(image_paths=["frame.jpg"], prompt="描述图片")
"""
import os
import sys
from typing import List, Optional, Tuple

from common.logging import get_logger

logger = get_logger(__name__)


class VisionAnalyzer:
    """统一 VLM 分析器：按全局配置选择 provider，失败时按默认顺序回退。"""

    # 默认回退顺序（当全局配置未指定或指定不可用时）
    DEFAULT_ORDER = ["minimax", "cctq", "volcengine"]

    def __init__(self, preferred_provider: Optional[str] = None):
        """
        Args:
            preferred_provider: 强制指定首选 provider；None 时读取全局 ai_config.vision.provider
        """
        self.preferred_provider = preferred_provider
        self._clients = self._load_clients()
        self._order = self._resolve_order()

    def _load_clients(self) -> dict:
        """懒加载底层视觉客户端，只加载当前环境能导入的。"""
        clients = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        try:
            from minimax_vision_client import analyze_images_m3 as _minimax_analyze
            clients["minimax"] = _minimax_analyze
        except Exception as e:
            logger.debug(f"MiniMax 视觉客户端未加载: {e}")

        try:
            from cctq_vision_client import analyze_images as _cctq_analyze
            clients["cctq"] = _cctq_analyze
        except Exception as e:
            logger.debug(f"cctq 视觉客户端未加载: {e}")

        try:
            from volcengine_vision_client import analyze_images as _volcengine_analyze
            clients["volcengine"] = _volcengine_analyze
        except Exception as e:
            logger.debug(f"火山视觉客户端未加载: {e}")

        return clients

    def _resolve_order(self) -> List[str]:
        """根据全局配置决定调用顺序。"""
        preferred = self.preferred_provider
        if not preferred:
            try:
                from configs.ai_config import get_ai_credentials
                preferred = get_ai_credentials().vision.provider
            except Exception as e:
                logger.debug(f"读取全局 vision provider 失败: {e}")

        if preferred and preferred in self._clients:
            others = [n for n in self.DEFAULT_ORDER if n != preferred and n in self._clients]
            return [preferred] + others

        return [n for n in self.DEFAULT_ORDER if n in self._clients]

    def analyze(
        self,
        image_paths: List[str],
        prompt: str,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        reasoning_effort: str = "low",
    ) -> Tuple[str, str]:
        """
        调用 VLM 分析图片，按配置顺序回退。

        Args:
            image_paths: 图片路径列表
            prompt: 提示词
            max_tokens: 最大输出 token（可选）
            timeout: 单次调用超时
            reasoning_effort: 火山引擎专用参数

        Returns:
            (text, provider_name)

        Raises:
            RuntimeError: 所有 provider 均失败
        """
        if not self._order:
            raise RuntimeError("没有可用的视觉分析客户端")

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        last_err = None
        for name in self._order:
            fn = self._clients[name]
            try:
                logger.info(f"[VisionAnalyzer] 尝试 provider: {name}")
                kwargs = {
                    "image_paths": image_paths,
                    "prompt": prompt,
                    "timeout": timeout,
                }
                if name == "volcengine":
                    kwargs["max_tokens"] = max_tokens or 4096
                    kwargs["reasoning_effort"] = reasoning_effort
                elif name == "cctq":
                    kwargs["max_tokens"] = max_tokens or 512
                elif name == "minimax":
                    kwargs["max_tokens"] = max_tokens

                result = fn(**kwargs)
                logger.info(f"[VisionAnalyzer] provider={name} 调用成功")
                return result, name
            except Exception as e:
                logger.warning(f"[VisionAnalyzer] provider={name} 失败: {e}")
                last_err = e
                continue

        raise last_err or RuntimeError("所有 VLM provider 均失败")

    def get_order(self) -> List[str]:
        """返回当前解析出的 provider 顺序（调试用）。"""
        return self._order.copy()
