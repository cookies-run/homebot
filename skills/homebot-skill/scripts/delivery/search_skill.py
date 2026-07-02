"""搜索技能：利用 VLM 在画面中寻找目标并判断可抓取性。

封装视觉帧捕获与 VLM 调用，返回目标 bbox、高度、姿态、可抓取性。
作为外部可复用技能，支持独立运行或被 applications/delivery_agent 导入。
"""
import os
import sys
import tempfile
import time
from typing import Optional

# 支持独立运行：将 software/src 加入路径以导入 common/configs
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../software/src"))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import cv2
import numpy as np

from common.logging import get_logger

logger = get_logger(__name__)


class SearchSkill:
    """视觉搜索技能。"""

    def __init__(self, vision_adapter=None, provider: str = "minimax"):
        self.vision_adapter = vision_adapter
        self.provider = provider
        self._temp_dir = tempfile.mkdtemp(prefix="delivery_search_")

    def search(self, target: str, max_retries: int = 2) -> dict:
        """搜索目标。

        Returns:
            {
                "found": bool,
                "bbox": [x1, y1, x2, y2] normalized,
                "height_cm": float,
                "pose": "upright" | "fallen" | "unknown",
                "reason": str
            }
        """
        if self.vision_adapter is None:
            return {"found": False, "reason": "视觉适配器未初始化"}

        last_reason = "多次尝试未在画面中找到目标"
        for attempt in range(max_retries + 1):
            frame_id, frame = self.vision_adapter.read_frame()
            if frame is None:
                last_reason = "未读取到摄像头画面（机身摄像头无帧）"
                time.sleep(0.2)
                continue

            path = os.path.join(self._temp_dir, f"search_{int(time.time()*1000)}.jpg")
            cv2.imwrite(path, frame)

            result = self._call_vlm(path, target)
            if result is None:
                last_reason = "VLM 无响应或返回无法解析（可能被截断/非 JSON）"
                continue
            if result.get("found"):
                return result
            last_reason = result.get("reason") or "模型判定画面中无该目标"

        return {"found": False, "reason": last_reason}

    def search_with_scan(self, target: str, rotate_fn, rotate_deg: float = 120.0,
                         max_rotations: int = 3, settle_s: float = 0.6) -> dict:
        """原地旋转扫描搜索单个目标。

        先看当前朝向；找不到则调用 rotate_fn 旋转 rotate_deg 后再看，如此循环，
        最多旋转 max_rotations 次（默认 3 次 * 120° = 转满一圈回到起始朝向）。
        因此最多分析 max_rotations + 1 帧画面。旋转由外部注入以解耦底盘实现。

        Args:
            target: 目标描述
            rotate_fn: 无参回调，执行一次底盘旋转（角度由调用方绑定）
            rotate_deg: 单次旋转角度，仅用于结果记录
            max_rotations: 最大旋转次数（转满一圈所需的次数）
            settle_s: 旋转后等待画面稳定的秒数

        Returns:
            {
                "found": bool,
                "views_used": int,       # 实际分析的画面数
                "rotations_used": int,   # 实际成功旋转的次数
                "bbox": [...], "height_cm": float, "pose": str,
                "graspable": bool,
                "reason": str
            }
        """
        views_used = 0
        rotations_used = 0
        while True:
            # 每个朝向只取一帧分析一次；转一圈已覆盖各角度，无需在原地重复截帧
            result = self.search(target, max_retries=0)
            views_used += 1
            if result.get("found"):
                result["graspable"] = self.check_graspable(result).get("graspable", False)
                result["views_used"] = views_used
                result["rotations_used"] = rotations_used
                return result
            if rotations_used >= max_rotations:
                break
            try:
                rotate_fn()
            except Exception as e:
                logger.error(f"旋转失败，中止扫描: {e}")
                return {
                    "found": False,
                    "views_used": views_used,
                    "rotations_used": rotations_used,
                    "reason": f"旋转失败: {e}",
                }
            rotations_used += 1
            time.sleep(settle_s)

        return {
            "found": False,
            "views_used": views_used,
            "rotations_used": rotations_used,
            "reason": f"旋转扫描一圈（旋转 {rotations_used} 次，约 {rotate_deg * rotations_used:.0f}°）"
                      f"未在 {views_used} 帧画面中找到目标",
        }

    def _call_vlm(self, image_path: str, target: str) -> Optional[dict]:
        """调用 VLM 搜索目标。"""
        prompt = f'''你是机器人视觉定位助手。判断图片中是否存在目标物体："{target}"。

匹配放宽：允许近义词/同类物体、部分遮挡、侧放或倒放、处于远处的小目标——只要能合理相信它就是该物体即可判定存在；画面中有多个时，选最可能、最完整的一个。是否存在只取决于能否看到该物体，不要因为估不出高度或姿态而判定为不存在。

仅输出以下 JSON，不要 markdown 代码块、不要解释、不要多余文字：
{{
  "found": true/false,
  "bbox": [x1, y1, x2, y2],
  "pose": "upright" | "fallen" | "unknown",
  "height_cm": 估计高度厘米数（可选，估不出填 null）,
  "reason": "简短说明"
}}

bbox 用图片左上角为原点的 0~1 归一化坐标，顺序为 [左, 上, 右, 下]。'''

        try:
            if self.provider == "minimax":
                return self._call_minimax(image_path, prompt)
            else:
                return self._call_openai_compatible(image_path, prompt)
        except Exception as e:
            logger.error(f"VLM 调用失败: {e}")
            return None

    def _call_minimax(self, image_path: str, prompt: str) -> Optional[dict]:
        try:
            import requests
        except ImportError:
            logger.error("未安装 requests")
            return None

        try:
            from configs.ai_config import get_ai_credentials
            secrets = get_ai_credentials()
            api_key = secrets.llm.api_key
            api_url = secrets.llm.api_url or "https://api.minimax.chat"
            if not api_key:
                logger.error("LLM API Key 未配置")
                return None

            from urllib.parse import urlparse
            parsed = urlparse(api_url)
            host = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme else api_url.rstrip("/v1")

            with open(image_path, "rb") as f:
                import base64
                b64 = base64.b64encode(f.read()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{b64}"

            resp = requests.post(
                f"{host}/v1/coding_plan/vlm",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"prompt": prompt, "image_url": image_url},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("text", "")
            return self._parse_json(text)
        except Exception as e:
            logger.error(f"MiniMax VLM 失败: {e}")
            return None

    def _call_openai_compatible(self, image_path: str, prompt: str) -> Optional[dict]:
        try:
            from configs.config import get_config
            from services.llm_service.llm_client import get_llm_client

            client = get_llm_client()
            cfg = get_config().llm

            import base64
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            response = client.chat_completion(
                model=cfg.model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]}
                ],
                temperature=0.1,
                max_tokens=1024,
                top_p=0.9,
            )
            text = response["choices"][0]["message"]["content"] or ""
            return self._parse_json(text)
        except Exception as e:
            logger.error(f"OpenAI-compatible VLM 失败: {e}")
            return None

    def _parse_json(self, text: str) -> Optional[dict]:
        """鲁棒解析 JSON。"""
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    start = i + 1
                    break
            end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()

        import json
        import re
        # 去除 think 标签
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return None
            # bbox 兼容 0~1000
            bbox = data.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4 and max(bbox) > 1.0:
                data["bbox"] = [v / 1000.0 for v in bbox]
            return data
        except Exception as e:
            logger.error(f"JSON 解析失败: {e}, raw={text[:200]}")
            return None

    def check_graspable(self, grab_info: dict) -> dict:
        """根据搜索返回信息判断可抓取性。"""
        if not grab_info.get("found"):
            return {"graspable": False, "reason": "未找到目标"}

        height = grab_info.get("height_cm")
        if height is not None and height > 25:
            return {"graspable": False, "reason": f"目标高度约{height}cm，超出机械臂安全抓取范围"}

        bbox = grab_info.get("bbox")
        if bbox:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < 0.005:
                return {"graspable": False, "reason": "目标在画面中过小，无法稳定抓取"}

        return {"graspable": True, "reason": "目标可抓取"}
