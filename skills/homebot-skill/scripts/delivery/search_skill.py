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

# 把当前 skills 脚本目录加入路径，确保能导入 vision_analyzer
_scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _scripts_path not in sys.path:
    sys.path.insert(0, _scripts_path)

import cv2
import numpy as np

from common.logging import get_logger
from vision_analyzer import VisionAnalyzer

logger = get_logger(__name__)


class SearchSkill:
    """视觉搜索技能。"""

    def __init__(self, vision_adapter=None, provider: str = "minimax"):
        self.vision_adapter = vision_adapter
        self.provider = provider
        self._temp_dir = tempfile.mkdtemp(prefix="delivery_search_")
        self._analyzer = VisionAnalyzer(preferred_provider=provider)

    def search(self, target: str, max_retries: int = 2) -> dict:
        """搜索目标。

        Returns:
            {
                "found": bool,
                "bbox": [x1, y1, x2, y2] normalized,
                "height_cm": float,
                "pose": "upright" | "fallen" | "unknown",
                "scene_description": str,  # 对整张画面内容的简短描述
                "reason": str
            }
        """
        if self.vision_adapter is None:
            return {"found": False, "reason": "视觉适配器未初始化"}

        last_reason = "多次尝试未在画面中找到目标"
        for attempt in range(max_retries + 1):
            logger.info(f"[search] 第 {attempt + 1}/{max_retries + 1} 次尝试读取机身摄像头画面")
            frame_id, frame = self.vision_adapter.read_frame()
            if frame is None:
                last_reason = "未读取到摄像头画面（机身摄像头无帧）"
                logger.warning(f"[search] 未读取到画面，frame_id={frame_id}")
                time.sleep(0.2)
                continue

            path = os.path.join(self._temp_dir, f"search_{int(time.time()*1000)}.jpg")
            cv2.imwrite(path, frame)
            logger.info(f"[search] 已保存待分析图片: {path}")

            result = self._call_vlm(path, target)
            if result is None:
                last_reason = "VLM 无响应或返回无法解析（可能被截断/非 JSON）"
                logger.warning(f"[search] VLM 返回 None，继续尝试")
                continue
            scene_desc = result.get("scene_description") or "（模型未返回画面描述）"
            logger.info(f"[search] VLM 返回结果: {result}")
            logger.info(f"[search] 模型对画面描述: {scene_desc}")
            if result.get("found"):
                logger.info(f"[search] ✅ 找到目标: {target}")
                return result
            last_reason = result.get("reason") or "模型判定画面中无该目标"
            logger.info(f"[search] ❌ 未找到目标，模型 reason: {last_reason}")

        logger.warning(f"[search] 全部 {max_retries + 1} 次尝试失败，最终 reason: {last_reason}")
        return {"found": False, "reason": last_reason}

    def search_with_scan(self, target: str, rotate_fn, rotate_deg: float = 120.0,
                         max_rotations: int = 3, settle_s: float = 0.6,
                         align_fn=None, camera_hfov_deg: float = 60.0,
                         align_threshold: float = 0.1,
                         max_align_attempts: int = 3) -> dict:
        """原地旋转扫描搜索单个目标，找到后尽量让机身正面正对目标。

        先看当前朝向；找不到则调用 rotate_fn 旋转 rotate_deg 后再看，如此循环，
        最多旋转 max_rotations 次（默认 3 次 * 120° = 转满一圈回到起始朝向）。
        找到目标后，如果提供了 align_fn，会根据目标 bbox 中心与画面中心的偏差
        计算角度并微调机身朝向，使机身正面正对目标。

        Args:
            target: 目标描述
            rotate_fn: 无参回调，执行一次底盘旋转（角度由调用方绑定，用于大角度扫描）
            rotate_deg: 单次扫描旋转角度，仅用于结果记录
            max_rotations: 最大扫描次数（转满一圈所需的次数）
            settle_s: 旋转后等待画面稳定的秒数
            align_fn: 可选，单参数回调 align_fn(deg: float)，正数左转、负数右转，
                      用于找到目标后的精对准；不提供则只返回找到结果
            camera_hfov_deg: 机身摄像头水平视场角，用于把像素偏差换算成旋转角度
            align_threshold: 目标中心与画面中心偏差阈值（0~1 归一化），低于此值认为已对准
            max_align_attempts: 最大对准尝试次数

        Returns:
            {
                "found": bool,
                "views_used": int,       # 实际分析的画面数
                "rotations_used": int,   # 实际成功旋转的次数
                "alignments_used": int,  # 实际执行的精对准旋转次数
                "bbox": [...], "height_cm": float, "pose": str,
                "scene_description": str,
                "graspable": bool,
                "reason": str
            }
        """
        views_used = 0
        rotations_used = 0
        alignments_used = 0
        logger.info(f"[search_with_scan] 开始旋转扫描目标: {target}, rotate_deg={rotate_deg}, max_rotations={max_rotations}")
        while True:
            # 每个朝向只取一帧分析一次；转一圈已覆盖各角度，无需在原地重复截帧
            logger.info(f"[search_with_scan] 当前朝向 #{views_used + 1}（已旋转 {rotations_used} 次）")
            result = self.search(target, max_retries=0)
            views_used += 1
            logger.info(f"[search_with_scan] 第 {views_used} 帧结果: found={result.get('found')}")
            if result.get("found"):
                result["graspable"] = self.check_graspable(result).get("graspable", False)
                result["views_used"] = views_used
                result["rotations_used"] = rotations_used
                # 精对准：让机身正面正对目标
                if align_fn is not None:
                    logger.info(f"[search_with_scan] 找到目标，开始精对准")
                    result, aligned_views, aligned_times = self._align_to_target(
                        target, result, align_fn, camera_hfov_deg,
                        align_threshold, max_align_attempts, settle_s
                    )
                    views_used += aligned_views
                    alignments_used += aligned_times
                else:
                    logger.info(f"[search_with_scan] 找到目标，未提供 align_fn，跳过精对准")
                result["views_used"] = views_used
                result["rotations_used"] = rotations_used
                result["alignments_used"] = alignments_used
                logger.info(f"[search_with_scan] ✅ 扫描中找到目标，最终返回: {result}")
                return result
            if rotations_used >= max_rotations:
                logger.info(f"[search_with_scan] 已旋转 {rotations_used} 次，达到上限，结束扫描")
                break
            try:
                logger.info(f"[search_with_scan] 未找到，执行第 {rotations_used + 1} 次旋转")
                rotate_fn()
            except Exception as e:
                logger.error(f"旋转失败，中止扫描: {e}")
                return {
                    "found": False,
                    "views_used": views_used,
                    "rotations_used": rotations_used,
                    "alignments_used": alignments_used,
                    "reason": f"旋转失败: {e}",
                }
            rotations_used += 1
            logger.info(f"[search_with_scan] 旋转完成，等待 {settle_s}s 画面稳定")
            time.sleep(settle_s)

        final = {
            "found": False,
            "views_used": views_used,
            "rotations_used": rotations_used,
            "alignments_used": alignments_used,
            "reason": f"旋转扫描一圈（旋转 {rotations_used} 次，约 {rotate_deg * rotations_used:.0f}°）"
                      f"未在 {views_used} 帧画面中找到目标",
        }
        logger.info(f"[search_with_scan] 扫描结束，最终返回: {final}")
        return final

    def _align_to_target(self, target: str, current_result: dict, align_fn,
                         camera_hfov_deg: float, align_threshold: float,
                         max_align_attempts: int, settle_s: float) -> tuple:
        """找到目标后微调机身朝向，使目标位于画面中央。

        Returns:
            (final_result, extra_views_used, alignment_rotations_used)
        """
        extra_views = 0
        alignments = 0
        result = current_result
        for attempt in range(max_align_attempts):
            bbox = result.get("bbox")
            if not bbox or len(bbox) != 4:
                logger.warning(f"[_align_to_target] 缺少 bbox，停止对准")
                break
            center_x = (bbox[0] + bbox[2]) / 2.0
            offset = center_x - 0.5
            logger.info(f"[_align_to_target] 第 {attempt + 1} 次对准，目标中心 x={center_x:.3f}, 偏差={offset:+.3f}")
            if abs(offset) <= align_threshold:
                logger.info(f"[_align_to_target] 偏差在阈值 {align_threshold} 内，对准完成")
                break
            # 偏差 > 0 说明目标偏右，机身需右转（负角度）；偏差 < 0 说明偏左，机身左转（正角度）
            angle_deg = -offset * camera_hfov_deg
            # 限制最小/最大步长，避免抖动或过度旋转
            min_step = 5.0
            if 0 < abs(angle_deg) < min_step:
                angle_deg = min_step if angle_deg > 0 else -min_step
            logger.info(f"[_align_to_target] 执行对准旋转 angle={angle_deg:+.1f}°")
            try:
                align_fn(angle_deg)
            except Exception as e:
                logger.error(f"[_align_to_target] 对准旋转失败: {e}")
                break
            alignments += 1
            time.sleep(settle_s)
            result = self.search(target, max_retries=0)
            extra_views += 1
            logger.info(f"[_align_to_target] 对准后第 {extra_views} 帧结果: {result}")
            if not result.get("found"):
                logger.warning(f"[_align_to_target] 对准后丢失目标，保留上次结果")
                result = current_result
                break
            current_result = result
        return result, extra_views, alignments

    def _call_vlm(self, image_path: str, target: str) -> Optional[dict]:
        """调用统一视觉分析器搜索目标。"""
        prompt = f'''你是机器人视觉定位助手。判断图片中是否存在目标物体："{target}"。

匹配放宽：允许近义词/同类物体、部分遮挡、侧放或倒放、处于远处的小目标——只要能合理相信它就是该物体即可判定存在；画面中有多个时，选最可能、最完整的一个。是否存在只取决于能否看到该物体，不要因为估不出高度或姿态而判定为不存在。

仅输出以下 JSON，不要 markdown 代码块、不要解释、不要多余文字：
{{
  "found": true/false,
  "bbox": [x1, y1, x2, y2],
  "pose": "upright" | "fallen" | "unknown",
  "height_cm": 估计高度厘米数（可选，估不出填 null）,
  "scene_description": "对整张画面内容的简短描述，例如'办公室场景，中央椅子上有蓝色纸巾盒'",
  "reason": "简短说明"
}}

bbox 用图片左上角为原点的 0~1 归一化坐标，顺序为 [左, 上, 右, 下]。scene_description 描述整张画面，不要只描述目标。'''

        try:
            logger.info(f"[_call_vlm] 调用 VisionAnalyzer，provider={self.provider}，图片: {image_path}")
            text, used_provider = self._analyzer.analyze(
                image_paths=image_path,
                prompt=prompt,
                max_tokens=1024,
                timeout=30,
            )
            logger.info(f"[_call_vlm] provider={used_provider} 模型原始返回: {text}")
            parsed = self._parse_json(text)
            logger.info(f"[_call_vlm] 解析后结果: {parsed}")
            if parsed and "scene_description" not in parsed:
                parsed["scene_description"] = "（模型未返回画面描述）"
            return parsed
        except Exception as e:
            logger.error(f"VLM 调用失败: {e}")
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
