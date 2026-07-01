"""接近技能：基于视觉跟踪的 PID 接近控制。

作为外部可复用技能，支持独立运行或被 applications/delivery_agent 导入。
"""
import os
import sys
import time
from typing import Optional, Tuple

# 支持独立运行：将 software/src 加入路径以导入 common
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../software/src"))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import numpy as np

from common.logging import get_logger
from tracker_skill import TargetTracker, Detection

logger = get_logger(__name__)


class ApproachSkill:
    """接近技能。

    使用跟踪器锁定目标，通过底盘 PID 控制使机器人接近到目标指定距离内。
    """

    def __init__(self,
                 chassis_adapter,
                 vision_adapter,
                 approach_distance_cm: float = 30.0,
                 max_linear_speed: float = 0.2,
                 max_angular_speed: float = 0.5,
                 kp_linear: float = 0.8,
                 kp_angular: float = 1.5,
                 dead_zone_x: float = 0.15,
                 dead_zone_area: float = 0.1,
                 target_area_at_30cm: float = 0.08):
        self.chassis = chassis_adapter
        self.vision = vision_adapter
        self.approach_distance_cm = approach_distance_cm
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.dead_zone_x = dead_zone_x
        self.dead_zone_area = dead_zone_area
        self.target_area_at_30cm = target_area_at_30cm
        self.tracker = TargetTracker(selection_strategy="center")

    def approach(self,
                 target_bbox: Tuple[float, float, float, float],
                 timeout_s: float = 30.0,
                 update_callback=None,
                 relocate_fn=None,
                 relocate_interval_s: float = 1.5) -> dict:
        """接近目标。

        Args:
            target_bbox: 归一化 xyxy 初始位置
            timeout_s: 超时时间
            update_callback: 可选回调，接收 (tracker, frame) 用于可视化
            relocate_fn: 可选无参回调，返回最新的归一化 bbox（None 表示本次未找到）。
                提供时会按 relocate_interval_s 周期性重定位并更新跟踪器，使目标框
                随机器人前进而放大，从而让面积停止阈值真正生效。不提供则退化为
                使用初始 bbox 的开环行为（面积恒定，通常只能靠超时结束）。
            relocate_interval_s: 两次重定位之间的最小间隔秒数

        Returns:
            {"success": bool, "message": str, "final_distance_cm": float}
        """
        self.tracker.reset()
        # 用初始 bbox 初始化跟踪器
        self.tracker.update([Detection(bbox=target_bbox, confidence=0.9)])

        start = time.time()
        last_move_time = start
        last_relocate = start
        while time.time() - start < timeout_s:
            frame_id, frame = self.vision.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]

            # 周期性重定位：调用方注入 VLM/检测回调，返回最新 bbox。
            # 只有更新了 bbox，面积才会随接近而增大，area 停止阈值才有意义。
            now = time.time()
            if relocate_fn is not None and now - last_relocate >= relocate_interval_s:
                last_relocate = now
                try:
                    new_bbox = relocate_fn()
                except Exception as e:
                    logger.warning(f"接近过程中重定位异常: {e}")
                    new_bbox = None
                if new_bbox is not None:
                    self.tracker.update([Detection(bbox=tuple(new_bbox), confidence=0.9)])

            target = self.tracker.get_primary_target()
            if target is None:
                # 跟踪丢失，使用最后已知位置
                target = self.tracker.targets[0] if self.tracker.targets else None

            if target is None:
                # 完全丢失，停止
                self.chassis.stop()
                return {"success": False, "message": "接近过程中丢失目标"}

            if update_callback:
                update_callback(self.tracker, frame)

            cx, cy = target.center
            area = target.area

            error_x = cx - 0.5
            if abs(error_x) < self.dead_zone_x:
                error_x = 0.0

            # 目标面积越大说明越近
            error_area = (self.target_area_at_30cm - area) / self.target_area_at_30cm
            if abs(error_area) < self.dead_zone_area:
                error_area = 0.0

            vz = self.kp_angular * error_x
            vx = self.kp_linear * error_area

            # 限制速度
            vz = max(-self.max_angular_speed, min(self.max_angular_speed, vz))
            vx = max(0.0, min(self.max_linear_speed, vx))  # 只允许前进

            # 到达目标距离
            if area >= self.target_area_at_30cm and abs(error_x) <= self.dead_zone_x:
                self.chassis.stop()
                estimated_distance = self.approach_distance_cm * (self.target_area_at_30cm / max(area, 1e-6)) ** 0.5
                return {
                    "success": True,
                    "message": f"已接近目标到约 {estimated_distance:.1f}cm",
                    "final_distance_cm": estimated_distance,
                }

            # 发送速度指令，周期 100ms
            self.chassis.send_velocity(vx, 0, vz, duration_ms=100)
            last_move_time = time.time()
            time.sleep(0.05)

        self.chassis.stop()
        return {"success": False, "message": "接近目标超时"}

    def rotate_search(self, step_deg: float = 30.0) -> dict:
        """旋转搜索目标。"""
        return self.chassis.rotate_left_deg(step_deg)
