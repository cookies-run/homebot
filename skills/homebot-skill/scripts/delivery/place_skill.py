"""放置技能：将物体释放到目标位置。

作为外部可复用技能，支持独立运行或被 applications/delivery_agent 导入。
"""
import os
import sys
import time
from typing import Optional

# 支持独立运行：将 software/src 加入路径以导入 common/configs
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../software/src"))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from common.logging import get_logger

logger = get_logger(__name__)


class PlaceSkill:
    """放置技能。"""

    def __init__(self, arm_adapter, vision_adapter=None):
        self.arm = arm_adapter
        self.vision = vision_adapter

    def execute(self, destination_bbox: Optional[tuple] = None, confirm_hand: bool = False) -> dict:
        """执行放置。

        Args:
            destination_bbox: 递送目标归一化 bbox（可选）
            confirm_hand: 是否先确认手部/目标位置再释放

        Returns:
            {"success": bool, "message": str}
        """
        try:
            # 移动到释放姿态（高举、前进到合适位置）
            # 这里使用相对安全的预释放姿态
            release_pose = {
                "base": -90,
                "shoulder": 30,
                "elbow": 120,
                "wrist_flex": 30,
                "wrist_roll": 0,
                "gripper": 0,  # 夹爪保持闭合，到达后再打开
            }
            result = self.arm.set_joint_angles(release_pose, speed=600)
            if result.get("status") != "success":
                return {"success": False, "message": f"移动到释放姿态失败: {result.get('message')}"}

            time.sleep(1.0)

            if confirm_hand and self.vision:
                # TODO: 用末端摄像头 + VLM 确认手在下方
                logger.info("手部确认已启用，但当前使用简化逻辑")

            # 打开夹爪释放
            result = self.arm.open_gripper()
            if result.get("status") != "success":
                return {"success": False, "message": f"打开夹爪失败: {result.get('message')}"}

            time.sleep(0.5)

            # 稍微后退避免碰到物体
            self.arm.set_joint_angles({"gripper": 90}, speed=600)
            time.sleep(0.3)

            return {"success": True, "message": "物体已释放"}
        except Exception as e:
            logger.error(f"放置异常: {e}")
            return {"success": False, "message": f"放置异常: {e}"}

    def reset_arm(self) -> dict:
        """机械臂复位。"""
        try:
            from configs.config import get_config
            rest = get_config().arm.rest_position
            return self.arm.set_joint_angles(rest, speed=800)
        except Exception as e:
            logger.error(f"复位异常: {e}")
            return {"success": False, "message": str(e)}
