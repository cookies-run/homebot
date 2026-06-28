"""抓取技能：封装现有抓取工作流。

作为外部可复用技能，支持独立运行或被 applications/delivery_agent 导入。
与 grab_optimized.py 位于同一 skills/homebot-skill/scripts/ 目录下，可直接导入。
"""
import os
import sys
from typing import Optional

# 支持独立运行：将 software/src 加入路径以导入 common
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../software/src"))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from common.logging import get_logger

logger = get_logger(__name__)


class GraspSkill:
    """抓取技能。"""

    def __init__(self, robot_ip: Optional[str] = None):
        self.robot_ip = robot_ip
        self._workflow = None

    def _get_workflow(self):
        """惰性导入并创建 AutoGrabWorkflow 实例。"""
        if self._workflow is not None:
            return self._workflow

        # 将父目录（scripts）加入路径，确保能导入 grab_optimized/robot_config
        scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        try:
            from grab_optimized import AutoGrabWorkflow
            import robot_config as config
            self._workflow = AutoGrabWorkflow(
                robot_ip=self.robot_ip or config.ROBOT_IP,
                video_port=config.VIDEO_PORT,
                end_video_port=config.END_VIDEO_PORT,
                arm_port=config.ARM_PORT,
                use_end_camera=True,
            )
            return self._workflow
        except ImportError as e:
            logger.error(f"无法导入 AutoGrabWorkflow: {e}")
            return None

    def execute(self, target: str) -> dict:
        """执行抓取。

        Args:
            target: 目标描述

        Returns:
            {"success": bool, "message": str}
        """
        workflow = self._get_workflow()
        if workflow is None:
            return {"success": False, "message": "抓取工作流不可用"}

        try:
            result = workflow.run(target_object=target)
            return {
                "success": result.get("success", False),
                "message": result.get("message", ""),
            }
        except Exception as e:
            logger.error(f"抓取执行异常: {e}")
            return {"success": False, "message": f"抓取异常: {e}"}
