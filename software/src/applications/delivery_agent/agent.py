"""递送智能体运行时（瘦身版）。

任务拆解（grab_target / deliver_target）已上移到调用侧的 LLM（picoclaw / 语音智能体），
Python 侧不再做语义解析。本类退化为 adapter + 技能原语的装配点，供后续按需接入。
细粒度技能通过 MCP 工具单独暴露（见 skills/homebot-skill/mcp_homebot_server.py）。
"""
import os
import sys
from typing import Optional

from common.logging import get_logger

from .config import DeliveryAgentConfig
from .state import SceneStateManager
from .adapter import ChassisAdapter, ArmAdapter, VisionAdapter

# 导入外部可复用技能（skills/homebot-skill/scripts/delivery/）
_delivery_skills_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../skills/homebot-skill/scripts/delivery")
)
if _delivery_skills_path not in sys.path:
    sys.path.insert(0, _delivery_skills_path)

from search_skill import SearchSkill
from approach_skill import ApproachSkill
from grasp_skill import GraspSkill
from place_skill import PlaceSkill

logger = get_logger(__name__)


class DeliveryAgent:
    """递送技能装配点。

    持有 adapter 与技能原语实例，不做语义解析、不做固定编排。
    """

    def __init__(self, config: Optional[DeliveryAgentConfig] = None):
        self.config = config or DeliveryAgentConfig.from_env()
        self.state = SceneStateManager(self.config.scene_state_persist_path)

        # 适配器与技能
        self.vision_adapter = VisionAdapter(self.config.vision_sub_addr)
        self.chassis_adapter = ChassisAdapter(self.config.chassis_service_addr)
        self.arm_adapter = ArmAdapter(self.config.arm_service_addr)

        self.search_skill = SearchSkill(self.vision_adapter)
        self.approach_skill = ApproachSkill(
            self.chassis_adapter,
            self.vision_adapter,
            approach_distance_cm=self.config.approach_distance_cm,
        )
        self.grasp_skill = GraspSkill()
        self.place_skill = PlaceSkill(self.arm_adapter, self.vision_adapter)

    def get_status(self) -> dict:
        """获取当前场景状态。"""
        state = self.state.get()
        return {
            "phase": state.phase.value,
            "grab_target": state.grab_target,
            "deliver_target": state.deliver_target,
            "held_object": state.held_object,
            "confirmed": state.confirmed,
            "precheck_passed": state.precheck_passed,
            "error_message": state.error_message,
        }

    def close(self):
        """释放资源。"""
        self.chassis_adapter.close()
        self.arm_adapter.close()
        self.vision_adapter.stop()
