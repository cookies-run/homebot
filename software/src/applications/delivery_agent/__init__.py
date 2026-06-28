"""HomeBot 递送任务智能体。

该包实现一个基于大模型手册（agents/delivery_manager.md）的递送任务管理智能体，
负责语义解析、用户确认、预检查、阶段编排和技能调用。所有实现均为新增文件，
不修改原有的 human_follow、grab_optimized、speech_interaction 等模块。
"""

from .agent import DeliveryAgent
from .state import SceneStateManager, DeliveryPhase, SceneState
from .config import DeliveryAgentConfig

__all__ = [
    "DeliveryAgent",
    "SceneStateManager",
    "DeliveryPhase",
    "SceneState",
    "DeliveryAgentConfig",
]
