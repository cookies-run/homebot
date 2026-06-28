"""递送智能体配置。"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeliveryAgentConfig:
    """递送智能体配置。

    该配置独立于全局 configs.config.Config，避免修改原始配置文件。
    关键参数也可通过环境变量覆盖。
    """

    # 智能体手册路径（相对项目根目录）
    handbook_path: str = "agents/delivery_manager.md"

    # 接近距离
    approach_distance_cm: float = 30.0

    # 搜索超时（秒）
    search_timeout_s: float = 15.0

    # 最大递送重试次数
    max_delivery_retries: int = 2

    # 场景状态持久化路径
    scene_state_persist_path: str = "/tmp/homebot_scene_state.json"

    # 目标切换迟滞：新目标评分必须优于当前目标的比例
    primary_target_switch_margin: float = 0.20

    # 目标切换迟滞：连续多少帧更优才切换
    primary_target_switch_min_frames: int = 3

    # 视觉服务订阅地址
    vision_sub_addr: str = "tcp://localhost:5560"

    # 底盘/机械臂服务地址
    chassis_service_addr: str = "tcp://localhost:5556"
    arm_service_addr: str = "tcp://localhost:5557"

    # LLM 配置（默认复用全局 LLM 配置）
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""

    # 调试显示
    display: bool = False

    def __post_init__(self):
        """支持通过环境变量覆盖关键配置。"""
        if env := os.environ.get("DELIVERY_HANDBOOK_PATH"):
            self.handbook_path = env
        if env := os.environ.get("DELIVERY_SCENE_STATE_PATH"):
            self.scene_state_persist_path = env
        if env := os.environ.get("DELIVERY_VISION_SUB_ADDR"):
            self.vision_sub_addr = env
        if env := os.environ.get("DELIVERY_CHASSIS_ADDR"):
            self.chassis_service_addr = env
        if env := os.environ.get("DELIVERY_ARM_ADDR"):
            self.arm_service_addr = env

    @classmethod
    def from_env(cls) -> "DeliveryAgentConfig":
        """从环境变量构建配置。"""
        return cls()

    def resolve_handbook_path(self, project_root: str) -> str:
        """解析手册绝对路径。"""
        path = Path(self.handbook_path)
        if path.is_absolute():
            return str(path)
        return str(Path(project_root) / path)
