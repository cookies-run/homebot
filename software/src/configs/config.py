# -*- coding: utf-8 -*-
"""配置聚合 - 组装系统配置(system_config)与 AI 模型配置(ai_config)

硬件/系统类定义在 configs.system_config；AI 模型(含凭证)类定义在 configs.ai_config。
本模块只负责把它们聚合成全局 Config 并提供 get_config()/set_config()。
为兼容历史写法（from configs.config import XxxConfig），下面对迁出的类做了 re-export。
"""
from typing import Optional
from dataclasses import dataclass, field, asdict

import logging

from configs.system_config import (
    CameraConfig,
    ArmConfig,
    ChassisConfig,
    ZMQConfig,
    LoggingConfig,
    SpeechConfig,
    GamepadConfig,
    HumanFollowConfig,
    BatteryConfig,
)
from configs.ai_config import (
    TTSConfig,
    LLMConfig,
    VisionConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """全局配置"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    chassis: ChassisConfig = field(default_factory=ChassisConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    human_follow: HumanFollowConfig = field(default_factory=HumanFollowConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    gamepad: GamepadConfig = field(default_factory=GamepadConfig)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """从字典创建配置"""
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            arm=ArmConfig(**data.get("arm", {})),
            chassis=ChassisConfig(**data.get("chassis", {})),
            battery=BatteryConfig(**data.get("battery", {})),
            zmq=ZMQConfig(**data.get("zmq", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            human_follow=HumanFollowConfig(**data.get("human_follow", {})),
            speech=SpeechConfig(**data.get("speech", {})),
            tts=TTSConfig(**data.get("tts", {})),
            llm=LLMConfig(**data.get("llm", {})),
            vision=VisionConfig(**data.get("vision", {})),
            gamepad=GamepadConfig(**data.get("gamepad", {}))
        )


# 全局配置实例
_config_instance: Optional[Config] = None


def _resolve_auto_ports(config: Config) -> None:
    """解析串口配置中的 'auto' 值，调用自动检测"""
    from common.platform_utils import resolve_auto_port

    config.chassis.serial_port = resolve_auto_port(
        config.chassis.serial_port, "chassis.serial_port"
    )
    config.arm.serial_port = resolve_auto_port(
        config.arm.serial_port, "arm.serial_port"
    )


def get_config() -> Config:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        _resolve_auto_ports(_config_instance)
    return _config_instance


def set_config(config: Config):
    """设置全局配置实例"""
    global _config_instance
    _config_instance = config

