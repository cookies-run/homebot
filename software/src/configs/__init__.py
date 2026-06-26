# -*- coding: utf-8 -*-
"""配置管理模块

使用方式:
    from configs import get_config, get_ai_credentials, check_ai_credentials
    
    # 获取配置
    config = get_config()
    print(config.chassis.serial_port)
    
    # 检查密钥配置状态
    check_ai_credentials()
"""
from configs.config import (
    get_config,
    set_config,
    Config,
    CameraConfig,
    ArmConfig,
    ChassisConfig,
    ZMQConfig,
    LoggingConfig,
    SpeechConfig,
    TTSConfig,
    LLMConfig,
    VisionConfig,
    GamepadConfig,
    HumanFollowConfig,
    BatteryConfig,
)

from configs.ai_config import (
    get_ai_credentials,
    reload_ai_credentials,
    check_ai_credentials,
    require_ai_credentials,
    AICredentials,
    TTSCredentials,
    LLMCredentials,
    VisionCredentials,
)

__all__ = [
    # 配置
    "get_config",
    "set_config",
    "Config",
    "CameraConfig",
    "ArmConfig",
    "ChassisConfig",
    "ZMQConfig",
    "LoggingConfig",
    "SpeechConfig",
    "TTSConfig",
    "LLMConfig",
    "VisionConfig",
    "GamepadConfig",
    "HumanFollowConfig",
    "BatteryConfig",
    # 密钥
    "get_ai_credentials",
    "reload_ai_credentials",
    "check_ai_credentials",
    "require_ai_credentials",
    "AICredentials",
    "TTSCredentials",
    "LLMCredentials",
    "VisionCredentials",
]
