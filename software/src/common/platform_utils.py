# -*- coding: utf-8 -*-
"""跨平台工具函数

提供串口自动检测等跨平台功能，使项目能在 Windows / Linux / macOS 上无缝切换。
"""
import platform
import sys
from typing import Optional, List

from common.logging import get_logger

logger = get_logger(__name__)


# 常见 USB 转串口芯片 VID:PID 白名单（精确匹配，最高优先级）
# 格式: (vid, pid) → 芯片名称
KNOWN_SERVO_CHIP_VID_PID = {
    (0x1A86, 0x55D3): "CH347",      # WCH CH347 单串口
    (0x1A86, 0x7523): "CH340",      # WCH CH340
    (0x1A86, 0x7522): "CH341",      # WCH CH341
    (0x10C4, 0xEA60): "CP2102",     # Silicon Labs CP2102
    (0x10C4, 0xEA70): "CP2105",     # Silicon Labs CP2105
    (0x0403, 0x6001): "FT232",      # FTDI FT232
    (0x0403, 0x6010): "FT2232",     # FTDI FT2232
    (0x0403, 0x6011): "FT4232",     # FTDI FT4232
}

# 常见 USB 转串口芯片关键词（按优先级排序，VID/PID 匹配失败后的 fallback）
DEFAULT_PREFERRED_KEYWORDS = [
    "CH340",
    "CH341",
    "CH347",
    "CP2102",
    "CP210x",
    "FT232",
    "FT231",
    "FT2232",
    "USB-SERIAL",
    "USB Serial",
    "Arduino",
    "ttyACM",
    "ttyUSB",
    "usbserial",    # 优先于 usbmodem，避免误匹配摄像头
    "usbmodem",
]


def get_platform_name() -> str:
    """获取当前平台名称"""
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Darwin":
        return "macos"
    elif system == "Linux":
        return "linux"
    return system.lower()


def get_platform_default_serial_port() -> str:
    """获取当前平台的默认串口路径（仅用于提示/显示）"""
    plat = get_platform_name()
    if plat == "windows":
        return "COM3"
    elif plat == "macos":
        return "/dev/tty.usbserial"
    elif plat == "linux":
        return "/dev/ttyUSB0"
    return "/dev/ttyUSB0"


def auto_detect_serial_port(preferred_keywords: Optional[List[str]] = None) -> Optional[str]:
    """自动检测可用的串口设备

    使用 pyserial 的 list_ports 枚举所有串口，优先匹配常见的 USB 转串口芯片。
    底盘和机械臂通常共用一个串口（通过 USB 转串口模块连接）。

    Args:
        preferred_keywords: 优先匹配的串口描述关键词列表，默认使用常见芯片名称

    Returns:
        检测到的串口路径，如 "COM3"、"/dev/ttyUSB0"、"/dev/tty.usbmodemxxx"
        未检测到任何串口时返回 None
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        logger.error("未安装 pyserial，无法自动检测串口。请先安装: pip install pyserial")
        return None

    keywords = preferred_keywords or DEFAULT_PREFERRED_KEYWORDS
    ports = list(list_ports.comports())

    if not ports:
        logger.warning("未检测到任何串口设备，请检查硬件连接")
        return None

    # 记录所有可用串口（方便调试）
    logger.info(f"检测到 {len(ports)} 个串口设备:")
    for p in ports:
        logger.info(f"  - {p.device}: {p.description or 'N/A'} (hwid={p.hwid or 'N/A'})")

    # 阶段1: VID/PID 精确匹配（最高优先级）
    for p in ports:
        if p.vid is not None and p.pid is not None:
            key = (p.vid, p.pid)
            if key in KNOWN_SERVO_CHIP_VID_PID:
                chip_name = KNOWN_SERVO_CHIP_VID_PID[key]
                logger.info(f"自动选择串口: {p.device} (VID/PID 精确匹配: {chip_name})")
                return p.device

    # 阶段2: 按关键词优先级匹配
    search_texts = []
    for p in ports:
        text = f"{p.device} {p.description or ''} {p.hwid or ''}".lower()
        search_texts.append((p.device, text))

    for keyword in keywords:
        keyword_lower = keyword.lower()
        for device, text in search_texts:
            if keyword_lower in text:
                logger.info(f"自动选择串口: {device} (匹配关键词: {keyword})")
                return device

    # 阶段3: 兜底策略 - 返回第一个非蓝牙/内建的串口
    excluded_prefixes = ("bluetooth", "bt", "wlan", "无线", "wireless", "builtin")
    for device, text in search_texts:
        if not any(ex in text for ex in excluded_prefixes):
            logger.info(f"自动选择串口: {device} (未匹配到优先关键词，使用第一个可用串口)")
            return device

    # 阶段4: 实在没有合适的，返回第一个
    first = ports[0].device
    logger.info(f"自动选择串口: {first} (兜底)")
    return first


def resolve_auto_port(current_value: str, config_name: str = "serial_port") -> str:
    """解析 'auto' 串口配置值

    如果当前值为 'auto'，则调用自动检测；否则原样返回。

    Args:
        current_value: 配置中的串口值
        config_name: 配置项名称（用于日志）

    Returns:
        解析后的串口路径，或原值（如果不是 auto）
    """
    if current_value != "auto":
        return current_value

    logger.info(f"{config_name} = 'auto'，开始自动检测串口...")
    detected = auto_detect_serial_port()

    if detected:
        logger.info(f"{config_name} 自动解析为: {detected}")
        return detected

    # 检测失败时保持 'auto'，让下游代码报错并提示用户
    logger.warning(
        f"{config_name} = 'auto' 但未能自动检测到串口设备。"
        f"请运行 'python tools/list_devices.py' 查看可用设备，"
        f"或在配置中手动指定串口路径（如 {get_platform_default_serial_port()}）"
    )
    return "auto"
