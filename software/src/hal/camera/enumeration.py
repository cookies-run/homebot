"""摄像头设备枚举工具

提供按名称查找摄像头的功能，避免 macOS 上 device_id 随插拔顺序漂移的问题。

在 macOS 上，本模块使用原生的 AVFoundation API（通过 PyObjC）进行枚举，
返回的设备信息包含稳定的 ``uniqueID``（硬件级标识，插拔不变）。当
VisionService 配置 ``camera.device_name`` 或 ``camera.unique_id`` 时，会
直接通过 ``AVFoundationCameraDriver`` 按 ``uniqueID`` 打开摄像头，完全绕
过 OpenCV 的整数索引机制。

在 Linux/Windows 上仍回退到平台特定的传统枚举方式。
"""

import platform
import subprocess
import re
import json
from typing import List, Dict, Optional


def list_camera_devices() -> List[Dict]:
    """枚举系统中所有可用的摄像头设备。

    Returns:
        设备列表。macOS 上每个元素包含 ``name``、``uniqueID``；其他平台
        包含 ``index``（OpenCV 设备索引）和 ``name``（设备名称）。
    """
    system = platform.system()
    if system == 'Darwin':
        return _list_cameras_macos()
    elif system == 'Linux':
        return _list_cameras_linux()
    elif system == 'Windows':
        return _list_cameras_windows()
    return []


def _list_cameras_macos() -> List[Dict]:
    """macOS: 使用 AVFoundation DiscoverySession 枚举摄像头。

    返回的 ``uniqueID`` 是硬件级稳定标识，可用于 ``AVFoundationCameraDriver``
    直接打开指定摄像头。
    """
    try:
        from hal.camera.avfoundation_driver import list_avfoundation_devices
        return list_avfoundation_devices()
    except Exception:
        # 如果 AVFoundation 驱动因任何原因不可用，回退到 system_profiler。
        return _list_macos_via_system_profiler()


def _list_macos_via_system_profiler() -> List[Dict]:
    """通过 system_profiler 枚举摄像头（旧方式，仅作 fallback）。"""
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "[CameraEnum] AVFoundation 枚举不可用，回退到 system_profiler。"
        "此方式无法提供与 OpenCV 一致的稳定索引，建议检查 PyObjC 安装。"
    )
    try:
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType', '-json'],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout)

        devices = []
        idx = 0
        for item in data.get('SPCameraDataType', []):
            name = item.get('_name', '')
            if name:
                devices.append({'index': idx, 'name': name})
                idx += 1
        return devices
    except Exception:
        return []


def _list_cameras_linux() -> List[Dict]:
    """Linux: 通过 v4l2-ctl 枚举。"""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True, timeout=5
        )
        devices = []
        current_name = None
        for line in result.stdout.splitlines():
            if line.strip().endswith(':'):
                current_name = line.strip().rstrip(':')
            elif '/dev/video' in line:
                match = re.search(r'/dev/video(\d+)', line)
                if match and current_name:
                    devices.append({'index': int(match.group(1)), 'name': current_name})
        return devices
    except FileNotFoundError:
        return []


def _list_cameras_windows() -> List[Dict]:
    """Windows: OpenCV 不直接提供设备名，先按索引枚举。"""
    import cv2
    devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            devices.append({'index': i, 'name': f'Camera {i}'})
            cap.release()
    return devices


def find_camera_index(name_substring: str) -> Optional[int]:
    """根据摄像头名称子串查找对应的 OpenCV 设备索引（非 macOS 场景使用）。

    Args:
        name_substring: 名称子串，不区分大小写。例如 "USB"、"FaceTime"。

    Returns:
        匹配到的设备索引；如果未找到则返回 None。

    Note:
        在 macOS 上建议直接使用 ``AVFoundationCameraDriver``，按
        ``device_name`` 或 ``unique_id`` 匹配，而不是 OpenCV 整数索引。
    """
    devices = list_camera_devices()
    for dev in devices:
        if name_substring.lower() in dev['name'].lower():
            # 旧 API 兼容：macOS 的 AVFoundation 结果没有 index，返回 -1 表示
            # "应使用 AVFoundation 原生驱动而非 OpenCV"。
            return dev.get('index', -1)
    return None


def resolve_device_id(config) -> int:
    """根据配置解析最终的设备索引。

    优先级：
        1. 如果 config.camera.unique_id 非空，直接返回 -1（表示使用 AVFoundation）
        2. 如果 config.camera.device_name 非空，macOS 返回 -1；其他平台按名称查找
        3. 回退到 config.camera.device_id

    Args:
        config: 配置对象，需包含 camera 配置

    Returns:
        解析后的 OpenCV 设备索引；-1 表示应使用 AVFoundation 原生驱动。
    """
    device_name = getattr(config, 'camera', None)
    if device_name is not None:
        device_name = getattr(config.camera, 'device_name', '') if hasattr(config, 'camera') else ''

    unique_id = ''
    if hasattr(config, 'camera'):
        unique_id = getattr(config.camera, 'unique_id', '')

    import platform
    if platform.system() == 'Darwin' and (unique_id or device_name):
        # macOS 原生路径：打印匹配到的摄像头信息供调试，返回 -1 让上层使用
        # AVFoundationCameraDriver。
        try:
            from hal.camera.avfoundation_driver import list_avfoundation_devices
            devices = list_avfoundation_devices()
            target = (unique_id or device_name).lower()
            for dev in devices:
                if unique_id and dev['uniqueID'].lower() == target:
                    import logging
                    logging.getLogger(__name__).info(
                        f"[CameraEnum] resolved by unique_id: {dev['name']} ({dev['uniqueID']})"
                    )
                    return -1
                if device_name and device_name.lower() in dev['name'].lower():
                    import logging
                    logging.getLogger(__name__).info(
                        f"[CameraEnum] resolved by name: {dev['name']} ({dev['uniqueID']})"
                    )
                    return -1
        except Exception:
            pass
        # 没找到也返回 -1，让 AVFoundationCameraDriver 自己抛出更具体的错误。
        import logging
        logging.getLogger(__name__).warning(
            f"Camera name/unique_id '{unique_id or device_name}' not found in AVFoundation"
        )
        return -1

    if device_name:
        idx = find_camera_index(device_name)
        if idx is not None:
            return idx
        import logging
        logging.getLogger(__name__).warning(
            f"Camera name '{device_name}' not found, falling back to device_id"
        )

    return getattr(config.camera, 'device_id', 0) if hasattr(config, 'camera') else 0


def print_camera_list():
    """打印当前系统可用的摄像头列表（命令行调试用）。"""
    devices = list_camera_devices()
    if not devices:
        print("未检测到摄像头，或当前平台不支持枚举")
        return
    print("可用摄像头列表：")
    for dev in devices:
        if 'uniqueID' in dev:
            print(f"  [name=\"{dev['name']}\"] uniqueID=\"{dev['uniqueID']}\"")
        else:
            print(f"  [{dev['index']}] {dev['name']}")


if __name__ == '__main__':
    print_camera_list()
