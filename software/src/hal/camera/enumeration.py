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
    """Windows: 优先枚举真实设备名，再映射到 OpenCV DirectShow 索引。"""
    import logging

    logger = logging.getLogger(__name__)
    open_indices = _probe_windows_camera_indices()

    real_names = _list_windows_camera_names()
    if real_names:
        # Windows PnP 枚举已过滤掉笔记本内置/音频设备；OpenCV 可读索引中
        # 仍可能夹着内置摄像头。实际在 Windows 10 上观测到：外接机身摄像头
        # 在 DSHOW index 0，笔记本摄像头在 index 1，末端摄像头在 MSMF/ANY
        # index 2。因此不要简单尾部对齐，否则会把机身摄像头错配到笔记本。
        devices = _map_windows_camera_names_to_indices(real_names, open_indices)
        if len(real_names) != len(open_indices):
            logger.warning(
                "[CameraEnum] Windows camera name count (%s) differs from readable OpenCV index count (%s); "
                "using role-aware external camera mapping: names=%s, indices=%s, devices=%s",
                len(real_names), len(open_indices), real_names, open_indices, devices,
            )
        return devices

    logger.warning("[CameraEnum] Windows real-name enumeration unavailable; falling back to Camera N labels")
    return [{'index': i, 'name': f'Camera {i}'} for i in open_indices]


def _probe_windows_camera_indices() -> List[int]:
    """探测 Windows 上可读取画面的 OpenCV 摄像头索引。"""
    import cv2

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    indices = []
    for i in range(10):
        for backend in backends:
            cap = cv2.VideoCapture(i, backend)
            ok = cap.isOpened()
            ret = False
            if ok:
                ret, frame = cap.read()
                ret = ret and frame is not None
            cap.release()
            if ret:
                indices.append(i)
                break
    return indices


def _map_windows_camera_names_to_indices(names: List[str], indices: List[int]) -> List[Dict]:
    """按 HomeBot 摄像头角色把 Windows 真实名称映射到 OpenCV 索引。"""
    if not names:
        return []

    # 当前 HomeBot Windows 设备形态：过滤内置/音频设备后，真实名称通常只剩
    # 机身 1080P USB Camera 和末端 USB Camera；OpenCV 可读索引仍可能包含
    # 笔记本摄像头。实测 /run 启动时 index 0 会落到笔记本，末端在最后一路，
    # 因此 3 路可读时跳过 index 0：机身取中间一路，末端取最后一路。
    if len(names) == 2 and len(indices) >= 3:
        return [
            {'index': indices[1], 'name': names[0]},
            {'index': indices[-1], 'name': names[1]},
        ]

    if len(indices) >= len(names):
        selected_indices = indices[:len(names)]
    else:
        selected_indices = indices + list(range(len(indices), len(names)))

    return [
        {'index': selected_indices[pos], 'name': name}
        for pos, name in enumerate(names)
    ]


def _list_windows_camera_names() -> List[str]:
    """通过 PowerShell/CIM 获取 Windows 摄像头真实名称。"""
    script = r"""
$devices = Get-CimInstance Win32_PnPEntity |
    Where-Object {
        $_.Name -and
        $_.Name -notmatch 'audio|microphone|麦克风|音频|integrated|built-in|builtin|内置|facetime' -and
        (
            $_.PNPClass -eq 'Camera' -or
            $_.PNPClass -eq 'Image' -or
            (
                $_.PNPClass -notin @('AudioEndpoint', 'MEDIA') -and
                $_.Name -match 'camera|摄像头|webcam|usb video|usb2\.0|1080p'
            )
        )
    } |
    Select-Object -ExpandProperty Name
$devices | ConvertTo-Json -Compress
""".strip()
    try:
        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command', script],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        data = json.loads(result.stdout)
        if isinstance(data, str):
            names = [data]
        else:
            names = [str(item) for item in data if item]
        return _dedupe_names(names)
    except Exception:
        return []


def _dedupe_names(names: List[str]) -> List[str]:
    """保留顺序去重。"""
    seen = set()
    result = []
    for name in names:
        normalized = _normalize_camera_name(name)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(name.strip())
    return result


def _normalize_camera_name(name: str) -> str:
    """统一摄像头名称，便于跨平台不区分大小写/空白/中英文后缀匹配。"""
    normalized = str(name or '').strip().casefold()
    normalized = normalized.replace('摄像头', 'camera')
    return re.sub(r'\s+', '', normalized)


def find_camera_index(name_substring: str) -> Optional[int]:
    """根据摄像头名称查找对应的 OpenCV 设备索引（非 macOS 场景使用）。

    Args:
        name_substring: 名称或子串，不区分大小写。例如 "USB"、"FaceTime"。

    Returns:
        匹配到的设备索引；如果未找到则返回 None。

    Note:
        在 macOS 上建议直接使用 ``AVFoundationCameraDriver``，按
        ``device_name`` 或 ``unique_id`` 匹配，而不是 OpenCV 整数索引。
    """
    import logging

    logger = logging.getLogger(__name__)
    devices = list_camera_devices()
    target = _normalize_camera_name(name_substring)
    if not target:
        return None

    exact_matches = [dev for dev in devices if _normalize_camera_name(dev.get('name', '')) == target]
    if exact_matches:
        if len(exact_matches) > 1:
            logger.warning("[CameraEnum] Multiple exact camera matches for '%s': %s", name_substring, exact_matches)
        return exact_matches[0].get('index', -1)

    substring_matches = [dev for dev in devices if target in _normalize_camera_name(dev.get('name', ''))]
    if substring_matches:
        if len(substring_matches) > 1:
            logger.warning("[CameraEnum] Multiple substring camera matches for '%s': %s", name_substring, substring_matches)
        return substring_matches[0].get('index', -1)

    reverse_matches = [dev for dev in devices if _normalize_camera_name(dev.get('name', '')) in target]
    if reverse_matches:
        logger.warning("[CameraEnum] Using weak reverse camera-name match for '%s': %s", name_substring, reverse_matches)
        return reverse_matches[0].get('index', -1)

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
        fallback_id = getattr(config.camera, 'device_id', 0) if hasattr(config, 'camera') else 0
        devices = list_camera_devices()
        if devices:
            device_lines = "\n".join(
                f"[{dev.get('index', 'name')}] {dev.get('name', '')}" for dev in devices
            )
            logging.getLogger(__name__).warning(
                "Camera name '%s' not found among cameras:\n%s\nfalling back to device_id=%s",
                device_name, device_lines, fallback_id,
            )
        else:
            logging.getLogger(__name__).warning(
                "Camera name '%s' not found; no cameras enumerated; falling back to device_id=%s",
                device_name, fallback_id,
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
