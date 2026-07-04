"""摄像头设备枚举工具

提供按名称/设备路径查找摄像头的功能，避免 Windows/macOS 上 device_id 随插拔顺序漂移的问题。

在 Windows 上，本模块优先使用 ``cv2-enumerate-cameras`` 枚举 DirectShow/MSMF 设备，返回
真实 friendly name、稳定 device path、VID/PID 以及对应的 OpenCV backend/index。若该依赖
不可用，则降级到 PowerShell/CIM + OpenCV 探测方案获取真实设备名并映射到可读索引。最后
才回退到传统 OpenCV 索引扫描。

在 macOS 上，本模块使用原生的 AVFoundation API（通过 PyObjC）进行枚举，返回的设备信息
包含稳定的 ``uniqueID``（硬件级标识，插拔不变）。当 VisionService 配置 ``camera.device_name``
或 ``camera.unique_id`` 时，会直接通过 ``AVFoundationCameraDriver`` 按 ``uniqueID`` 打开摄像
头，完全绕过 OpenCV 的整数索引机制。

在 Linux 上仍回退到 ``v4l2-ctl --list-devices`` 枚举。
"""

import platform
import subprocess
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class CameraDescriptor:
    """统一描述一个摄像头设备。

    Attributes:
        index: 当前会话可传给 ``cv2.VideoCapture`` 的索引。
        name: 设备 friendly name。
        backend: OpenCV backend 常量（如 ``cv2.CAP_DSHOW``），可为 None。
        path: Windows DirectShow/MSMF 稳定设备路径；Linux 下可为 ``/dev/videoN``；
              macOS 下与 ``unique_id`` 含义相同。
        unique_id: macOS AVFoundation 稳定硬件标识。
        vid: USB Vendor ID（若可获取）。
        pid: USB Product ID（若可获取）。
    """
    index: int
    name: str
    backend: Optional[int] = None
    path: Optional[str] = None
    unique_id: Optional[str] = None
    vid: Optional[int] = None
    pid: Optional[int] = None

    def __getitem__(self, key: str):
        """兼容旧代码的字典访问方式。"""
        if key == 'index':
            return self.index
        if key == 'name':
            return self.name
        if key == 'uniqueID':
            return self.unique_id
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """兼容旧代码的 ``'uniqueID' in dev`` 判断。"""
        return key in ('index', 'name', 'uniqueID')


def list_camera_devices() -> List[CameraDescriptor]:
    """枚举系统中所有可用的摄像头设备。

    Returns:
        设备列表，元素为 ``CameraDescriptor``。Windows/macOS 下包含稳定的
        ``path`` / ``unique_id``；Linux 下包含 ``index``、``name`` 和 ``path``。
    """
    system = platform.system()
    if system == 'Darwin':
        return _list_cameras_macos()
    elif system == 'Linux':
        return _list_cameras_linux()
    elif system == 'Windows':
        return _list_cameras_windows()
    return []


def _list_cameras_macos() -> List[CameraDescriptor]:
    """macOS: 使用 AVFoundation DiscoverySession 枚举摄像头。

    返回的 ``unique_id`` 是硬件级稳定标识，可用于 ``AVFoundationCameraDriver``
    直接打开指定摄像头。
    """
    try:
        from hal.camera.avfoundation_driver import list_avfoundation_devices
        devices = list_avfoundation_devices()
        return [
            CameraDescriptor(
                index=-1,
                name=dev.get('name', ''),
                unique_id=dev.get('uniqueID', ''),
                path=dev.get('uniqueID', ''),
            )
            for dev in devices
        ]
    except Exception:
        return _list_macos_via_system_profiler()


def _list_macos_via_system_profiler() -> List[CameraDescriptor]:
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
                devices.append(CameraDescriptor(index=idx, name=name))
                idx += 1
        return devices
    except Exception:
        return []


def _list_cameras_linux() -> List[CameraDescriptor]:
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
                    index = int(match.group(1))
                    devices.append(
                        CameraDescriptor(
                            index=index,
                            name=current_name,
                            path=f'/dev/video{index}',
                        )
                    )
        return devices
    except FileNotFoundError:
        return []


def _list_cameras_windows() -> List[CameraDescriptor]:
    """Windows: 优先获取真实设备名与稳定路径，再映射到 OpenCV 索引。"""
    # 1) 优先 cv2-enumerate-cameras：可拿到 path / VID / PID / backend
    try:
        return _list_cameras_windows_cv2_enum()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"[CameraEnum] cv2-enumerate-cameras 不可用或枚举失败 ({e})，"
            "降级到 PowerShell/CIM + OpenCV 探测方案。"
        )

    # 2) 降级到 PowerShell/CIM 名称 + OpenCV 索引探测
    try:
        return _list_cameras_windows_cim()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"[CameraEnum] PowerShell/CIM 枚举失败 ({e})，降级到 OpenCV 索引扫描。"
        )

    # 3) 最后回退到纯 OpenCV 索引扫描
    return _list_cameras_windows_legacy()


def _list_cameras_windows_cv2_enum() -> List[CameraDescriptor]:
    """Windows: 使用 cv2-enumerate-cameras 枚举 DirectShow 摄像头。"""
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras

    devices = []
    for cam in enumerate_cameras(cv2.CAP_DSHOW):
        devices.append(
            CameraDescriptor(
                index=cam.index,
                name=cam.name,
                backend=cam.backend,
                path=getattr(cam, 'path', None),
                vid=getattr(cam, 'vid', None),
                pid=getattr(cam, 'pid', None),
            )
        )
    return devices


def _list_cameras_windows_cim() -> List[CameraDescriptor]:
    """Windows: 通过 PowerShell/CIM 获取真实名称，并映射到 OpenCV 可读索引。"""
    import cv2
    import logging

    logger = logging.getLogger(__name__)
    open_indices = _probe_windows_camera_indices()
    real_names = _list_windows_camera_names()

    if real_names:
        mapped = _map_windows_camera_names_to_indices(real_names, open_indices)
        if len(real_names) != len(open_indices):
            logger.warning(
                "[CameraEnum] Windows camera name count (%s) differs from readable OpenCV index count (%s); "
                "using role-aware external camera mapping: names=%s, indices=%s, mapped=%s",
                len(real_names), len(open_indices), real_names, open_indices, mapped,
            )
        return [
            CameraDescriptor(index=dev['index'], name=dev['name'], backend=cv2.CAP_DSHOW)
            for dev in mapped
        ]

    logger.warning("[CameraEnum] Windows real-name enumeration unavailable; falling back to Camera N labels")
    return [
        CameraDescriptor(index=i, name=f'Camera {i}', backend=cv2.CAP_DSHOW)
        for i in open_indices
    ]


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


def _match_score(name: str, target: str) -> int:
    """名称匹配强度评分，越高越优先。

    Returns:
        3: 精确匹配
        2: target 是 name 的子串
        1: name 是 target 的子串（弱匹配）
        0: 不匹配
    """
    norm_name = _normalize_camera_name(name)
    norm_target = _normalize_camera_name(target)
    if not norm_name or not norm_target:
        return 0
    if norm_name == norm_target:
        return 3
    if norm_target in norm_name:
        return 2
    if norm_name in norm_target:
        return 1
    return 0


def find_camera_descriptor(
    name: str = "",
    path: str = "",
    unique_id: str = "",
) -> Optional[CameraDescriptor]:
    """按名称、设备路径或唯一标识查找摄像头。

    名称匹配规则：
        - 优先精确匹配（忽略大小写/空白/中英文摄像头后缀）
        - 其次 ``target`` 是设备名子串
        - 最后设备名是 ``target`` 子串（弱匹配）

    Args:
        name: friendly name。
        path: Windows DirectShow/MSMF 稳定设备路径（精确匹配，不区分大小写）。
        unique_id: macOS AVFoundation uniqueID（精确匹配，不区分大小写）。

    Returns:
        匹配到的 ``CameraDescriptor``；未找到返回 None。
    """
    import logging
    logger = logging.getLogger(__name__)
    devices = list_camera_devices()

    # 1) path / unique_id 精确匹配（最稳定）
    target_path = (path or unique_id).lower()
    if target_path:
        for dev in devices:
            dev_path = (dev.path or dev.unique_id or '').lower()
            if dev_path and dev_path == target_path:
                return dev

    if not name:
        return None

    # 2) 名称匹配：按强度分组
    best_score = 0
    candidates = []
    for dev in devices:
        score = _match_score(dev.name, name)
        if score > best_score:
            best_score = score
            candidates = [dev]
        elif score == best_score and score > 0:
            candidates.append(dev)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # 多个同强度匹配：警告并返回第一个，避免静默错配
    logger.warning(
        "[CameraEnum] Multiple camera matches for '%s' (score=%s): %s; "
        "consider using device_path for deterministic matching.",
        name, best_score, [(d.index, d.name, d.path) for d in candidates]
    )
    return candidates[0]


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
    desc = find_camera_descriptor(name=name_substring)
    return desc.index if desc else None


def resolve_camera_descriptor(config) -> CameraDescriptor:
    """根据配置解析最终的摄像头描述符。

    匹配优先级：
        1. ``config.camera.device_path`` 精确匹配
        2. ``config.camera.unique_id`` 精确匹配
        3. ``config.camera.device_name`` 名称匹配
        4. 回退到 ``config.camera.device_id``

    Args:
        config: 配置对象，需包含 camera 配置。

    Returns:
        解析后的 ``CameraDescriptor``。macOS 上返回 ``index=-1`` 表示应使用
        ``AVFoundationCameraDriver``。
    """
    import logging
    logger = logging.getLogger(__name__)

    camera_cfg = getattr(config, 'camera', None) if hasattr(config, 'camera') else None
    device_name = getattr(camera_cfg, 'device_name', '') or '' if camera_cfg else ''
    unique_id = getattr(camera_cfg, 'unique_id', '') or '' if camera_cfg else ''
    device_path = getattr(camera_cfg, 'device_path', '') or '' if camera_cfg else ''
    device_id = getattr(camera_cfg, 'device_id', 0) if camera_cfg else 0

    if platform.system() == 'Darwin' and (device_path or unique_id or device_name):
        # macOS 原生路径：返回 index=-1，让上层使用 AVFoundationCameraDriver。
        return CameraDescriptor(
            index=-1,
            name=device_name,
            unique_id=unique_id or device_path,
            path=device_path or unique_id,
        )

    try:
        # 1) device_path
        if device_path:
            desc = find_camera_descriptor(path=device_path)
            if desc is not None:
                logger.info(f"[CameraEnum] resolved by device_path: {desc}")
                return desc
            logger.warning(f"[CameraEnum] device_path '{device_path}' not found")

        # 2) unique_id
        if unique_id:
            desc = find_camera_descriptor(unique_id=unique_id)
            if desc is not None:
                logger.info(f"[CameraEnum] resolved by unique_id: {desc}")
                return desc
            logger.warning(f"[CameraEnum] unique_id '{unique_id}' not found")

        # 3) device_name
        if device_name:
            desc = find_camera_descriptor(name=device_name)
            if desc is not None:
                logger.info(f"[CameraEnum] resolved by device_name: {desc}")
                return desc
            logger.warning(f"[CameraEnum] device_name '{device_name}' not found")
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"[CameraEnum] failed to resolve camera descriptor: {e}")

    # 4) fallback device_id
    backend = None
    import cv2
    if platform.system() == 'Windows':
        backend = cv2.CAP_DSHOW
    elif platform.system() == 'Linux':
        backend = cv2.CAP_V4L2
    elif platform.system() == 'Darwin':
        backend = cv2.CAP_AVFOUNDATION

    logger.info(f"[CameraEnum] fallback to device_id={device_id}")
    return CameraDescriptor(
        index=device_id,
        name=f"Camera {device_id}",
        backend=backend,
    )


def resolve_device_id(config) -> int:
    """根据配置解析最终的设备索引（兼容旧 API）。

    优先级与 ``resolve_camera_descriptor`` 一致。

    Args:
        config: 配置对象，需包含 camera 配置。

    Returns:
        解析后的 OpenCV 设备索引；-1 表示应使用 AVFoundation 原生驱动。
    """
    return resolve_camera_descriptor(config).index


def print_camera_list():
    """打印当前系统可用的摄像头列表（命令行调试用）。"""
    devices = list_camera_devices()
    if not devices:
        print("未检测到摄像头，或当前平台不支持枚举")
        return
    print("可用摄像头列表：")
    for dev in devices:
        extras = []
        if dev.path:
            extras.append(f'path="{dev.path}"')
        if dev.vid is not None and dev.pid is not None:
            extras.append(f"VID/PID={dev.vid:04X}:{dev.pid:04X}")
        if dev.backend is not None:
            extras.append(f"backend={dev.backend}")
        if dev.unique_id:
            extras.append(f'uniqueID="{dev.unique_id}"')
        extra_str = f"  {' '.join(extras)}" if extras else ""
        print(f"  [{dev.index}] \"{dev.name}\"{extra_str}")


if __name__ == '__main__':
    print_camera_list()
