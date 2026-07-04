"""摄像头设备枚举工具

提供按名称/设备路径查找摄像头的功能，避免 Windows/macOS 上 device_id 随插拔顺序漂移的问题。

在 Windows 上，本模块优先使用 ``cv2-enumerate-cameras`` 枚举 DirectShow/MSMF 设备，返回
真实 friendly name、稳定 device path、VID/PID 以及对应的 OpenCV backend/index。缺失该
依赖时自动降级到传统 OpenCV 索引扫描。

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
from dataclasses import dataclass, field
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
    """Windows: 优先使用 cv2-enumerate-cameras 获取真实设备名与稳定路径。"""
    try:
        return _list_cameras_windows_cv2_enum()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"[CameraEnum] cv2-enumerate-cameras 不可用或枚举失败 ({e})，"
            "降级到 OpenCV 索引扫描。"
        )
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


def _list_cameras_windows_legacy() -> List[CameraDescriptor]:
    """Windows: OpenCV 不直接提供设备名，按索引枚举（fallback）。"""
    import cv2
    devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            devices.append(
                CameraDescriptor(
                    index=i,
                    name=f'Camera {i}',
                    backend=cv2.CAP_DSHOW,
                )
            )
            cap.release()
    return devices


def find_camera_descriptor(
    name: str = "",
    path: str = "",
    unique_id: str = "",
) -> Optional[CameraDescriptor]:
    """按名称子串、设备路径或唯一标识查找摄像头。

    Args:
        name: friendly name 子串，不区分大小写。
        path: Windows DirectShow/MSMF 稳定设备路径（精确匹配，不区分大小写）。
        unique_id: macOS AVFoundation uniqueID（精确匹配，不区分大小写）。

    Returns:
        匹配到的 ``CameraDescriptor``；未找到返回 None。
    """
    devices = list_camera_devices()

    target_path = (path or unique_id).lower()
    if target_path:
        for dev in devices:
            dev_path = (dev.path or dev.unique_id or '').lower()
            if dev_path and dev_path == target_path:
                return dev

    if name:
        matches = [dev for dev in devices if name.lower() in dev.name.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Camera name '{name}' 匹配到多个摄像头，请使用 device_path 区分：\n" +
                "\n".join(f"  [{d.index}] {d.name}  path={d.path}" for d in matches)
            )

    return None


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
    desc = find_camera_descriptor(name=name_substring)
    return desc.index if desc else None


def resolve_camera_descriptor(config) -> CameraDescriptor:
    """根据配置解析最终的摄像头描述符。

    匹配优先级：
        1. ``config.camera.device_path`` 精确匹配
        2. ``config.camera.unique_id`` 精确匹配
        3. ``config.camera.device_name`` 子串匹配（命中多个时抛错）
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
            extras.append(f"path=\"{dev.path}\"")
        if dev.vid is not None and dev.pid is not None:
            extras.append(f"VID/PID={dev.vid:04X}:{dev.pid:04X}")
        if dev.backend is not None:
            extras.append(f"backend={dev.backend}")
        if dev.unique_id:
            extras.append(f"uniqueID=\"{dev.unique_id}\"")
        extra_str = f"  {' '.join(extras)}" if extras else ""
        print(f"  [{dev.index}] \"{dev.name}\"{extra_str}")


if __name__ == '__main__':
    print_camera_list()
