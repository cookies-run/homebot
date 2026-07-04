"""Camera driver implementation using OpenCV."""

from typing import Optional
from common.logging import get_logger

logger = get_logger(__name__)


class CameraDriver:
    def __init__(
        self,
        device: int = 0,
        device_name: str = "",
        device_path: str = "",
        unique_id: str = "",
        backend: Optional[int] = None,
        flip_horizontal: bool = False,
        width: int = 0,
        height: int = 0,
    ):
        """Open the camera by index, name, or stable device path.

        Args:
            device: 摄像头设备索引（当 device_name/device_path/unique_id 都为空时使用）
            device_name: 摄像头名称子串（优先级低于 device_path/unique_id）
            device_path: 稳定设备路径（Windows DirectShow/MSMF；最优先）
            unique_id: macOS AVFoundation 稳定硬件标识
            backend: 显式指定 OpenCV backend；None 时按平台默认选择
            flip_horizontal: 是否水平翻转画面（默认 False，保持物理真实方向）
            width: 请求的分辨率宽度（0 表示使用摄像头默认）
            height: 请求的分辨率高度（0 表示使用摄像头默认）
        """
        self._device_name = device_name
        self._device_path = device_path
        self._unique_id = unique_id
        self._flip_horizontal = flip_horizontal
        self._width = width
        self._height = height
        self._fallback_device = device
        self._backend = backend

        desc = self._resolve_descriptor(device)
        self._device = desc.index
        if self._backend is None:
            self._backend = desc.backend

        self._cap = self._create_capture()
        if not self._cap or not self._cap.isOpened():
            logger.error(f"failed to open camera device {self._device}")
            raise RuntimeError(f"Camera {self._device} open failed")
        logger.info(
            f"camera {self._device} opened "
            f"(name='{device_name}', path='{device_path}', unique_id='{unique_id}', "
            f"backend={self._backend}, flip_horizontal={flip_horizontal}, resolution={width}x{height})"
        )

    def _resolve_descriptor(self, fallback_device: int):
        """根据 device_name/device_path/unique_id 解析为运行时描述符。"""
        if self._device_path or self._unique_id or self._device_name:
            try:
                from hal.camera.enumeration import find_camera_descriptor
                desc = find_camera_descriptor(
                    name=self._device_name,
                    path=self._device_path,
                    unique_id=self._unique_id,
                )
                if desc is not None:
                    logger.info(
                        f"CameraDriver resolved "
                        f"name='{self._device_name}' path='{self._device_path}' unique_id='{self._unique_id}' "
                        f"to OpenCV index {desc.index}"
                    )
                    return desc
                logger.warning(
                    f"CameraDriver name/path/unique_id not resolved "
                    f"(name='{self._device_name}', path='{self._device_path}', unique_id='{self._unique_id}'), "
                    f"falling back to device={fallback_device}"
                )
            except Exception as e:
                logger.warning(f"CameraDriver failed to resolve descriptor: {e}")

        import sys
        import cv2
        if self._backend is not None:
            backend = self._backend
        elif sys.platform == "win32":
            backend = cv2.CAP_DSHOW
        elif sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = cv2.CAP_V4L2

        from hal.camera.enumeration import CameraDescriptor
        return CameraDescriptor(index=fallback_device, name=f"Camera {fallback_device}", backend=backend)

    def _create_capture(self):
        """创建并配置 VideoCapture 实例."""
        import cv2
        import sys

        if self._backend is not None:
            backend = self._backend
        elif sys.platform == "win32":
            backend = cv2.CAP_DSHOW
        elif sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = cv2.CAP_V4L2

        cap = cv2.VideoCapture(self._device, backend)
        if cap.isOpened() and (self._width > 0 or self._height > 0):
            if self._width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            if self._height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"camera {self._device} resolution set to {actual_w}x{actual_h}")
        return cap

    def capture_frame(self, retries: int = 3):
        """Capture a single frame and return as numpy array (BGR).

        Args:
            retries: 读帧失败时重试次数（macOS AVFoundation 后端偶发需要重试）
        """
        import cv2
        import time
        if self._cap is None:
            raise RuntimeError("camera not initialized")

        for attempt in range(retries):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                # 水平翻转画面，解决 USB 摄像头镜像问题
                if self._flip_horizontal:
                    frame = cv2.flip(frame, 1)
                logger.debug("captured frame")
                return frame
            if attempt < retries - 1:
                logger.warning(f"failed to read frame (attempt {attempt + 1}/{retries}), retrying...")
                time.sleep(0.05)

        logger.warning("failed to read frame after retries")
        return None

    def reopen(self):
        """重新打开摄像头（用于连续丢帧后恢复，并重新解析设备描述符以应对索引漂移）"""
        logger.info(f"reopening camera {self._device}")

        # 配置了稳定标识时，重开前重新解析索引/backend
        if self._device_name or self._device_path or self._unique_id:
            try:
                desc = self._resolve_descriptor(self._fallback_device)
                if desc.index != self._device or desc.backend != self._backend:
                    logger.info(
                        f"camera descriptor changed: index {self._device}->{desc.index}, "
                        f"backend {self._backend}->{desc.backend}"
                    )
                    self._device = desc.index
                    self._backend = desc.backend
            except Exception as e:
                logger.warning(f"failed to re-resolve camera descriptor during reopen: {e}")

        if self._cap:
            self._cap.release()
        self._cap = self._create_capture()
        if self._cap and self._cap.isOpened():
            logger.info(f"camera {self._device} reopened")
            return True
        else:
            logger.error(f"failed to reopen camera {self._device}")
            return False

    def release(self):
        """Release the camera resource."""
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info("camera released")
