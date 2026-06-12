"""Camera driver implementation using OpenCV."""

from common.logging import get_logger

logger = get_logger(__name__)


class CameraDriver:
    def __init__(self, device: int = 0, device_name: str = "", flip_horizontal: bool = False, width: int = 0, height: int = 0):
        """Open the camera by index or name.

        Args:
            device: 摄像头设备索引（当 device_name 为空时使用）
            device_name: 摄像头名称子串（优先于 device），非空时按名称查找索引
            flip_horizontal: 是否水平翻转画面（默认 False，保持物理真实方向）
            width: 请求的分辨率宽度（0 表示使用摄像头默认）
            height: 请求的分辨率高度（0 表示使用摄像头默认）
        """
        import cv2
        import sys

        resolved_device = device
        if device_name:
            try:
                from hal.camera.enumeration import find_camera_index
                idx = find_camera_index(device_name)
                if idx is not None and idx >= 0:
                    resolved_device = idx
                    logger.info(f"CameraDriver resolved '{device_name}' to OpenCV index {idx}")
                else:
                    logger.warning(
                        f"CameraDriver name '{device_name}' not resolved, falling back to device={device}"
                    )
            except Exception as e:
                logger.warning(f"CameraDriver failed to resolve name '{device_name}': {e}")

        self._device = resolved_device
        self._flip_horizontal = flip_horizontal
        self._width = width
        self._height = height
        self._cap = self._create_capture()
        if not self._cap or not self._cap.isOpened():
            logger.error(f"failed to open camera device {resolved_device}")
            raise RuntimeError(f"Camera {resolved_device} open failed")
        logger.info(
            f"camera {resolved_device} opened "
            f"(flip_horizontal={flip_horizontal}, resolution={width}x{height})"
        )

    def _create_capture(self):
        """创建并配置 VideoCapture 实例."""
        import cv2
        import sys
        if sys.platform == "win32":
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
        """重新打开摄像头（用于连续丢帧后恢复）"""
        logger.info(f"reopening camera {self._device}")
        if self._cap:
            self._cap.release()
            self._cap = self._create_capture()
            if self._cap and self._cap.isOpened():
                logger.info(f"camera {self._device} reopened")
                return True
            else:
                logger.error(f"failed to reopen camera {self._device}")
                return False
        return False

    def release(self):
        """Release the camera resource."""
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info("camera released")
