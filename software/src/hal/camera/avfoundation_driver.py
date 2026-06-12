"""macOS AVFoundation native camera driver.

Uses PyObjC to call AVFoundation directly, bypassing OpenCV's volatile
integer device indices. Cameras are discovered and opened by their stable
``uniqueID``; only the resulting numpy frames are passed to OpenCV for
processing.

Design notes for the three well-known pitfalls:
1. Memory leaks: every callback wraps work in NSAutoreleasePool; the
   CVPixelBuffer is locked only for the minimum time; the numpy frame is
   ``.copy()``-ed before unlock so the Objective-C buffer can be released.
2. Latency buildup: the frame queue uses ``maxsize=1``. New frames evict
   old ones instead of blocking or piling up history.
3. Threading/GIL: AVFoundation callbacks arrive on a GCD queue. The
   callback performs the pixel conversion while the buffer is locked, then
   pushes the resulting numpy array into a thread-safe ``queue.Queue``.
   The Python consumer reads from the same queue on the main thread.
"""

from __future__ import annotations

import ctypes
import platform
import queue
import sys
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2

from common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Platform guard: this module only makes sense on macOS.
# ---------------------------------------------------------------------------
if platform.system() != "Darwin":
    raise ImportError("avfoundation_driver is only supported on macOS")

# PyObjC imports
from Foundation import NSObject, NSAutoreleasePool  # type: ignore
from AVFoundation import (  # type: ignore
    AVCaptureSession,
    AVCaptureDevice,
    AVCaptureDeviceInput,
    AVCaptureVideoDataOutput,
    AVCaptureDeviceDiscoverySession,
    AVMediaTypeVideo,
    AVCaptureDeviceTypeBuiltInWideAngleCamera,
    AVCaptureDeviceTypeExternal,
    AVCaptureSessionPreset1280x720,
    AVCaptureSessionPreset1920x1080,
    AVCaptureSessionPreset640x480,
    AVCaptureSessionPresetLow,
    AVCaptureSessionPresetMedium,
    AVCaptureSessionPresetHigh,
)
from CoreMedia import CMSampleBufferGetImageBuffer  # type: ignore
from Quartz.CoreVideo import (  # type: ignore
    CVPixelBufferLockBaseAddress,
    CVPixelBufferUnlockBaseAddress,
    CVPixelBufferGetBaseAddress,
    CVPixelBufferGetWidth,
    CVPixelBufferGetHeight,
    CVPixelBufferGetBytesPerRow,
)
import objc  # type: ignore


# kCVPixelFormatType_32BGRA: BGRA is the only pixel format both natively
# supported by AVFoundation and zero-copy friendly for OpenCV conversion.
_KCVPIXELFORMATTYPE_32BGRA: int = 1111970369


def _get_global_dispatch_queue():
    """Return a bridged ``dispatch_queue_t`` suitable for AVCapture callbacks."""
    lib = ctypes.CDLL("/usr/lib/system/libdispatch.dylib")
    lib.dispatch_get_global_queue.restype = ctypes.c_void_p
    lib.dispatch_get_global_queue.argtypes = [ctypes.c_long, ctypes.c_void_p]
    # DISPATCH_QUEUE_PRIORITY_DEFAULT == 0
    ptr = lib.dispatch_get_global_queue(0, None)
    return objc.objc_object(c_void_p=ptr)


class _AVCaptureDelegate(NSObject):
    """Objective-C delegate receiving AVCaptureVideoDataOutput callbacks."""

    def initWithQueue_flipHorizontal_(self, frame_queue, flip_horizontal: bool):
        self = self.init()
        self.frame_queue: "queue.Queue[np.ndarray]" = frame_queue
        self.flip_horizontal = flip_horizontal
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(
        self, output, sample_buffer, connection
    ) -> None:
        """Callback invoked on the GCD queue for each captured frame."""
        pool = NSAutoreleasePool.alloc().init()
        try:
            image_buffer = CMSampleBufferGetImageBuffer(sample_buffer)
            if image_buffer is None:
                return

            CVPixelBufferLockBaseAddress(image_buffer, 0)
            try:
                width = CVPixelBufferGetWidth(image_buffer)
                height = CVPixelBufferGetHeight(image_buffer)
                bytes_per_row = CVPixelBufferGetBytesPerRow(image_buffer)
                base_address = CVPixelBufferGetBaseAddress(image_buffer)

                # as_buffer(n) returns a memoryview over the first n elements.
                # The element size is a byte here, so n == bytes to read.
                buf = base_address.as_buffer(bytes_per_row * height)
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                    (height, bytes_per_row // 4, 4)
                )
                # Crop to visible width, convert BGRA -> BGR, then deep-copy
                # before unlocking so the OS can reuse the pixel buffer.
                bgr = cv2.cvtColor(arr[:, :width, :], cv2.COLOR_BGRA2BGR)
                if self.flip_horizontal:
                    bgr = cv2.flip(bgr, 1)
                frame = bgr.copy()
            finally:
                CVPixelBufferUnlockBaseAddress(image_buffer, 0)

            # Drop-oldest semantics: a robot must never consume stale frames.
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Should be unreachable because of the eviction above, but
                # guard against races just in case.
                pass
        finally:
            del pool


def _format_dimensions(format_) -> Tuple[int, int]:
    """Extract pixel width/height from an AVCaptureDeviceFormat."""
    # format_.formatDescription() -> CMFormatDescription
    # CMVideoFormatDescriptionGetDimensions() returns a struct {int32 width, int32 height}
    desc = format_.formatDescription()
    # PyObjC wraps the return as an NSValue-like structure; use the
    # ``dimensions`` property if present, otherwise fall back to the
    # CoreMedia function exposed via the CoreMedia module.
    try:
        dims = desc.dimensions()
        return int(dims.width), int(dims.height)
    except Exception:
        pass
    # Fallback: read from the CMVideoDimensions struct manually.
    import CoreMedia as CM  # type: ignore
    dims = CM.CMVideoFormatDescriptionGetDimensions(desc)
    return int(dims.width), int(dims.height)


def _choose_best_format(device, target_width: int, target_height: int):
    """Pick the AVCaptureDeviceFormat whose resolution is closest to target.

    If ``target_width`` or ``target_height`` is zero, the device's current
    active format is returned unchanged.
    """
    if target_width <= 0 or target_height <= 0:
        return device.activeFormat()

    best_format = device.activeFormat()
    best_w, best_h = _format_dimensions(best_format)
    best_cost = abs(best_w - target_width) + abs(best_h - target_height)

    for fmt in device.formats():
        w, h = _format_dimensions(fmt)
        cost = abs(w - target_width) + abs(h - target_height)
        if cost < best_cost:
            best_format = fmt
            best_cost = cost
            best_w, best_h = w, h

    logger.info(
        f"[AVFCamera] selected format {best_w}x{best_h} "
        f"(requested {target_width}x{target_height})"
    )
    return best_format


def list_avfoundation_devices() -> List[Dict[str, str]]:
    """Return all video capture devices visible to AVFoundation.

    Each entry contains ``name`` (localized display name) and ``uniqueID``
    (the stable hardware identifier that survives reboots and re-plugs).
    """
    device_types = [
        AVCaptureDeviceTypeBuiltInWideAngleCamera,
        AVCaptureDeviceTypeExternal,
    ]
    discovery = AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        device_types, AVMediaTypeVideo, 0
    )
    devices: List[Dict[str, str]] = []
    for dev in discovery.devices():
        devices.append(
            {
                "name": str(dev.localizedName()),
                "uniqueID": str(dev.uniqueID()),
            }
        )
    return devices


def find_avfoundation_device(name_substring: str) -> Optional[AVCaptureDevice]:
    """Find an AVCaptureDevice whose localized name contains ``name_substring``."""
    target = name_substring.lower()
    for info in list_avfoundation_devices():
        if target in info["name"].lower():
            return AVCaptureDevice.deviceWithUniqueID_(info["uniqueID"])
    return None


class AVFoundationCameraDriver:
    """Camera driver implemented directly on top of macOS AVFoundation.

    The public API mimics the existing ``CameraDriver`` class so it can be
    used as a drop-in replacement in ``VisionService``.
    """

    def __init__(
        self,
        device_name: str = "",
        unique_id: str = "",
        width: int = 0,
        height: int = 0,
        fps: int = 30,
        flip_horizontal: bool = True,
    ):
        """Open the camera identified by ``device_name`` or ``unique_id``.

        Args:
            device_name: Substring of the camera's localized name. On your
                system this is e.g. ``"USB摄像头"`` or ``"1080P USB Camera"``.
            unique_id: Hardware-level stable identifier (e.g. ``"0x11200000bdc8088"``).
                Takes precedence over ``device_name`` when both are provided.
            width: Requested capture width (0 = use camera default).
            height: Requested capture height (0 = use camera default).
            fps: Requested capture rate (best-effort; see ``_apply_fps``).
            flip_horizontal: Whether to mirror the frame horizontally.
        """
        if not device_name and not unique_id:
            raise ValueError("Either device_name or unique_id must be provided for AVFoundation driver")

        self._device_name = device_name
        self._target_unique_id = unique_id
        self._target_width = int(width)
        self._target_height = int(height)
        self._fps = int(fps)
        self._flip_horizontal = bool(flip_horizontal)

        self._frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._session: Optional[AVCaptureSession] = None
        self._output: Optional[AVCaptureVideoDataOutput] = None
        self._delegate: Optional[_AVCaptureDelegate] = None
        self._unique_id: Optional[str] = None
        self._actual_width = 0
        self._actual_height = 0

        self._start_session()
        logger.info(
            f"[AVFCamera] opened '{self._device_name or self._target_unique_id}' (uid={self._unique_id}) "
            f"at {self._actual_width}x{self._actual_height}, flip={self._flip_horizontal}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _start_session(self) -> None:
        device = None
        # 1. 优先通过 unique_id 精确匹配
        if self._target_unique_id:
            device = AVCaptureDevice.deviceWithUniqueID_(self._target_unique_id)
            if device is None:
                logger.warning(
                    f"[AVFCamera] unique_id '{self._target_unique_id}' not found, "
                    f"falling back to device_name matching"
                )
        # 2. 通过 device_name 子串匹配
        if device is None and self._device_name:
            device = find_avfoundation_device(self._device_name)
        if device is None:
            raise RuntimeError(
                f"AVFoundation camera not found: "
                f"unique_id='{self._target_unique_id}', device_name='{self._device_name}'"
            )
        self._unique_id = str(device.uniqueID())

        # Lock the device for configuration so we can set resolution & FPS.
        ok, err = device.lockForConfiguration_(None)
        if not ok:
            logger.warning(f"[AVFCamera] failed to lock device for configuration: {err}")

        try:
            fmt = _choose_best_format(device, self._target_width, self._target_height)
            device.setActiveFormat_(fmt)
            self._actual_width, self._actual_height = _format_dimensions(fmt)
            if self._fps > 0:
                self._apply_fps(device, self._fps)
        finally:
            device.unlockForConfiguration()

        session = AVCaptureSession.alloc().init()
        inp, err = AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if err or inp is None:
            raise RuntimeError(
                f"[AVFCamera] unable to create capture input: {err}"
            )
        if not session.canAddInput_(inp):
            raise RuntimeError("[AVFCamera] session cannot add capture input")
        session.addInput_(inp)

        output = AVCaptureVideoDataOutput.alloc().init()
        output.setAlwaysDiscardsLateVideoFrames_(True)
        output.setVideoSettings_({"PixelFormatType": _KCVPIXELFORMATTYPE_32BGRA})

        delegate = _AVCaptureDelegate.alloc().initWithQueue_flipHorizontal_(
            self._frame_queue, self._flip_horizontal
        )
        output.setSampleBufferDelegate_queue_(delegate, _get_global_dispatch_queue())

        if not session.canAddOutput_(output):
            raise RuntimeError("[AVFCamera] session cannot add video data output")
        session.addOutput_(output)

        session.startRunning()

        self._session = session
        self._output = output
        self._delegate = delegate

    def _apply_fps(self, device, fps: int) -> None:
        """Best-effort frame-rate configuration using supported ranges."""
        try:
            fmt = device.activeFormat()
            ranges = fmt.videoSupportedFrameRateRanges()
            if not ranges:
                return

            best_range = None
            best_cost = float('inf')
            for rng in ranges:
                lo = rng.minFrameRate()
                hi = rng.maxFrameRate()
                if lo <= fps <= hi:
                    best_range = rng
                    break
                cost = min(abs(lo - fps), abs(hi - fps))
                if cost < best_cost:
                    best_cost = cost
                    best_range = rng

            if best_range is None:
                return

            device.setActiveVideoMinFrameDuration_(best_range.minFrameDuration())
            device.setActiveVideoMaxFrameDuration_(best_range.maxFrameDuration())
            actual_fps = best_range.maxFrameRate()
            logger.info(f"[AVFCamera] set FPS to {actual_fps:.2f} (requested {fps})")
        except Exception as e:
            logger.warning(f"[AVFCamera] could not set FPS to {fps}: {e}")

    def _stop_session(self) -> None:
        if self._output:
            # Detach the delegate first so callbacks stop immediately.
            self._output.setSampleBufferDelegate_queue_(None, None)
            self._output = None
        if self._session:
            self._session.stopRunning()
            self._session = None
        self._delegate = None
        # Drain the frame queue so stale frames do not survive a reopen.
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Public API (mirrors CameraDriver)
    # ------------------------------------------------------------------
    def is_opened(self) -> bool:
        return self._session is not None and self._session.isRunning()

    def capture_frame(self, retries: int = 3, timeout: float = 1.0):
        """Return the latest frame as a BGR numpy array, or None on failure."""
        for attempt in range(retries):
            try:
                frame = self._frame_queue.get(timeout=timeout)
                logger.debug("[AVFCamera] captured frame")
                return frame
            except queue.Empty:
                if attempt < retries - 1:
                    logger.warning(
                        f"[AVFCamera] frame queue empty (attempt {attempt + 1}/{retries}), retrying..."
                    )
                    time.sleep(0.05)
                else:
                    logger.warning("[AVFCamera] failed to read frame after retries")
        return None

    # Alias for callers expecting cv2.VideoCapture-like semantics.
    read = capture_frame

    def reopen(self) -> bool:
        """Restart the capture session (used after consecutive failures)."""
        logger.info("[AVFCamera] reopening session")
        self._stop_session()
        try:
            self._start_session()
            logger.info("[AVFCamera] session restarted")
            return True
        except Exception as e:
            logger.error(f"[AVFCamera] failed to reopen session: {e}")
            return False

    def release(self) -> None:
        """Release all AVFoundation resources."""
        self._stop_session()
        logger.info("[AVFCamera] camera released")

    # Convenience accessors used by diagnostics / logging.
    @property
    def unique_id(self) -> Optional[str]:
        return self._unique_id

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._actual_width, self._actual_height
