"""Camera geometry utilities for 3D-aware grasping.

Provides:
  - CameraCalibration dataclass
  - Loading/saving calibration JSON files
  - Pixel-to-ray projection using intrinsics and distortion

All coordinates follow OpenCV convention:
  - camera frame: z forward, x right, y down
  - image frame: origin top-left, x right, y down

The module has no runtime dependencies beyond numpy and cv2.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_CV2_AVAILABLE = False
try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsic + extrinsic calibration for a single camera.

    Args:
        fx, fy: focal lengths in pixels
        cx, cy: principal point in pixels
        dist_coeffs: distortion coefficients (k1, k2, p1, p2, k3) or empty
        width, height: image resolution used during calibration
        T_camera_frame: optional 4x4 homogeneous transform from camera to a
            reference frame (e.g., gripper or robot base). None until extrinsic
            calibration is performed.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray
    width: int
    height: int
    T_camera_frame: Optional[np.ndarray] = None

    @property
    def camera_matrix(self) -> np.ndarray:
        """3x3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def has_intrinsics(self) -> bool:
        return self.fx > 0 and self.fy > 0 and self.width > 0 and self.height > 0

    def has_extrinsics(self) -> bool:
        return self.T_camera_frame is not None


def _numpy_to_list(obj):
    """Recursively convert numpy arrays to Python lists for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_list(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    return obj


def save_calibration(
    calib: CameraCalibration,
    camera_name: str,
    calibration_dir: str,
) -> str:
    """Save a CameraCalibration to JSON and return the file path."""
    os.makedirs(calibration_dir, exist_ok=True)
    path = os.path.join(calibration_dir, f"{camera_name}_camera_calibration.json")
    data = {
        "camera_name": camera_name,
        "fx": calib.fx,
        "fy": calib.fy,
        "cx": calib.cx,
        "cy": calib.cy,
        "dist_coeffs": _numpy_to_list(calib.dist_coeffs),
        "width": calib.width,
        "height": calib.height,
        "T_camera_frame": _numpy_to_list(calib.T_camera_frame),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_calibration(
    camera_name: str,
    calibration_dir: str,
) -> Optional[CameraCalibration]:
    """Load a CameraCalibration from JSON if it exists and is valid.

    Args:
        camera_name: e.g. "end" or "body"
        calibration_dir: directory containing `{camera_name}_camera_calibration.json`

    Returns:
        CameraCalibration or None if missing/invalid.
    """
    if not _CV2_AVAILABLE:
        return None

    path = os.path.join(calibration_dir, f"{camera_name}_camera_calibration.json")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dist_coeffs = np.array(data.get("dist_coeffs", []), dtype=np.float64)
        if dist_coeffs.size == 0:
            dist_coeffs = np.zeros((0,), dtype=np.float64)

        T_camera_frame = data.get("T_camera_frame")
        if T_camera_frame is not None:
            T_camera_frame = np.array(T_camera_frame, dtype=np.float64)
            if T_camera_frame.shape != (4, 4):
                T_camera_frame = None

        calib = CameraCalibration(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            dist_coeffs=dist_coeffs,
            width=int(data["width"]),
            height=int(data["height"]),
            T_camera_frame=T_camera_frame,
        )
        return calib if calib.has_intrinsics() else None

    except Exception as e:
        print(f"[camera_geometry] 加载 {camera_name} 标定数据失败: {e}")
        return None


def undistort_point(
    px: float,
    py: float,
    calib: CameraCalibration,
) -> Tuple[float, float]:
    """Undistort a pixel and return normalized image coordinates (x', y').

    The normalized coordinates are defined such that:
      ray_direction_camera = (x', y', 1)  (before normalization)
    """
    if not _CV2_AVAILABLE:
        # Fallback: assume perfect pinhole with no distortion.
        return (px - calib.cx) / calib.fx, (py - calib.cy) / calib.fy

    src = np.array([[[px, py]]], dtype=np.float64)
    K = calib.camera_matrix.reshape(3, 3)
    D = calib.dist_coeffs.reshape(-1, 1) if calib.dist_coeffs.size > 0 else None

    # cv2.undistortPoints returns normalized coordinates with z=1.
    undistorted = cv2.undistortPoints(src, K, D, P=K)
    x_norm = float(undistorted[0, 0, 0] - calib.cx) / calib.fx
    y_norm = float(undistorted[0, 0, 1] - calib.cy) / calib.fy
    return x_norm, y_norm


def pixel_to_ray(
    px: float,
    py: float,
    calib: CameraCalibration,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a pixel to a ray (origin, direction) in the camera frame.

    Returns:
        origin: (0, 0, 0) in camera frame
        direction: unit 3-vector in camera frame
    """
    x_norm, y_norm = undistort_point(px, py, calib)
    direction = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    origin = np.zeros(3, dtype=np.float64)
    return origin, direction


def transform_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    T_source_to_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a ray from source frame to target frame.

    Args:
        origin: 3-vector in source frame
        direction: 3-vector in source frame (need not be unit)
        T_source_to_target: 4x4 homogeneous transform from source to target

    Returns:
        (origin_target, direction_target)
    """
    T = np.asarray(T_source_to_target, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]

    origin_target = R @ np.asarray(origin) + t
    direction_target = R @ np.asarray(direction)
    direction_target /= np.linalg.norm(direction_target)
    return origin_target, direction_target


def ray_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
) -> Optional[np.ndarray]:
    """Intersect a ray with a plane. Returns None if parallel."""
    d = np.asarray(direction, dtype=np.float64)
    n = np.asarray(plane_normal, dtype=np.float64)
    denom = np.dot(n, d)
    if abs(denom) < 1e-9:
        return None
    p0 = np.asarray(plane_point, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    t = np.dot(n, p0 - o) / denom
    return o + t * d


def body_camera_estimate_xy(
    px: float,
    py: float,
    body_calib: CameraCalibration,
    ground_z: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Back-project a body-camera pixel to the ground plane (z = ground_z).

    Returns:
        (x, y) in the reference frame stored in body_calib.T_camera_frame,
        or None if no extrinsics.
    """
    if not body_calib.has_extrinsics():
        return None

    origin_cam, direction_cam = pixel_to_ray(px, py, body_calib)
    origin_ref, direction_ref = transform_ray(
        origin_cam, direction_cam, body_calib.T_camera_frame
    )
    point = ray_plane_intersection(
        origin_ref, direction_ref, plane_normal=np.array([0, 0, 1]), plane_point=np.array([0, 0, ground_z])
    )
    if point is None:
        return None
    return float(point[0]), float(point[1])


def triangulate_point(
    ray1_origin: np.ndarray,
    ray1_dir: np.ndarray,
    ray2_origin: np.ndarray,
    ray2_dir: np.ndarray,
) -> Optional[np.ndarray]:
    """Triangulate a 3D point as the midpoint of the common perpendicular
    between two rays. Returns None if rays are nearly parallel.
    """
    p1 = np.asarray(ray1_origin, dtype=np.float64)
    d1 = np.asarray(ray1_dir, dtype=np.float64)
    p2 = np.asarray(ray2_origin, dtype=np.float64)
    d2 = np.asarray(ray2_dir, dtype=np.float64)

    n = np.cross(d1, d2)
    denom = np.dot(n, n)
    if denom < 1e-12:
        return None

    # Solve for closest points on the two lines.
    # p1 + t1*d1, p2 + t2*d2
    diff = p2 - p1
    t1 = np.dot(np.cross(diff, d2), n) / denom
    t2 = np.dot(np.cross(diff, d1), n) / denom

    closest1 = p1 + t1 * d1
    closest2 = p2 + t2 * d2
    return (closest1 + closest2) / 2.0
