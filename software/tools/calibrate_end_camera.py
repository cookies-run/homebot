#!/usr/bin/env python3
"""End camera intrinsic calibration tool using a ChArUco board.

Usage:
    cd /Users/yi/works/智绘屿/yu-freetime/robots/HomeBot
    python -m software.tools.calibrate_end_camera

Procedure:
    1. Print a ChArUco board (or use a monitor/screen displaying one).
    2. Hold/move the board in front of the end camera at various distances
       and angles.
    3. Press SPACE to capture a frame, or let auto-capture run.
    4. Press ESC / 'q' when enough frames are collected (aim for 20-30).
    5. The script computes fx, fy, cx, cy, distortion and saves them to:
       skills/homebot-skill/scripts/calibration_data/end_camera_calibration.json

Default board: 5x7 squares, 25mm squares, 15mm markers, DICT_4X4_50.
You can override these with command-line flags.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: this script lives in software/tools but needs VideoSubscriber
# and camera_geometry from skills/homebot-skill/scripts.
# ---------------------------------------------------------------------------
_ROBOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_SCRIPTS_DIR = os.path.join(_ROBOT_ROOT, "skills", "homebot-skill", "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from video_subscriber import VideoSubscriber
from camera_geometry import CameraCalibration, save_calibration

# ---------------------------------------------------------------------------
# OpenCV / ArUco helpers with version compatibility.
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError as e:
    print("[ERROR] OpenCV is required. Install with: pip install opencv-contrib-python")
    raise


def _get_aruco_dict():
    """Return the 4x4_50 ArUco dictionary across OpenCV versions."""
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def _make_charuco_board(squares_x: int, squares_y: int, square_len: float, marker_len: float):
    """Create a CharucoBoard compatible with the installed OpenCV version."""
    aruco_dict = _get_aruco_dict()
    if hasattr(cv2.aruco, "CharucoBoard"):
        # OpenCV 4.7+
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, aruco_dict)
    # Older OpenCV
    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_len, marker_len, aruco_dict)


def _detect_charuco_corners(
    gray: np.ndarray,
    board,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect Charuco corners and IDs. Returns (charucoCorners, charucoIds)."""
    aruco_dict = _get_aruco_dict()
    if hasattr(cv2.aruco, "CharucoDetector"):
        # OpenCV 4.7+
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        return charuco_corners, charuco_ids

    # Older OpenCV path
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is None or len(ids) == 0:
        return None, None
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if not ret or charuco_ids is None or len(charuco_ids) < 4:
        return None, None
    return charuco_corners, charuco_ids


def _draw_board(board, size: Tuple[int, int], output_path: str):
    """Render the ChArUco board to an image so the user can print it."""
    margin = 0
    if hasattr(board, "generateImage"):
        img = board.generateImage(size, margin)
    else:
        # Fallback for very old versions
        img = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.imwrite(output_path, img)
    print(f"[INFO] ChArUco board saved to: {output_path}")


# ---------------------------------------------------------------------------
# Calibration logic
# ---------------------------------------------------------------------------
class EndCameraCalibrator:
    def __init__(
        self,
        robot_ip: str,
        video_port: int,
        squares_x: int = 5,
        squares_y: int = 7,
        square_length_m: float = 0.025,
        marker_length_m: float = 0.015,
        auto_capture_interval_s: float = 0.8,
        min_corners: int = 8,
    ):
        self.robot_ip = robot_ip
        self.video_port = video_port
        self.board = _make_charuco_board(
            squares_x, squares_y, square_length_m, marker_length_m
        )
        self.auto_capture_interval_s = auto_capture_interval_s
        self.min_corners = min_corners

        self.all_charuco_corners: List[np.ndarray] = []
        self.all_charuco_ids: List[np.ndarray] = []
        self.frame_size: Optional[Tuple[int, int]] = None

    def collect_frames(self, min_frames: int = 20, max_frames: int = 40) -> bool:
        """Open the video stream and collect valid calibration frames."""
        subscriber = VideoSubscriber(self.robot_ip, self.video_port)
        print(f"[INFO] Connecting to end camera at {self.robot_ip}:{self.video_port}...")

        window_name = "End Camera Calibration - SPACE: capture, Q/ESC: finish"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        last_auto_capture = 0.0
        frame_count = 0
        instructions_printed = False

        try:
            while True:
                frame_bytes = subscriber.wait_for_frame(timeout_seconds=2.0)
                if frame_bytes is None:
                    print("[WARN] Frame timeout, retrying...")
                    continue

                frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                if self.frame_size is None:
                    self.frame_size = (frame.shape[1], frame.shape[0])

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                charuco_corners, charuco_ids = _detect_charuco_corners(gray, self.board)

                display = frame.copy()
                if charuco_corners is not None:
                    cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                    count = len(charuco_ids)
                    color = (0, 255, 0) if count >= self.min_corners else (0, 165, 255)
                    cv2.putText(
                        display,
                        f"corners: {count} | collected: {len(self.all_charuco_ids)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
                else:
                    cv2.putText(
                        display,
                        f"No board detected | collected: {len(self.all_charuco_ids)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.putText(
                    display,
                    "SPACE: capture | A: auto-capture toggle | Q/ESC: finish",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:  # ESC
                    break
                elif key == ord(" "):
                    if charuco_corners is not None and len(charuco_ids) >= self.min_corners:
                        self.all_charuco_corners.append(charuco_corners)
                        self.all_charuco_ids.append(charuco_ids)
                        frame_count += 1
                        print(
                            f"[INFO] Captured frame {frame_count}/{max_frames} "
                            f"({len(charuco_ids)} corners)"
                        )
                    else:
                        print("[WARN] Board not detected clearly, frame not captured")
                elif key == ord("a"):
                    instructions_printed = not instructions_printed
                    print(
                        f"[INFO] Auto-capture {'enabled' if instructions_printed else 'disabled'}"
                    )

                # Auto-capture when enabled and enough corners are visible.
                if instructions_printed and charuco_corners is not None:
                    now = time.time()
                    if now - last_auto_capture >= self.auto_capture_interval_s:
                        if len(self.all_charuco_ids) == 0 or not np.array_equal(
                            charuco_ids, self.all_charuco_ids[-1]
                        ):
                            self.all_charuco_corners.append(charuco_corners)
                            self.all_charuco_ids.append(charuco_ids)
                            frame_count += 1
                            last_auto_capture = now
                            print(
                                f"[INFO] Auto-captured frame {frame_count}/{max_frames} "
                                f"({len(charuco_ids)} corners)"
                            )

                if frame_count >= max_frames:
                    print(f"[INFO] Reached max frames ({max_frames})")
                    break

        finally:
            cv2.destroyAllWindows()
            subscriber.close()

        if len(self.all_charuco_ids) < min_frames:
            print(
                f"[ERROR] Only {len(self.all_charuco_ids)} valid frames collected; "
                f"need at least {min_frames}."
            )
            return False
        return True

    def calibrate(self) -> Optional[CameraCalibration]:
        """Run OpenCV camera calibration on collected frames."""
        if self.frame_size is None or len(self.all_charuco_ids) < 5:
            print("[ERROR] Not enough calibration data.")
            return None

        print(f"[INFO] Calibrating with {len(self.all_charuco_ids)} frames...")

        # Prepare object points for each captured frame.
        object_points = []
        image_points = []
        for corners, ids in zip(self.all_charuco_corners, self.all_charuco_ids):
            obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
            if obj_pts is None or len(obj_pts) < self.min_corners:
                continue
            object_points.append(obj_pts)
            image_points.append(img_pts)

        if len(object_points) < 5:
            print("[ERROR] Too few frames with enough matched points.")
            return None

        image_size = self.frame_size
        flags = cv2.CALIB_RATIONAL_MODEL if hasattr(cv2, "CALIB_RATIONAL_MODEL") else 0

        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=object_points,
            imagePoints=image_points,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=flags,
        )

        if not ret:
            print("[ERROR] Calibration failed.")
            return None

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        D = D.flatten()

        # Compute reprojection error for reporting.
        total_error = 0.0
        total_points = 0
        for obj_pts, img_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
            error = cv2.norm(img_pts, proj, cv2.NORM_L2) / len(proj)
            total_error += error
            total_points += len(proj)
        mean_error = total_error / len(object_points) if object_points else float("inf")

        print("[INFO] Calibration succeeded:")
        print(f"       resolution: {image_size[0]}x{image_size[1]}")
        print(f"       fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
        print(f"       distortion: {D.tolist()}")
        print(f"       mean reprojection error: {mean_error:.3f} px")

        return CameraCalibration(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist_coeffs=D,
            width=image_size[0],
            height=image_size[1],
            T_camera_frame=None,  # extrinsics calibrated separately in Phase 2
        )


def main():
    parser = argparse.ArgumentParser(description="End camera intrinsic calibration")
    parser.add_argument("--ip", default=None, help="Robot IP (default from robot_config)")
    parser.add_argument("--port", type=int, default=None, help="End camera ZeroMQ port")
    parser.add_argument("--squares-x", type=int, default=5, help="ChArUco squares in X")
    parser.add_argument("--squares-y", type=int, default=7, help="ChArUco squares in Y")
    parser.add_argument(
        "--square-length", type=float, default=0.025, help="Square side length in meters"
    )
    parser.add_argument(
        "--marker-length", type=float, default=0.015, help="Marker side length in meters"
    )
    parser.add_argument(
        "--min-frames", type=int, default=20, help="Minimum valid frames to collect"
    )
    parser.add_argument(
        "--max-frames", type=int, default=40, help="Maximum frames to collect"
    )
    parser.add_argument(
        "--generate-board",
        action="store_true",
        help="Render a printable ChArUco board and exit",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save calibration JSON (default: skills/homebot-skill/scripts/calibration_data)",
    )
    args = parser.parse_args()

    # Import robot_config now that _SCRIPTS_DIR is on sys.path.
    import robot_config as config

    robot_ip = args.ip or config.ROBOT_IP
    video_port = args.port or config.END_VIDEO_PORT

    board = _make_charuco_board(args.squares_x, args.squares_y, args.square_length, args.marker_length)

    if args.generate_board:
        # Render board at 300 DPI-ish pixel size for A4 printing.
        pixel_size = (
            int(args.squares_x * args.square_length * 1000 * 11.81),
            int(args.squares_y * args.square_length * 1000 * 11.81),
        )
        _draw_board(board, pixel_size, "charuco_board.png")
        return

    output_dir = args.output_dir or os.path.join(_SCRIPTS_DIR, "calibration_data")

    calibrator = EndCameraCalibrator(
        robot_ip=robot_ip,
        video_port=video_port,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length,
        marker_length_m=args.marker_length,
    )

    print("=" * 60)
    print("End Camera Intrinsic Calibration")
    print("=" * 60)
    print("Instructions:")
    print("  1. Show a ChArUco board to the end camera.")
    print("  2. Press SPACE to capture a frame when the board is detected.")
    print("  3. Move the board to different distances/angles; collect 20-30 frames.")
    print("  4. Press 'a' to toggle auto-capture.")
    print("  5. Press 'q' or ESC when done.")
    print("=" * 60)

    if not calibrator.collect_frames(min_frames=args.min_frames, max_frames=args.max_frames):
        sys.exit(1)

    calibration = calibrator.calibrate()
    if calibration is None:
        sys.exit(1)

    saved_path = save_calibration(calibration, "end", output_dir)
    print(f"[OK] Calibration saved to: {saved_path}")

    # Also print JSON content for manual inspection.
    with open(saved_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
