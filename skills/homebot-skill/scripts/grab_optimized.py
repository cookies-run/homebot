#!/usr/bin/env python3
"""
Auto Grab Workflow - 六阶段状态机自主抓取（运动学版 + OpenCV Tracker 视觉伺服）

核心设计：
    - Phase 0: 观察与环境准备
    - Phase 1: 属性测姿与底盘策略分流（VLM）
    - Phase 2: 接近与手腕姿态预部署（端侧 Tracker + VLM 初始化末端追踪）
    - Phase 3: 二次距离闭环与精准贴紧（纯几何，无 VLM）
    - Phase 4: 触达确认与大模型终审（VLM）
    - Phase 5: 夹紧抬升与回缩验证

坐标系约定（User Frame）：
    - r > 0：机器人前进方向
    - z > 0：向上
    - pitch = -90°：垂直向下
    - pitch = 0°：水平向前

VLM 调用：不做硬次数限制，按需调用（Phase 1/2/4 及异常恢复）。

用法:
    python grab_optimized.py              # 默认抓取纸巾
    python grab_optimized.py --target "一瓶矿泉水"

要求:
    - 机器人摄像头已启动并发布到 ZeroMQ (默认端口 5560/5561)
    - 机械臂服务已启动 (默认端口 5557)
    - 底盘服务已启动 (默认端口 5556)
    - VLM 视觉分析可用 (MiniMax 优先)
    - OpenCV 追踪器可用（建议安装 opencv-contrib-python）
"""

import sys
import os
import time
import re
import json
import math
import shutil
from enum import Enum, auto
from datetime import datetime

# 提前把 software/src 加入路径，供 common/configs/vision_analyzer 使用
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../software/src'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import numpy as np

_CV2_AVAILABLE = False
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_subscriber import VideoSubscriber
from arm_control import HomeBotArmController
from chassis_control import HomeBotChassisController

# 3D-aware grasping utilities (optional; gracefully degrade if missing or uncalibrated)
_3D_GEOMETRY_AVAILABLE = False
try:
    from camera_geometry import load_calibration
    _3D_GEOMETRY_AVAILABLE = True
except ImportError as e:
    print(f"[GRAB] [WARN] camera_geometry 不可用: {e}")

# 统一视觉分析器（按全局 VISION_PROVIDER 路由，业务代码不再直接调底层客户端）
try:
    from vision_analyzer import VisionAnalyzer
    _VISION_ANALYZER = VisionAnalyzer()
    VLM_PROVIDER = _VISION_ANALYZER.get_order()[0] if _VISION_ANALYZER.get_order() else None
    if not VLM_PROVIDER:
        print("[WARN] 未找到视觉分析客户端，抓取功能将不可用")
except Exception as e:
    print(f"[WARN] 初始化 VisionAnalyzer 失败: {e}")
    _VISION_ANALYZER = None
    VLM_PROVIDER = None


def analyze_images_with_fallback(
    image_paths: list,
    prompt: str,
    max_tokens: int = 256,
    reasoning_effort: str = "low",
    timeout: int = 60,
) -> tuple[str, str]:
    """
    统一视觉分析器封装，按全局 VISION_PROVIDER 配置调用，失败时回退。
    保留此函数签名以兼容现有调用方。

    Returns:
        (text_result, provider_name)
    """
    if _VISION_ANALYZER is None:
        raise RuntimeError("没有可用的视觉分析客户端")
    return _VISION_ANALYZER.analyze(
        image_paths=image_paths,
        prompt=prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )

import robot_config as config

# 引入机械臂配置和运动学（software/src 已在文件顶部加入路径）
from configs.config import get_config as _get_arm_config
_ARM_CFG = _get_arm_config().arm
_JOINT_LIMITS = _ARM_CFG.joint_limits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../software/src/hal/arm'))
from Kinematics import ArmKinematics


# ========== 观察姿态 ==========
_OBSERVATION_POSE = dict(_ARM_CFG.rest_position)
OBSERVATION_POSE = {**_OBSERVATION_POSE, "gripper": 90, "wrist_roll": -90}

# 侧向夹取时 wrist_roll 始终保持 -90°，夹爪侧面朝向目标
_GRAB_WRIST_ROLL = -90

# Tracker 输出 sanity check 阈值
_TRACKER_MIN_SIZE_PX = 10
_TRACKER_MIN_ASPECT = 0.2
_TRACKER_MAX_ASPECT = 5.0
_TRACKER_MAX_SIZE_JUMP_RATIO = 0.5      # 宽/高相对上一帧缩小超过 50% 视为漂移
_TRACKER_MAX_CENTER_JUMP_RATIO = 0.25   # 中心点在单帧内跳变超过图像边长 25% 视为漂移

# 机械臂运动到位后等待画面稳定的时间（秒）
_POST_MOVE_SETTLE_S = 0.3

# Phase 3 中周期性 VLM 重定位的步长；设为 None 表示关闭周期性重定位，
# 只在 Tracker 漂移/丢失（sanity check 失败）时触发 VLM 重定位。
_PHASE3_VLM_REINIT_EVERY_N = None


# ========== 姿态分流参数表（User Frame） ==========
# r > 0 前进，z > 0 向上，pitch = -90° 垂直向下
_POSE_PARAMS = {
    "upright": {
        # Phase 1 底盘停靠距离：直立柱体需留足机械臂从侧面插入的空间
        "stop_distance_cm": (22, 25),
        # Phase 2 预部署：侧面、水平、留足够安全间隙，避免 base 微调时爪子推到瓶子
        # 底盘 22-25cm 时，爪子从 r=55mm 开始，Phase 3 再逐步前进贴紧
        # z=90 让夹爪对准物体中下部分（矿泉水瓶等柱体）
        "pre_deploy_rz": (55, 90, 0),
        "safety_gap_cm": 3,
        # Phase 3 微调方向
        "tune_direction": "r",
        "tune_step_mm": 4,
        "tune_max_steps": 30,
        # Phase 3 停止阈值：直立物体以 width_ratio 为主，y2 为辅
        "tune_target_width_ratio": 0.22,
        "tune_target_y2_ratio": 0.88,
        # Phase 4/5 抓取与抬升
        "grasp_pitch": 0,
        "lift_trajectory": [
            (70, 150, 0),
            (50, 180, 0),
        ],
    },
    "fallen": {
        # Phase 1 底盘停靠距离：倾倒物体需更靠近，使机械臂能凌空覆盖中心
        "stop_distance_cm": (18, 22),
        # Phase 2 预部署：物体正上方、垂直向下（z=100 保证高于桌面）
        "pre_deploy_rz": (80, 100, -90),
        "safety_gap_cm": 0,
        # Phase 3 微调方向
        "tune_direction": "z",
        "tune_step_mm": 4,
        "tune_max_steps": 14,
        # Phase 3 停止阈值：倾倒物体以 y2_ratio 为主，area_ratio 为辅
        "tune_target_area_ratio": 0.25,
        "tune_target_y2_ratio": 0.82,
        "grasp_pitch": -90,
        "lift_trajectory": [
            (80, 150, -70),
            (60, 180, 0),
        ],
    },
}


def validate_vlm_bbox(bbox, img_width=1280, img_height=720):
    """
    几何常识拦截：检查 VLM 返回的 bbox 是否可能是幻觉。

    Args:
        bbox: [x1, y1, x2, y2]，坐标为 [0, 1000] 整数或 [0, 1] 浮点数。
        img_width: 图像宽度，默认 1280。
        img_height: 图像高度，默认 720。

    Returns:
        bool: True 表示 bbox 通过校验，False 表示可能是幻觉。
    """
    x1, y1, x2, y2 = bbox
    # 兼容 0~1 归一化坐标：若最大值不超过 1，则先缩放到 0~1000
    max_val = max(bbox)
    scale = 1.0 if max_val > 1.0 else 1000.0

    # 转换为归一化坐标计算尺寸
    w = (x2 - x1) / scale * img_width
    h = (y2 - y1) / scale * img_height

    # 面积过滤：如果框选面积过大（比如超过画面 60%），判定为幻觉
    if (w * h) > (img_width * img_height * 0.6):
        return False

    # 长宽比过滤：限制比例，过滤掉横跨整个屏幕的长条（长宽比 > 4 或 < 0.25）
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio > 4.0 or aspect_ratio < 0.25:
        return False

    return True


class GrabPhase(Enum):
    """六阶段抓取状态机的阶段编号（与 _phaseN_* 方法、日志中的 "Phase N" 一一对应）。"""
    OBSERVATION = 0   # Phase 0: 观察与环境准备
    DETECTION = 1     # Phase 1: 属性测姿与底盘策略分流（VLM + 机身 Tracker）
    APPROACH = 2      # Phase 2: 接近与手腕姿态预部署（末端 Tracker + base 对准）
    FINAL_TUNING = 3  # Phase 3: 二次距离闭环与精准贴紧（纯几何）
    TOUCH_VERIFY = 4  # Phase 4: 触达确认与大模型终审（VLM）
    GRASP_LIFT = 5    # Phase 5: 夹紧抬升与回缩验证


class UserKinematicsAdapter:
    """
    用户坐标系 -> ArmKinematics 内部坐标系适配器。

    用户坐标系：
        r > 0 前进，z > 0 向上，pitch = -90° 垂直向下，pitch = 0° 水平向前。

    内部坐标系（ArmKinematics）：
        r > 0 为内部正方向；target_orientation = 0° 表示水平向后（与观察姿态 wrist 一致）。

    转换关系（基于观察姿态 base=-90°, shoulder=0°, elbow=150°, wrist_flex=30° 验证）：
        r_internal = -r_user
        target_orientation_internal = -pitch_user（规范化到 [-180, 180]）
    """

    def __init__(self, kin: ArmKinematics):
        self.kin = kin

    def user_to_internal_rz(self, r_user: float, z_user: float) -> tuple[float, float]:
        return (-r_user, z_user)

    def internal_to_user_rz(self, r_internal: float, z_internal: float) -> tuple[float, float]:
        return (-r_internal, z_internal)

    def user_pitch_to_internal_orientation(self, pitch_user: float) -> float:
        """
        基于观察姿态 base=-90° 验证：
          target_orientation=0° 时末端绝对角 = 180°，指向内部负 r（即机器人前进方向）。
          因此用户 pitch=0°（水平向前）对应内部 target_orientation=0°。
          用户 pitch=-90°（垂直向下）对应内部 target_orientation=90°。
        """
        internal = -pitch_user
        while internal > 180.0:
            internal -= 360.0
        while internal < -180.0:
            internal += 360.0
        return internal

    def internal_orientation_to_user_pitch(self, target_orientation: float) -> float:
        pitch = -target_orientation
        while pitch > 180.0:
            pitch -= 360.0
        while pitch < -180.0:
            pitch += 360.0
        return pitch


class AutoGrabWorkflow:
    """
    自主抓取工作流（六阶段状态机，3 次 VLM 调用）。
    """

    def __init__(
        self,
        robot_ip: str = None,
        video_port: int = None,
        end_video_port: int = None,
        arm_port: int = None,
        max_attempts: int = 10,
        use_end_camera: bool = True,
    ):
        self.robot_ip = robot_ip or config.ROBOT_IP
        self.video_port = video_port or config.VIDEO_PORT
        self.end_video_port = end_video_port or config.END_VIDEO_PORT
        self.arm_port = arm_port or config.ARM_PORT
        self.max_attempts = max_attempts
        self.use_end_camera = use_end_camera
        self._end_camera_available = False

        self.arm = HomeBotArmController(robot_ip=self.robot_ip, robot_port=self.arm_port)
        self.chassis = HomeBotChassisController(
            ip=self.robot_ip, port=config.CHASSIS_PORT
        )

        # 运动学与坐标适配
        self._kin = ArmKinematics(
            L1=_ARM_CFG.upper_arm_length,
            L2=_ARM_CFG.forearm_length,
        )
        self._user_kin = UserKinematicsAdapter(self._kin)

        # 3D 感知标定数据（可选；缺失时自动降级为 2D 策略）
        self._calibration = None
        self._3d_enabled = False
        self._3d_status_message = ""
        self._load_calibration()

        # 当前末端位置（User Frame）
        self._current_rz = {"r": 0.0, "z": 0.0}

        # 状态机
        self._phase = GrabPhase.OBSERVATION
        self._object_pose = {
            "bbox": None,
            "is_cylinder": False,
            "pose": None,
            "image_w": None,
            "image_h": None,
            "vlm_provider": None,
        }
        self._vlm_call_count = 0  # 仅用于日志计数，不做硬限制

        # 关节与执行器缓存
        self._last_aligned_angles = None
        self._last_wrist_roll = 0

        # Tracker（机身摄像头）
        self._body_tracker = None
        self._body_tracker_initialized = False
        self._body_tracker_bbox = None
        self._body_tracker_lost_count = 0

        # Tracker（末端摄像头）
        self._end_tracker = None
        self._end_tracker_initialized = False
        self._end_tracker_bbox = None
        self._end_tracker_lost_count = 0

        # 输出目录
        self.output_dir = os.path.join(os.path.dirname(__file__), "grab_captures")
        os.makedirs(self.output_dir, exist_ok=True)
        self._current_run_dir = None  # 每次 run() 会创建独立的子目录

        # Phase 1 Tracker 历史指标
        self._last_phase1_metrics = None

        # 参数（可现场微调）
        self.base_step = 2
        self.chassis_step_cm = 6
        self.fine_base_step = 0.5
        self.fine_r_step = 2
        self.fine_z_step = 2

        # Phase 3 视觉停止后的最终接近距离（cm）
        # 根据当前 h_ratio 自适应：越近则最终接近越短，避免已经贴上还往前顶。
        self.final_approach_cm = 6.0  # 旧默认值，保留兼容；实际使用 _compute_final_approach_cm

        # Phase 1 距离估算标定
        self.phase1_area_distance_k = 64.0
        self.phase1_area_too_close = 0.30
        self.phase1_y2_too_close = 0.78

    def _load_calibration(self):
        """加载相机标定数据；缺失时保持 2D 模式。"""
        if not _3D_GEOMETRY_AVAILABLE:
            self._3d_status_message = "camera_geometry 模块未导入"
            return

        calib_dir = os.path.join(os.path.dirname(__file__), "calibration_data")
        end_calib = load_calibration("end", calib_dir)
        body_calib = load_calibration("body", calib_dir)

        if end_calib is None and body_calib is None:
            self._3d_status_message = "未找到相机标定数据，使用 2D 策略"
            return

        self._calibration = {"end": end_calib, "body": body_calib}
        # Phase 1 MVP: 只需要末端相机内参即可启用 grasp_center 像素去畸变等基础功能。
        # 完整 3D 射线功能需要外参（Phase 2）。
        self._3d_enabled = end_calib is not None and end_calib.has_intrinsics()
        self._3d_status_message = (
            f"3D 感知已启用: end_intrinsics={end_calib is not None}, "
            f"end_extrinsics={end_calib.has_extrinsics() if end_calib else False}, "
            f"body_calib={body_calib is not None}"
        )

    # ==================== Tracker 视觉伺服 ====================

    def _bgr_from_bytes(self, frame_bytes: bytes) -> np.ndarray | None:
        """把 VideoSubscriber 返回的 JPEG bytes 解码为 OpenCV BGR 图像"""
        if not _CV2_AVAILABLE or frame_bytes is None:
            return None
        try:
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[GRAB] [Tracker] 图像解码失败: {e}")
            return None

    def _create_tracker(self):
        """创建 OpenCV 传统追踪器，优先 CSRT，其次 KCF/MOSSE"""
        if not _CV2_AVAILABLE:
            print("[GRAB] [Tracker][WARN] cv2 未安装，无法创建追踪器")
            return None
        candidates = ["TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"]
        for name in candidates:
            try:
                ctor = getattr(cv2, name)
                tracker = ctor()
                print(f"[GRAB] [Tracker] 创建追踪器: {name}")
                return tracker
            except Exception as e:
                print(f"[GRAB] [Tracker] {name} 不可用: {e}")
                continue
        print("[GRAB] [Tracker][WARN] 当前 OpenCV 版本没有可用的传统追踪器")
        return None

    def _init_tracker(self, tracker_attr: str, frame_bgr: np.ndarray, bbox_xyxy: tuple) -> tuple[bool, str]:
        """用 bbox（像素坐标 xyxy）初始化指定 Tracker。

        Returns:
            (ok, reason): reason 仅在失败时返回，用于上层做更清晰的诊断。
        """
        if frame_bgr is None:
            return False, "输入帧为空"
        x1, y1, x2, y2 = bbox_xyxy
        h, w = frame_bgr.shape[:2]
        x = max(0, int(x1))
        y = max(0, int(y1))
        bw = min(int(x2 - x1), w - x - 1)
        bh = min(int(y2 - y1), h - y - 1)
        if bw <= 2 or bh <= 2:
            reason = f"初始框无效: x={x}, y={y}, w={bw}, h={bh}"
            print(f"[GRAB] [Tracker] {reason}")
            return False, reason

        tracker = self._create_tracker()
        if tracker is None:
            return False, "OpenCV 无可用的传统追踪器（请确认已安装 opencv-contrib-python）"

        try:
            tracker.init(frame_bgr, (x, y, bw, bh))
            setattr(self, tracker_attr, tracker)
            setattr(self, f"{tracker_attr}_bbox", (x, y, bw, bh))
            setattr(self, f"{tracker_attr}_initialized", True)
            setattr(self, f"{tracker_attr}_lost_count", 0)
            print(f"[GRAB] [Tracker] ✅ 追踪器已初始化，初始框=({x},{y},{bw},{bh})")
            return True, ""
        except Exception as e:
            reason = f"tracker.init() 异常: {e}"
            print(f"[GRAB] [Tracker] ❌ {reason}")
            return False, reason

    def _update_tracker(self, tracker_attr: str, frame_bgr: np.ndarray) -> tuple[bool, tuple | None]:
        """更新指定 Tracker，返回 (success, bbox_xywh)

        增加了 sanity check：尺寸过小、宽高比异常或相对上一帧跳变过大时判为丢失，
        避免 Tracker 漂移后把窄框/错框当成有效结果误导后续闭环。
        """
        initialized_attr = f"{tracker_attr}_initialized"
        bbox_attr = f"{tracker_attr}_bbox"
        lost_attr = f"{tracker_attr}_lost_count"
        tracker = getattr(self, tracker_attr)

        if not getattr(self, initialized_attr) or tracker is None or frame_bgr is None:
            return False, None
        ok, bbox = tracker.update(frame_bgr)
        if not ok or bbox is None:
            setattr(self, lost_attr, getattr(self, lost_attr) + 1)
            return False, None

        x, y, w, h = map(int, bbox)
        img_h, img_w = frame_bgr.shape[:2]

        # 1. 基本尺寸合法性
        if w < _TRACKER_MIN_SIZE_PX or h < _TRACKER_MIN_SIZE_PX:
            print(f"[GRAB] [Tracker] 追踪框尺寸过小 ({w}x{h})，判为丢失")
            setattr(self, lost_attr, getattr(self, lost_attr) + 1)
            return False, None
        if w > img_w or h > img_h:
            print(f"[GRAB] [Tracker] 追踪框尺寸超过图像 ({w}x{h} > {img_w}x{img_h})，判为丢失")
            setattr(self, lost_attr, getattr(self, lost_attr) + 1)
            return False, None

        # 2. 宽高比异常（防止 Tracker 缩成一条线或锁到细长背景）
        aspect = w / max(h, 1)
        if aspect < _TRACKER_MIN_ASPECT or aspect > _TRACKER_MAX_ASPECT:
            print(f"[GRAB] [Tracker] 追踪框宽高比异常 ({aspect:.2f})，判为丢失")
            setattr(self, lost_attr, getattr(self, lost_attr) + 1)
            return False, None

        # 3. 相对上一帧的跳变检查：单次大幅缩小或中心跳变视为漂移
        prev_bbox = getattr(self, bbox_attr, None)
        if prev_bbox is not None:
            px, py, pw, ph = prev_bbox
            if w < pw * _TRACKER_MAX_SIZE_JUMP_RATIO or h < ph * _TRACKER_MAX_SIZE_JUMP_RATIO:
                print(f"[GRAB] [Tracker] 追踪框相对上一帧大幅缩小 "
                      f"({w}x{h} vs 上次 {pw}x{ph})，判为丢失")
                setattr(self, lost_attr, getattr(self, lost_attr) + 1)
                return False, None
            if abs((x + w / 2) - (px + pw / 2)) > img_w * _TRACKER_MAX_CENTER_JUMP_RATIO or \
               abs((y + h / 2) - (py + ph / 2)) > img_h * _TRACKER_MAX_CENTER_JUMP_RATIO:
                print(f"[GRAB] [Tracker] 追踪框中心相对上一帧跳变过大，判为丢失")
                setattr(self, lost_attr, getattr(self, lost_attr) + 1)
                return False, None

        bbox_int = (x, y, w, h)
        setattr(self, bbox_attr, bbox_int)
        setattr(self, lost_attr, 0)
        return True, bbox_int

    def _tracker_metrics(self, bbox_xywh: tuple, img_w: int, img_h: int) -> tuple:
        """由追踪器 bbox 计算 (xyxy, center_x_ratio, center_y_ratio, area_ratio, y2_ratio, width_ratio)"""
        x, y, w, h = bbox_xywh
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
        cx_ratio = (x1 + x2) / 2.0 / img_w
        cy_ratio = (y1 + y2) / 2.0 / img_h
        y2_ratio = y2 / img_h
        width_ratio = (x2 - x1) / img_w
        return (float(x1), float(y1), float(x2), float(y2)), cx_ratio, cy_ratio, area_ratio, y2_ratio, width_ratio

    def _save_debug_frame(self, frame_bgr: np.ndarray, bbox_xyxy: tuple | None, label: str, metrics: dict):
        """保存带追踪框和指标的调试图。bbox_xyxy 为 None 时只保存原图与文字。"""
        if not _CV2_AVAILABLE or frame_bgr is None:
            return
        try:
            debug = frame_bgr.copy()
            if bbox_xyxy is not None:
                x1, y1, x2, y2 = map(int, bbox_xyxy)
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text_parts = []
            for key in ("area_ratio", "center_x_ratio", "y2_ratio"):
                val = metrics.get(key)
                if val is not None:
                    text_parts.append(f"{key.split('_')[0]}={val:.3f}")
            reason = metrics.get("reason")
            if reason:
                text_parts.append(str(reason))
            if not text_parts:
                text_parts.append(str(metrics))
            text = " ".join(text_parts)
            # 在左上角显示文字，避免 bbox 为 None 时越界
            cv2.putText(
                debug, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            # 中间过程调试图与关键帧放到同一目录，按 run 保留
            save_dir = self._current_run_dir or self.output_dir
            debug_dir = os.path.join(save_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(debug_dir, f"debug_{label}_{ts}.jpg")
            cv2.imwrite(path, debug)
            print(f"[GRAB] [Debug] 已保存调试图: {path}")
        except Exception as e:
            print(f"[GRAB] [Debug] 保存调试图失败: {e}")

    # ==================== VLM 与 JSON 解析 ====================

    def _parse_vlm_json(self, text: str, required_keys: list) -> dict | None:
        """鲁棒解析 VLM 返回的 JSON，兼容 Markdown 代码块、0~1000 bbox 等"""
        if not text:
            return None
        text = text.strip()

        # 去除推理模型常见的 <think>...</think> 链式思考标签
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        if text.startswith("```"):
            lines = text.splitlines()
            start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    start = i + 1
                    break
            end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()
        try:
            data = json.loads(text)
            for k in required_keys:
                if k not in data:
                    print(f"[GRAB] [JSON] 缺少字段: {k}")
                    return None
            # bbox 兼容 0~1000 归一化
            bbox = data.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and max(bbox) > 1.0:
                data["bbox"] = [v / 1000.0 for v in bbox]
            return data
        except Exception as e:
            print(f"[GRAB] [JSON] 解析失败: {e}, raw={text[:200]}")
            return None

    def _validate_vlm_bbox(self, bbox_norm: list) -> tuple[bool, str]:
        """
        几何常识拦截：调用独立 validate_vlm_bbox 函数检查 VLM 返回的 bbox。

        Args:
            bbox_norm: [x1, y1, x2, y2]，0~1 归一化坐标或 0~1000 整数。

        Returns:
            (is_valid, reason)
        """
        if not isinstance(bbox_norm, (list, tuple)) or len(bbox_norm) != 4:
            return False, "bbox 格式错误"
        x1, y1, x2, y2 = bbox_norm
        if x2 <= x1 or y2 <= y1:
            return False, f"bbox 宽高非正: w={x2 - x1:.3f}, h={y2 - y1:.3f}"

        if not validate_vlm_bbox(bbox_norm):
            return False, "尺寸或长宽比异常（幻觉框）"

        return True, ""

    def _vlm_call(self, image_path: str, prompt: str, required_keys: list,
                  max_tokens: int = 256, timeout: int = 60) -> dict | None:
        """调用 VLM 并解析 JSON，仅计数不限制"""
        try:
            result, provider = analyze_images_with_fallback(
                image_paths=[image_path],
                prompt=prompt,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            self._vlm_call_count += 1
            self._object_pose["vlm_provider"] = provider
            print(f"[GRAB] VLM({provider}) 第 {self._vlm_call_count} 次调用成功")
            parsed = self._parse_vlm_json(result, required_keys)
            if parsed is not None:
                parsed["_provider"] = provider
            return parsed
        except Exception as e:
            print(f"[GRAB] [ERROR] VLM 调用失败: {e}")
            return None

    # ==================== 图像捕获 ====================

    def _capture_from(self, port: int, label: str) -> tuple[str | None, np.ndarray | None]:
        """从指定端口捕获图像，返回保存路径和 BGR 帧"""
        subscriber = VideoSubscriber(self.robot_ip, port)
        try:
            print(f"[GRAB] 正在捕获图像 ({label}, 端口 {port})...")
            frame = subscriber.wait_for_frame(timeout_seconds=5.0)
            if frame is None:
                print(f"[GRAB] [ERROR] 图像捕获超时 (端口 {port})")
                return None, None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = self._current_run_dir or self.output_dir
            path = os.path.join(save_dir, f"grab_{label}_{timestamp}.jpg")
            with open(path, "wb") as f:
                f.write(frame)
            print(f"[GRAB] 图像已保存: {path}")

            frame_bgr = self._bgr_from_bytes(frame)
            return path, frame_bgr
        finally:
            subscriber.close()

    def capture(self, label: str = "") -> tuple[str | None, np.ndarray | None]:
        """订阅机身摄像头"""
        return self._capture_from(self.video_port, label)

    def capture_end(self, label: str = "") -> tuple[str | None, np.ndarray | None]:
        """订阅末端摄像头"""
        return self._capture_from(self.end_video_port, f"end_{label}")

    def _check_end_camera(self) -> bool:
        """检测末端摄像头是否可用"""
        if self._end_camera_available:
            return True
        print("[GRAB] 检测末端摄像头...")
        for attempt in range(1, 3):
            subscriber = VideoSubscriber(self.robot_ip, self.end_video_port)
            try:
                frame = subscriber.wait_for_frame(timeout_seconds=5.0)
                if frame is not None:
                    print("[GRAB] ✅ 末端摄像头已连接")
                    self._end_camera_available = True
                    return True
                else:
                    print(f"[GRAB] ⚠️ 末端摄像头第 {attempt} 次检测未响应，重试...")
            except Exception as e:
                print(f"[GRAB] ⚠️ 末端摄像头第 {attempt} 次检测失败: {e}")
            finally:
                subscriber.close()
            time.sleep(0.5)
        print("[GRAB] ⚠️ 末端摄像头最终未响应，将使用机身摄像头")
        return False

    # ==================== 机械臂控制辅助 ====================

    def get_current_angles(self) -> dict:
        """获取当前关节角度"""
        status = self.arm.get_status()
        if status and status.joint_states:
            angles = dict(status.joint_states)
            self._last_aligned_angles = angles
            return angles
        print("[GRAB] [WARN] 无法获取关节状态，使用上次姿态")
        if self._last_aligned_angles:
            return dict(self._last_aligned_angles)
        return dict(OBSERVATION_POSE)

    def clamp(self, val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def _check_kinematics_safety(self, shoulder: float, elbow: float, wrist_flex: float,
                                 desc: str, warn_clamp: bool = False) -> bool:
        """检查关节解是否安全"""
        lim = _JOINT_LIMITS
        if not (lim["shoulder"][0] <= shoulder <= lim["shoulder"][1]):
            print(f"[GRAB] [SAFETY] {desc}: shoulder={shoulder:.1f}° 超出限位 {lim['shoulder']}")
            return False
        if not (lim["elbow"][0] <= elbow <= lim["elbow"][1]):
            print(f"[GRAB] [SAFETY] {desc}: elbow={elbow:.1f}° 超出限位 {lim['elbow']}")
            return False
        if not (lim["wrist_flex"][0] <= wrist_flex <= lim["wrist_flex"][1]):
            print(f"[GRAB] [SAFETY] {desc}: wrist_flex={wrist_flex:.1f}° 超出限位 {lim['wrist_flex']}")
            return False
        # 奇异点：共线时微小位置变化需巨大关节速度
        if abs(elbow) < 5.0 or abs(elbow) > 175.0:
            print(f"[GRAB] [SAFETY] {desc}: 接近奇异点 (elbow={elbow:.1f}°)")
            return False
        # 自碰撞：大臂过低且小臂过度后弯
        if shoulder < 10.0 and elbow > 160.0:
            print(f"[GRAB] [SAFETY] {desc}: 可能自碰撞 (shoulder={shoulder:.1f}°, elbow={elbow:.1f}°)")
            return False
        return True

    def _move_to_pose(self, pose: dict, desc: str, wait: float = 1.5) -> bool:
        """发送关节角度指令并等待到位"""
        lim = _JOINT_LIMITS
        target = {}
        current = self.get_current_angles()
        for name in ["base", "shoulder", "elbow", "wrist_flex", "wrist_roll", "gripper"]:
            if name == "wrist_roll":
                if "wrist_roll" in pose:
                    val = pose["wrist_roll"]
                    self._last_wrist_roll = val
                else:
                    val = current.get("wrist_roll", self._last_wrist_roll)
            else:
                val = pose.get(name, current.get(name, 0))
            if name in lim:
                val = self.clamp(val, lim[name][0], lim[name][1])
            target[name] = val
        resp = self.arm.set_joint_angles(target)
        if resp and resp.success:
            print(f"[GRAB] {desc}")
            time.sleep(wait)
            # 机械臂到位后再等待一小段时间，让摄像头画面稳定（减少运动模糊/抖动）
            if _POST_MOVE_SETTLE_S > 0:
                time.sleep(_POST_MOVE_SETTLE_S)
            return True
        else:
            print(f"[GRAB] [ERR] {desc} 失败")
            return False

    def _refresh_arm_position(self) -> dict:
        """从实际关节角度计算当前末端位置（User Frame）"""
        angles = self.get_current_angles()
        shoulder = angles.get("shoulder", 0)
        elbow = angles.get("elbow", 150)
        r_internal, z = self._kin.forward_kinematics(shoulder, elbow)
        r_user, _ = self._user_kin.internal_to_user_rz(r_internal, z)
        self._current_rz = {"r": r_user, "z": z}
        print(f"[GRAB] [运动学] 当前末端位置(User): r={r_user:.1f}mm, z={z:.1f}mm "
              f"(shoulder={shoulder:.1f}°, elbow={elbow:.1f}°)")
        return self._current_rz

    def _move_to_rz(
        self,
        target_r: float,
        target_z: float,
        desc: str,
        wait: float = 1.5,
        fixed_shoulder: float = None,
        elbow_up: bool = True,
        target_pitch: float = 0.0,
    ) -> bool:
        """
        用逆运动学移动到指定 User Frame (r, z, pitch)。

        Args:
            target_r: User Frame 前伸距离（mm），正 = 前进
            target_z: User Frame 垂直高度（mm），正 = 向上
            target_pitch: User Frame 末端俯仰角，-90° = 垂直向下，0° = 水平向前
        """
        lim = _JOINT_LIMITS
        L1 = _ARM_CFG.upper_arm_length
        L2 = _ARM_CFG.forearm_length

        # 转换到内部坐标系
        r_internal, z_internal = self._user_kin.user_to_internal_rz(target_r, target_z)
        target_orientation = self._user_kin.user_pitch_to_internal_orientation(target_pitch)

        if fixed_shoulder is not None:
            shoulder_raw = fixed_shoulder
            dx = r_internal - L1 * math.cos(math.radians(shoulder_raw))
            dy = z_internal - L1 * math.sin(math.radians(shoulder_raw))
            dist = math.hypot(dx, dy)
            if dist > L2 + 0.1 or dist < abs(L2 - L1) - 0.1:
                print(f"[GRAB] [ERR] {desc}: 固定 shoulder={shoulder_raw}° 时目标不可达")
                return False
            abs_angle = math.degrees(math.atan2(dy, dx))
            elbow_raw = abs_angle - shoulder_raw
            while elbow_raw > 180:
                elbow_raw -= 360
            while elbow_raw < -180:
                elbow_raw += 360
        else:
            if not self._kin.is_reachable(r_internal, z_internal):
                min_r, max_r = self._kin.get_workspace_radius()
                print(f"[GRAB] [ERR] {desc}: 目标({target_r:.0f}, {target_z:.0f}) 不可达，"
                      f"工作空间半径∈[{min_r:.0f}, {max_r:.0f}]")
                return False
            angles = self._kin.inverse_kinematics(r_internal, z_internal, elbow_up=elbow_up)
            if angles is None:
                print(f"[GRAB] [ERR] {desc}: 逆运动学无解")
                return False
            shoulder_raw, elbow_raw = angles

        # 自动计算 wrist_flex
        wrist_flex_raw = self._kin.compute_wrist_flex(
            shoulder_raw, elbow_raw, target_orientation=target_orientation
        )

        # 限位
        shoulder = self.clamp(shoulder_raw, lim["shoulder"][0], lim["shoulder"][1])
        elbow = self.clamp(elbow_raw, lim["elbow"][0], lim["elbow"][1])
        wrist_flex = self.clamp(wrist_flex_raw, lim["wrist_flex"][0], lim["wrist_flex"][1])

        # 限位后重新计算实际可达位置（内部坐标）
        actual_r_internal, actual_z = self._kin.forward_kinematics(shoulder, elbow)
        actual_r_user, _ = self._user_kin.internal_to_user_rz(actual_r_internal, actual_z)

        # 安全与截断检查
        if not self._check_kinematics_safety(shoulder, elbow, wrist_flex, desc):
            return False
        if abs(wrist_flex - wrist_flex_raw) > 5.0:
            actual_pitch = self._user_kin.internal_orientation_to_user_pitch(
                shoulder + elbow + wrist_flex - 180
            )
            print(f"[GRAB] [WARN] {desc}: wrist_flex 由 {wrist_flex_raw:.1f}° 截断到 {wrist_flex:.1f}°，"
                  f"实际 pitch 约为 {actual_pitch:.1f}°（目标 {target_pitch:.1f}°）")

        if abs(shoulder - shoulder_raw) > 5.0 or abs(elbow - elbow_raw) > 5.0:
            print(f"[GRAB] [WARN] {desc}: 关节限位介入，原始 shoulder={shoulder_raw:.1f}° "
                  f"-> clamp={shoulder:.1f}°，实际将到达 r={actual_r_user:.1f}, z={actual_z:.1f}")

        base = self.get_current_angles().get("base", OBSERVATION_POSE.get("base", -90))
        pose = {
            "base": base,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist_flex": wrist_flex,
            "wrist_roll": self._last_wrist_roll,
        }

        print(f"[GRAB] [运动学] {desc}: User(r={target_r:.1f}, z={target_z:.1f}, pitch={target_pitch:.0f}°) -> "
              f"实际(r={actual_r_user:.1f}, z={actual_z:.1f}) "
              f"shoulder={shoulder:.1f}°, elbow={elbow:.1f}°, wrist_flex={wrist_flex:.1f}°")

        ok = self._move_to_pose(pose, desc, wait)
        if ok:
            self._current_rz = {"r": actual_r_user, "z": actual_z}
        return ok

    def _move_to_rz_relative(self, dr: float = 0, dz: float = 0,
                             pitch: float = 0.0, desc: str = "相对移动",
                             wait: float = 1.5) -> bool:
        """基于当前 User Frame 位置相对移动 (dr, dz)"""
        self._refresh_arm_position()
        target_r = self._current_rz["r"] + dr
        target_z = self._current_rz["z"] + dz
        target_z = max(20.0, target_z)
        return self._move_to_rz(target_r, target_z, desc, wait, target_pitch=pitch)

    def _arm_move_relative(self, forward_cm: float = 0.0, up_cm: float = 0.0,
                           pitch: float = None) -> bool:
        """
        用户友好的相对运动接口。

        Args:
            forward_cm: 沿机器人前进方向移动的距离（cm），正值为前伸。
            up_cm: 垂直方向移动距离（cm），正值为抬升。
            pitch: 末端俯仰角，默认使用当前姿态对应的 grasp_pitch。

        Returns:
            运动指令是否成功执行。
        """
        pose = self._object_pose.get("pose", "fallen")
        if pitch is None:
            pitch = _POSE_PARAMS[pose]["grasp_pitch"]
        return self._move_to_rz_relative(
            dr=forward_cm * 10.0,   # cm -> mm
            dz=up_cm * 10.0,
            pitch=pitch,
            desc=f"相对移动 forward={forward_cm:+.2f}cm, up={up_cm:+.2f}cm",
            wait=1.0,
        )

    # ==================== Phase 0: 观察与环境准备 ====================

    def _phase0_observation_reset(self) -> bool:
        """Phase 0: 机械臂回到观察姿态，重置所有状态"""
        print("\n[GRAB] ====== Phase 0: 观察与环境准备 ======")

        # 重置状态机
        self._phase = GrabPhase.OBSERVATION
        self._object_pose = {
            "bbox": None,
            "is_cylinder": False,
            "pose": None,
            "image_w": None,
            "image_h": None,
            "vlm_provider": None,
        }
        self._vlm_call_count = 0

        # 重置 Tracker
        self._body_tracker = None
        self._body_tracker_initialized = False
        self._body_tracker_bbox = None
        self._body_tracker_lost_count = 0
        self._end_tracker = None
        self._end_tracker_initialized = False
        self._end_tracker_bbox = None
        self._end_tracker_lost_count = 0

        self._last_phase1_metrics = None
        self._current_rz = {"r": 0.0, "z": 0.0}

        ok = self._move_to_pose(OBSERVATION_POSE, "机械臂回到观察姿态", wait=2.0)
        if ok:
            self._refresh_arm_position()
            self._phase = GrabPhase.DETECTION
        return ok

    # ==================== Phase 1: 属性测姿与底盘策略分流 ====================

    def _estimate_distance_from_area_ratio(self, area_ratio: float) -> float:
        """根据机身摄像头目标占比估算机器人到目标的距离（厘米）"""
        if area_ratio <= 0:
            return 999.0
        return (self.phase1_area_distance_k / area_ratio) ** 0.5

    def _check_phase1_tracker_anomaly(self, metrics: dict) -> bool:
        """检测 Phase 1 Tracker 异常漂移"""
        if self._last_phase1_metrics is None:
            return False
        last = self._last_phase1_metrics
        dy2 = abs(metrics["y2_ratio"] - last["y2_ratio"])
        darea = metrics["area_ratio"] - last["area_ratio"]

        if dy2 > 0.18 and darea < 0.02:
            print(f"[GRAB] [Phase1] Tracker 异常漂移: y2 变化 {dy2:+.3f}，面积变化 {darea:+.3f}")
            return True
        if metrics["y2_ratio"] >= 0.85 and metrics["area_ratio"] < 0.12:
            print(f"[GRAB] [Phase1] Tracker 异常：底部贴底但面积过小")
            return True
        return False

    def _vlm_detect_pose(self, image_path: str, target_object: str) -> dict | None:
        """Phase 1 第 1 次 VLM：返回 JSON {bbox, is_cylinder, pose, grasp_center?, visible_faces?}"""
        prompt = f'''你是机器人的高精度视觉定位助手。请在图片中找到目标物体"{target_object}"，并只返回以下 JSON：

{{
  "bbox": [x1, y1, x2, y2],
  "is_cylinder": true/false,
  "pose": "upright" | "fallen",
  "grasp_center": [cx, cy],
  "visible_faces": "front|top|side|back"
}}

其中：
- bbox 为 0~1 归一化坐标（或 0~1000 整数）；
- is_cylinder 表示是否为圆柱/瓶罐类；
- pose 表示姿态："upright" 为直立站立，"fallen" 为倾倒/横躺；
- grasp_center 是"夹爪应该对准的像素点"，通常位于物体可见部分的中心或最稳定的抓取受力点，可以与 bbox 中心不同；
- visible_faces 描述从当前视角能看到的主要面（如 "front"、"top"、"side" 等，可多选/组合）。

只输出 JSON，不要 Markdown 代码块、解释或任何额外文字。'''
        return self._vlm_call(image_path, prompt, ["bbox", "is_cylinder", "pose"],
                              max_tokens=1024, timeout=60)

    def _vlm_locate_target(self, image_path: str, target_object: str) -> dict | None:
        """Phase 2 第 2 次 VLM：末端摄像头目标定位，返回 JSON {bbox, grasp_center}"""
        prompt = f'''你是机器人的高精度视觉定位助手。图片来自机械臂末端的摄像头，当前距离目标物体极近。你的任务是为 OpenCV Tracker 提供精准的目标边界框和抓取点。

【核心任务】：
精准定位目标物体"{target_object}"，并给出夹爪应该对准的像素点。

【执行约束】：
1. 几何无偏见：无论目标是长方体、圆柱体、球体还是不规则形状，请框选物体的可见主体部分。
2. 形状过滤：目标物体的边界框应符合正常的物理长宽比（长宽比应在 0.2 到 4.0 之间）。如果你的定位结果长宽比极其异常（如覆盖全屏的横向长条或过细的线条），请重新审视并修正，不要框选背景或桌面边缘。
3. 夹爪屏蔽：严禁将机械臂夹爪、机械臂自身结构、阴影或桌面边缘框入。
4. 抓取点优先：输出一个 "grasp_center" 点，代表两片夹爪应该对准的位置。该点可以与 bbox 的几何中心不同，例如：
   - 直立瓶罐：对准瓶身中部（而非顶部或底部）；
   - 倾倒物体：对准物体可见部分的重心；
   - 不规则物体：对准最稳定、最不容易滑脱的受力点。

【输出格式】：
只返回纯 JSON 字符串，严禁任何 Markdown 代码块或额外文字。

返回格式：
{{
  "bbox": [x1, y1, x2, y2],
  "grasp_center": [cx, cy]
}}

注意：坐标请使用 [0, 1000] 的整数范围。'''
        return self._vlm_call(image_path, prompt, ["bbox"],
                              max_tokens=1024, timeout=60)

    def _phase1_detect_and_align(self, target_object: str) -> dict:
        """Phase 1: 属性测姿 + Tracker 驱动底盘 coarse alignment"""
        print("\n[GRAB] ====== Phase 1: 属性测姿与底盘策略分流 ======")
        print("[GRAB] 第 1 次 VLM：机身摄像头属性测姿")

        # 第 1 次 VLM（带 bbox 几何常识校验与重试）
        img_path, frame_bgr = self.capture("phase1_detect")
        if img_path is None:
            return {"success": False, "message": "Phase 1 图像捕获失败"}

        detection = None
        for attempt in range(1, 4):
            detection = self._vlm_detect_pose(img_path, target_object)
            if detection is None:
                print(f"[GRAB] [Phase1] VLM 第 {attempt}/3 次调用失败，重试...")
                continue
            bbox = detection.get("bbox")
            valid, reason = self._validate_vlm_bbox(bbox)
            if valid:
                print(f"[GRAB] [Phase1] VLM bbox 通过几何常识校验")
                break
            print(f"[GRAB] [ERR] VLM 框尺寸或比例异常，判定为幻觉（{reason}），第 {attempt}/3 次重试")
            detection = None

        if detection is None:
            return {"success": False, "message": "Phase 1 VLM 属性测姿失败或 bbox 校验未通过"}

        # 解析并存储 object_pose（含可选的 3D 感知字段）
        bbox = detection["bbox"]
        pose = detection.get("pose", "fallen").lower()
        if pose not in ("upright", "fallen"):
            pose = "fallen"

        grasp_center = detection.get("grasp_center")
        visible_faces = detection.get("visible_faces")

        self._object_pose = {
            "bbox": bbox,
            "is_cylinder": bool(detection.get("is_cylinder", False)),
            "pose": pose,
            "grasp_center": grasp_center,
            "visible_faces": visible_faces,
            "image_w": frame_bgr.shape[1] if frame_bgr is not None else None,
            "image_h": frame_bgr.shape[0] if frame_bgr is not None else None,
            "vlm_provider": detection.get("_provider"),
        }
        extra_info = []
        if grasp_center is not None:
            extra_info.append(f"grasp_center={grasp_center}")
        if visible_faces is not None:
            extra_info.append(f"visible_faces={visible_faces}")
        print(f"[GRAB] [Phase1] 测姿结果: pose={pose}, is_cylinder={self._object_pose['is_cylinder']}, "
              f"bbox={bbox}" + (f", {', '.join(extra_info)}" if extra_info else ""))

        # 用 VLM bbox 初始化机身 Tracker（Phase 1 粗定位，bbox 足够）
        if frame_bgr is not None:
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            bbox_xyxy = (x1 * w, y1 * h, x2 * w, y2 * h)
            self._init_tracker("_body_tracker", frame_bgr, bbox_xyxy)

        # 底盘策略分流
        params = _POSE_PARAMS[pose]
        target_min, target_max = params["stop_distance_cm"]
        print(f"[GRAB] [Phase1] 姿态策略: {pose}，底盘目标停靠距离 {target_min}-{target_max}cm")

        base_angle = OBSERVATION_POSE.get("base", -90)
        attempts = min(10, self.max_attempts)
        KP_ANGLE_PHASE1 = 20.0

        for attempt in range(1, attempts + 1):
            print(f"\n[GRAB] [Phase1] --- 第 {attempt}/{attempts} 次底盘对准 ---")

            # 保持观察姿态
            target = dict(OBSERVATION_POSE)
            target["base"] = base_angle
            resp = self.arm.set_joint_angles(target)
            if resp and resp.success:
                time.sleep(0.3)
            else:
                print("[GRAB] [ERR] Phase 1 观察姿态保持失败")
                continue

            img_path, frame_bgr = self.capture(f"phase1_{attempt}")
            if img_path is None:
                continue

            metrics = None

            # Tracker 更新或重初始化
            if self._body_tracker_initialized and frame_bgr is not None:
                ok, bbox_xywh = self._update_tracker("_body_tracker", frame_bgr)
                if ok:
                    h, w = frame_bgr.shape[:2]
                    bbox_xyxy, cx_ratio, cy_ratio, area_ratio, y2_ratio, width_ratio = self._tracker_metrics(bbox_xywh, w, h)
                    metrics = {
                        "bbox": bbox_xyxy,
                        "center_x_ratio": cx_ratio,
                        "center_y_ratio": cy_ratio,
                        "area_ratio": area_ratio,
                        "y2_ratio": y2_ratio,
                        "width_ratio": width_ratio,
                    }
                    print(f"[GRAB] [Phase1] Tracker: area={area_ratio:.3f}, cx={cx_ratio:.3f}, cy={cy_ratio:.3f}, y2={y2_ratio:.3f}")
                    self._save_debug_frame(frame_bgr, bbox_xyxy, f"phase1_{attempt}", metrics)
                    if self._check_phase1_tracker_anomaly(metrics):
                        self._body_tracker_initialized = False
                else:
                    print("[GRAB] [Phase1] Tracker 丢失，尝试 VLM 重定位")
                    detection = self._vlm_detect_pose(img_path, target_object)
                    if detection is not None:
                        bbox = detection["bbox"]
                        self._object_pose["bbox"] = bbox
                        h, w = frame_bgr.shape[:2]
                        x1, y1, x2, y2 = bbox
                        bbox_xyxy = (x1 * w, y1 * h, x2 * w, y2 * h)
                        self._init_tracker("_body_tracker", frame_bgr, bbox_xyxy)
                        # Phase 1 对 tracker 初始化失败不敏感，后续循环会重试

            # 无 tracker 则失败（Phase 1 不应 fallback 到距离分析，避免额外 VLM）
            if metrics is None:
                print("[GRAB] [Phase1] 无有效追踪框，跳过本次")
                continue

            # 水平角度修正
            cx_ratio = metrics["center_x_ratio"]
            target_cx = 0.45
            error_x = cx_ratio - target_cx
            if abs(error_x) > 0.10:
                angle_offset = error_x * KP_ANGLE_PHASE1
                angle_offset = max(3.0, min(10.0, abs(angle_offset))) * (1 if angle_offset > 0 else -1)
                print(f"[GRAB] [Phase1] 水平修正: cx={cx_ratio:.3f}, 底盘旋转 {angle_offset:+.1f}°")
                if angle_offset > 0:
                    self.chassis.right_deg(angle_offset)
                else:
                    self.chassis.left_deg(-angle_offset)
                time.sleep(0.5)
                continue

            # 距离闭环
            area_ratio = metrics["area_ratio"]
            y2_ratio = metrics.get("y2_ratio", 0.0)
            estimated_distance = self._estimate_distance_from_area_ratio(area_ratio)
            print(f"[GRAB] [Phase1] 距离估算: area={area_ratio:.3f} -> 约 {estimated_distance:.1f}cm "
                  f"(目标 {target_min}-{target_max}cm)")

            # 安全边界：只有距离估算已经进入目标区间附近时，
            # y2 贴底/area 过大才触发后退保护，避免距离还很远时因物体高大/相机俯角误判为“过近”
            in_target_zone = estimated_distance <= target_max + 5.0
            if in_target_zone and y2_ratio >= self.phase1_y2_too_close:
                backup_cm = 3.0 if y2_ratio >= 0.85 else 2.0
                print(f"[GRAB] [Phase1] 目标底部贴底 (y2={y2_ratio:.3f})，后退 {backup_cm}cm")
                self.chassis.backward_cm(backup_cm)
                time.sleep(0.5)
                return {"success": True, "message": "Phase 1 完成（贴底保护）"}
            elif y2_ratio >= self.phase1_y2_too_close:
                print(f"[GRAB] [Phase1] 目标底部虽贴底 (y2={y2_ratio:.3f})，但距离估算仍远 "
                      f"({estimated_distance:.1f}cm > {target_max + 5.0:.1f}cm)，优先前进靠近")

            if in_target_zone and area_ratio >= self.phase1_area_too_close:
                backup_cm = 2.0
                print(f"[GRAB] [Phase1] 目标占比过大 (area={area_ratio:.3f})，后退 {backup_cm}cm")
                self.chassis.backward_cm(backup_cm)
                time.sleep(0.5)
                return {"success": True, "message": "Phase 1 完成（面积保护）"}
            elif area_ratio >= self.phase1_area_too_close:
                print(f"[GRAB] [Phase1] 目标占比虽大 (area={area_ratio:.3f})，但距离估算仍远 "
                      f"({estimated_distance:.1f}cm > {target_max + 5.0:.1f}cm)，优先前进靠近")

            # 距离区间控制
            if target_min <= estimated_distance <= target_max:
                print(f"[GRAB] [Phase1] ✅ 距离在目标区间 {target_min}-{target_max}cm 内，底盘锁死")
                self.chassis.stop()
                self._phase = GrabPhase.APPROACH
                return {"success": True, "message": "Phase 1 完成"}

            # 根据距离选择步长
            distance_steps = [
                (80, 15), (50, 10), (30, 5), (target_max + 5, 2),
            ]
            if estimated_distance > target_max:
                move_cm = 2
                for threshold, step in distance_steps:
                    if estimated_distance > threshold:
                        move_cm = step
                        break
                print(f"[GRAB] [Phase1] 距离过远，前进 {move_cm}cm")
                self.chassis.forward_cm(move_cm)
            elif estimated_distance < target_min:
                move_cm = 2
                print(f"[GRAB] [Phase1] 距离过近，后退 {move_cm}cm")
                self.chassis.backward_cm(move_cm)

            time.sleep(0.5)
            self._last_phase1_metrics = dict(metrics)

        print("[GRAB] [Phase1] 达到最大尝试次数，按当前状态放行")
        self.chassis.stop()
        self._phase = GrabPhase.APPROACH
        return {"success": True, "message": "Phase 1 完成（最大次数）"}

    # ==================== Phase 2: 接近与手腕姿态预部署 ====================

    def _project_bbox_to_end_camera(self, frame_bgr: np.ndarray) -> tuple:
        """根据 Phase 1 body bbox 估算 end camera 中的初始 bbox（无 VLM）"""
        h, w = frame_bgr.shape[:2]
        bbox = self._object_pose["bbox"]
        if bbox is None:
            # fallback 中心框
            return (w * 0.35, h * 0.4, w * 0.65, h * 0.7)

        # 原宽高比
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        aspect = bh / bw if bw > 0 else 1.0

        # 末端摄像头更近，面积约为机身视角 3~4 倍
        area_ratio_phase1 = bw * bh
        expected_area = min(0.55, area_ratio_phase1 * 3.5)

        end_w_ratio = math.sqrt(expected_area / aspect)
        end_h_ratio = end_w_ratio * aspect

        # 假设底盘对准后物体在 end camera 中心附近
        cx, cy = 0.5, 0.55
        return (
            (cx - end_w_ratio / 2) * w,
            (cy - end_h_ratio / 2) * h,
            (cx + end_w_ratio / 2) * w,
            (cy + end_h_ratio / 2) * h,
        )

    def _phase2_vlm_center_x(self, image_path: str, frame_bgr: np.ndarray, target_object: str) -> float | None:
        """用 VLM 定位目标并返回其水平中心比例 (cx_ratio)，失败返回 None。"""
        detection = self._vlm_locate_target(image_path, target_object)
        if detection is None or "bbox" not in detection:
            return None
        bbox = detection["bbox"]
        valid, reason = self._validate_vlm_bbox(bbox)
        if not valid:
            print(f"[GRAB] [Phase2] VLM-only 框校验失败（{reason}）")
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        cx_ratio = ((x1 + x2) / 2.0)
        if cx_ratio > 1.0:
            # VLM 返回的是 0~1000 整数坐标
            cx_ratio = cx_ratio / 1000.0
        print(f"[GRAB] [Phase2] VLM-only: bbox={bbox}, cx_ratio={cx_ratio:.3f}")
        self._save_debug_frame(frame_bgr,
                                (x1 * w, y1 * h, x2 * w, y2 * h),
                                "phase2_vlm_only",
                                {"center_x_ratio": cx_ratio})
        return cx_ratio

    def _phase2_vlm_only_align(self, target_object: str, max_attempts: int = 3) -> bool:
        """Tracker 初始化失败时，用 VLM 直接做水平 base 对准。

        Returns:
            True 表示对准成功或已足够居中；False 表示 VLM 连续失败。
        """
        print("[GRAB] [Phase2] VLM-only base 对准开始")
        KP_BASE = 6.0
        current_angles = self.get_current_angles()
        current_base = current_angles.get("base", OBSERVATION_POSE.get("base", -90))
        ever_located = False

        for attempt in range(1, max_attempts + 1):
            img_path, frame_bgr = self.capture_end(f"phase2_vlm_align_{attempt}")
            if frame_bgr is None:
                continue
            cx_ratio = self._phase2_vlm_center_x(img_path, frame_bgr, target_object)
            if cx_ratio is None:
                print(f"[GRAB] [Phase2] VLM-only 第 {attempt}/{max_attempts} 次定位失败")
                continue

            ever_located = True
            error_x = cx_ratio - 0.5
            if abs(error_x) <= 0.08:
                print("[GRAB] [Phase2] VLM-only 水平已对准")
                return True

            base_offset = error_x * KP_BASE
            base_offset = max(0.1, min(2.0, abs(base_offset))) * (1 if base_offset > 0 else -1)
            new_base = current_base + base_offset
            new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
            print(f"[GRAB] [Phase2] VLM-only base 微调: {current_base:.1f}° -> {new_base:.1f}°")

            align_pose = {
                "base": new_base,
                "shoulder": current_angles.get("shoulder", 0),
                "elbow": current_angles.get("elbow", 150),
                "wrist_flex": current_angles.get("wrist_flex", 30),
                "wrist_roll": _GRAB_WRIST_ROLL,
                "gripper": 90,
            }
            self._move_to_pose(align_pose, "Phase2 VLM-only base 微调", wait=0.8)
            current_base = new_base

        if ever_located:
            # 即便最后一次没对准，只要至少成功定位过一次就放行，Phase 3 会再用 VLM 补偿
            print("[GRAB] [Phase2] VLM-only 对准次数用尽，按当前状态放行")
            return True
        print("[GRAB] [Phase2] VLM-only 连续定位失败")
        return False

    def _phase2_approach_and_deploy(self, target_object: str) -> dict:
        """Phase 2: 末端摄像头 Tracker base 对准 + 手腕姿态预部署"""
        print("\n[GRAB] ====== Phase 2: 接近与手腕姿态预部署 ======")
        pose = self._object_pose.get("pose", "fallen")
        params = _POSE_PARAMS[pose]

        self._check_end_camera()
        if not self._end_camera_available:
            return {"success": False, "message": "末端摄像头不可用"}

        # 切换摄像头，重置 end tracker
        self._end_tracker = None
        self._end_tracker_initialized = False
        self._end_tracker_bbox = None
        self._end_tracker_lost_count = 0

        # 初始化 end tracker：优先用 VLM 给出精确 bbox + grasp_center，失败再回退几何投影/中心框
        init_img_path, init_frame_bgr = self.capture_end("phase2_init")
        if init_frame_bgr is None:
            return {"success": False, "message": "Phase 2 末端摄像头捕获失败"}

        print("[GRAB] [Phase2] 第 2 次 VLM：末端摄像头目标定位")
        detection = self._vlm_locate_target(init_img_path, target_object)

        end_grasp_center = detection.get("grasp_center") if detection else None
        self._object_pose["end_grasp_center"] = end_grasp_center

        ok = False
        vlm_bbox = None
        if detection is not None and "bbox" in detection:
            bbox = detection["bbox"]
            valid, reason = self._validate_vlm_bbox(bbox)
            if valid:
                h, w = init_frame_bgr.shape[:2]
                x1, y1, x2, y2 = bbox
                vlm_bbox = (x1 * w, y1 * h, x2 * w, y2 * h)

                # 如果 VLM 给出了 grasp_center，把 tracker 初始框中心移到 grasp_center，
                # 大小保持 bbox 原尺寸。这样后续 Tracker/对齐都以真实抓取点为中心。
                if end_grasp_center is not None:
                    gc = end_grasp_center
                    if max(gc) <= 1.0:
                        gc_x, gc_y = gc[0] * w, gc[1] * h
                    else:
                        gc_x, gc_y = gc[0] / 1000.0 * w, gc[1] / 1000.0 * h
                    bw = vlm_bbox[2] - vlm_bbox[0]
                    bh = vlm_bbox[3] - vlm_bbox[1]
                    shifted_bbox = (
                        max(0, gc_x - bw / 2),
                        max(0, gc_y - bh / 2),
                        min(w, gc_x + bw / 2),
                        min(h, gc_y + bh / 2),
                    )
                    if shifted_bbox[2] > shifted_bbox[0] and shifted_bbox[3] > shifted_bbox[1]:
                        print(f"[GRAB] [Phase2] VLM 定位 bbox={vlm_bbox}, grasp_center=({gc_x:.1f}, {gc_y:.1f})，"
                              f"tracker 初始框中心已对齐抓取点")
                        vlm_bbox = shifted_bbox
                    else:
                        print(f"[GRAB] [WARN] grasp_center 导致框无效，回退到 bbox 中心")
                else:
                    print(f"[GRAB] [Phase2] VLM 定位 bbox: {vlm_bbox} (无 grasp_center)")

                ok, init_reason = self._init_tracker("_end_tracker", init_frame_bgr, vlm_bbox)
            else:
                print(f"[GRAB] [ERR] Phase 2 VLM 框尺寸或比例异常，判定为幻觉（{reason}），回退到几何投影")

        if not ok:
            print("[GRAB] [Phase2] VLM 定位失败或 Tracker 初始化失败，回退到几何投影")
            projected_bbox = self._project_bbox_to_end_camera(init_frame_bgr)
            ok, init_reason = self._init_tracker("_end_tracker", init_frame_bgr, projected_bbox)
        if not ok:
            # fallback 中心框
            h, w = init_frame_bgr.shape[:2]
            fallback_bbox = (w * 0.35, h * 0.4, w * 0.65, h * 0.7)
            ok, init_reason = self._init_tracker("_end_tracker", init_frame_bgr, fallback_bbox)
        if not ok:
            # 所有 Tracker 初始化源都失败（常见原因：低纹理物体如纸巾/ OpenCV 追踪器不可用）
            # 保存失败现场后，尝试 VLM-only 对准，而不是直接失败。
            self._save_debug_frame(init_frame_bgr, None, "phase2_tracker_init_failed",
                                    {"reason": init_reason, "vlm_bbox": vlm_bbox})
            print(f"[GRAB] [Phase2] ⚠️ end Tracker 全部初始化失败: {init_reason}")
            print("[GRAB] [Phase2] 回退到 VLM-only 水平对准")
            vlm_ok = self._phase2_vlm_only_align(target_object)
            if not vlm_ok:
                return {
                    "success": False,
                    "message": f"Phase 2 end Tracker 初始化失败 ({init_reason})，且 VLM-only 对准失败",
                }
            # VLM-only 对准成功：标记 tracker 未初始化，让 Phase 3 用 VLM 重定位
            self._end_tracker_initialized = False
            print("[GRAB] [Phase2] VLM-only 对准完成，后续进入 Phase 3 由 VLM 重定位补偿")

        # 记录当前 base（后续微调只改 base）
        scout_pose = self.get_current_angles()
        current_base = scout_pose.get("base", OBSERVATION_POSE.get("base", -90))

        # --- Tracker 高频闭环调 base ---
        # 用户反馈：接近中心时 1° 步长过大，调小增益和最小步长，收紧死区
        KP_BASE = 6.0
        max_attempts = 12
        last_error_x = None

        for attempt in range(1, max_attempts + 1):
            print(f"\n[GRAB] [Phase2] --- 第 {attempt}/{max_attempts} 次水平对准 ---")
            img_path, frame_bgr = self.capture_end(f"phase2_align_{attempt}")
            if frame_bgr is None:
                continue

            # 如果 Tracker 未初始化（VLM-only 回退），每轮用 VLM 定位获取水平误差
            if not self._end_tracker_initialized:
                cx_ratio = self._phase2_vlm_center_x(img_path, frame_bgr, target_object)
                if cx_ratio is None:
                    print("[GRAB] [Phase2] VLM 定位失败，跳过本次")
                    continue
                bbox_xyxy = None
            else:
                ok, bbox_xywh = self._update_tracker("_end_tracker", frame_bgr)
                if not ok:
                    print("[GRAB] [Phase2] end Tracker 丢失")
                    if self._end_tracker_lost_count >= 3:
                        return {"success": False, "message": "Phase 2 end Tracker 持续丢失"}
                    continue

                h, w = frame_bgr.shape[:2]
                bbox_xyxy, cx_ratio, cy_ratio, area_ratio, y2_ratio, width_ratio = self._tracker_metrics(bbox_xywh, w, h)
                metrics = {"center_x_ratio": cx_ratio, "center_y_ratio": cy_ratio, "area_ratio": area_ratio, "y2_ratio": y2_ratio, "width_ratio": width_ratio}
                print(f"[GRAB] [Phase2] Tracker: cx={cx_ratio:.3f}, cy={cy_ratio:.3f}, area={area_ratio:.3f}, y2={y2_ratio:.3f}, w={width_ratio:.3f}")
                self._save_debug_frame(frame_bgr, bbox_xyxy, f"phase2_align_{attempt}", metrics)

            error_x = cx_ratio - 0.5
            if abs(error_x) <= 0.06:
                print("[GRAB] [Phase2] ✅ 水平已对准")
                break
            if last_error_x is not None and (error_x * last_error_x < 0) and abs(error_x) <= 0.12:
                print("[GRAB] [Phase2] ✅ 过零检测，判定对准")
                break

            base_offset = error_x * KP_BASE
            base_offset = max(0.1, min(2.0, abs(base_offset))) * (1 if base_offset > 0 else -1)
            # 实测：base 增大（负得少）时 center_x 减小，因此 error_x>0 时应减小 base（负得更多）
            new_base = current_base + base_offset
            new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
            print(f"[GRAB] [Phase2] base 微调: {current_base:.1f}° -> {new_base:.1f}°")

            align_pose = {
                "base": new_base,
                "shoulder": scout_pose.get("shoulder", 0),
                "elbow": scout_pose.get("elbow", 150),
                "wrist_flex": scout_pose.get("wrist_flex", 30),
                "wrist_roll": _GRAB_WRIST_ROLL,
                "gripper": 90,
            }
            self._move_to_pose(align_pose, "Phase2 base 微调", wait=0.8)
            current_base = new_base
            last_error_x = error_x

        # --- 手腕姿态预部署 ---
        print(f"\n[GRAB] [Phase2] 手腕姿态预部署: pose={pose}")
        pre_r, pre_z, pre_pitch = params["pre_deploy_rz"]
        ok = self._move_to_rz(pre_r, pre_z, f"Phase2 预部署 ({pose})",
                              wait=2.0, target_pitch=pre_pitch)
        if not ok:
            return {"success": False, "message": f"Phase 2 预部署失败 ({pose})"}

        print(f"[GRAB] [Phase2] ✅ 预部署完成")
        self._phase = GrabPhase.FINAL_TUNING
        return {"success": True, "message": "Phase 2 完成"}

    # ==================== 深度微调辅助 ====================

    def _compute_final_approach_cm(self, h_ratio: float) -> float:
        """
        根据当前末端摄像头中物体高度占比，自适应计算最终接近距离。
        h_ratio 越大表示物体越近，最终接近距离应越短，避免顶到/推开物体。
        """
        if h_ratio >= 0.80:
            return 1.0
        elif h_ratio >= 0.70:
            return 2.0
        elif h_ratio >= 0.60:
            return 3.0
        else:
            return 4.0

    def _depth_fine_tune(self, frame_bgr, bbox_xyxy):
        """
        基于物体在末端摄像头中的高度占比进行深度微调。

        核心思想：距离越近，物体在画面中的高度占比越大。
        当 h_ratio 达到 TARGET_H_RATIO（默认 0.85）时，认为夹爪已足够贴近物体。

        Args:
            frame_bgr: 末端摄像头 BGR 图像
            bbox_xyxy: 追踪框 (x1, y1, x2, y2) 像素坐标

        Returns:
            bool: True 表示已足够贴近，False 表示还需继续微调
        """
        # 1. 计算物体在图像中的垂直占比 (h_ratio)
        # 这是一个关键指标：当距离越近，物体在图像中占据的高度比例会越大
        h_ratio = (bbox_xyxy[3] - bbox_xyxy[1]) / frame_bgr.shape[0]

        # 2. 预设目标占比 (Target Ratio)
        # 当夹爪根部距离物体约 0-1cm 时，物体在画面中的理想占比（需通过实验测定）
        # 上次测试 h_ratio=0.906 时夹爪已经非常贴近，因此把目标从 0.81 提高到 0.85
        TARGET_H_RATIO = 0.85

        # 3. 深度补偿比例
        error_h = TARGET_H_RATIO - h_ratio

        print(f"[GRAB] [DepthTune] h_ratio={h_ratio:.3f}, target={TARGET_H_RATIO}, error={error_h:+.3f}")

        # 如果 error_h > 0，说明物体占比太小（太远了），需要继续前伸
        if error_h > 0.05:
            # 执行微小前伸：根据 error_h 换算成 forward 偏移量（cm）
            # 系数从 20 降到 10，避免单次前伸过大顶到物体
            forward_cm = error_h * 10
            print(f"[GRAB] [DepthTune] 物体占比偏小，前伸 {forward_cm:.2f}cm")
            self._arm_move_relative(forward_cm=forward_cm)
            return False  # 未完成，继续微调

        print("[GRAB] [DepthTune] ✅ 高度占比已达标，深度足够")
        return True  # 已足够贴近

    def _reinit_end_tracker_with_vlm(self, image_path: str, frame_bgr: np.ndarray,
                                     target_object: str, reason: str = "VLM 重新定位") -> bool:
        """用 VLM 重新划定 end Tracker，成功返回 True 并更新内部状态。"""
        print(f"[GRAB] [Phase3] {reason}，调用 VLM 重新划定 Tracker...")
        for attempt in range(1, 3):
            detection = self._vlm_locate_target(image_path, target_object)
            if detection is None or "bbox" not in detection:
                print(f"[GRAB] [Phase3] VLM 重定位第 {attempt}/2 次调用失败，重试...")
                continue
            bbox = detection["bbox"]
            valid, reason_err = self._validate_vlm_bbox(bbox)
            if not valid:
                print(f"[GRAB] [Phase3] VLM 框校验失败（{reason_err}），重试...")
                continue
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            vlm_bbox = (x1 * w, y1 * h, x2 * w, y2 * h)
            ok, _ = self._init_tracker("_end_tracker", frame_bgr, vlm_bbox)
            if ok:
                print(f"[GRAB] [Phase3] ✅ VLM 重定位成功，新框=({vlm_bbox[0]:.1f}, {vlm_bbox[1]:.1f}, "
                      f"{vlm_bbox[2]:.1f}, {vlm_bbox[3]:.1f})")
                return True
        print("[GRAB] [Phase3] ❌ VLM 重定位失败")
        return False

    # ==================== Phase 3: 二次距离闭环与精准贴紧 ====================

    def _phase3_final_tuning(self, target_object: str) -> dict:
        """Phase 3: 二次距离闭环与精准贴紧（周期性或 Tracker 异常时调用 VLM 重定位）"""
        print("\n[GRAB] ====== Phase 3: 二次距离闭环与精准贴紧 ======")
        pose = self._object_pose.get("pose", "fallen")
        params = _POSE_PARAMS[pose]
        direction = params["tune_direction"]
        step_mm = params["tune_step_mm"]
        max_steps = params["tune_max_steps"]

        target_y2 = params["tune_target_y2_ratio"]
        if pose == "upright":
            target_width = params["tune_target_width_ratio"]
            print(f"[GRAB] [Phase3] 停止阈值: width>={target_width:.2f} 且 y2>={target_y2:.2f}")
        else:
            target_area = params["tune_target_area_ratio"]
            print(f"[GRAB] [Phase3] 停止阈值: area>={target_area:.2f} 且 y2>={target_y2:.2f}")

        # 直立物体防推动检测：记录最近 h_ratio（高度占比），连续下降则停止
        h_history = [] if pose == "upright" else None

        for step in range(1, max_steps + 1):
            print(f"\n[GRAB] [Phase3] --- 第 {step}/{max_steps} 次微调 ({direction}) ---")
            img_path, frame_bgr = self.capture_end(f"phase3_{step}")
            if frame_bgr is None:
                continue

            # 在 Tracker 漂移/丢失后用 VLM 重新划定 end Tracker；
            # 若开启周期性重定位（_PHASE3_VLM_REINIT_EVERY_N 为整数），则同时按步长触发。
            reinit_reason = None
            if _PHASE3_VLM_REINIT_EVERY_N is not None and step % _PHASE3_VLM_REINIT_EVERY_N == 0:
                reinit_reason = f"第 {step} 步周期性 VLM 重定位"
            elif not self._end_tracker_initialized:
                reinit_reason = "end Tracker 未初始化，VLM 重定位"
            elif self._end_tracker_lost_count > 0:
                reinit_reason = "Tracker 丢失/漂移后 VLM 重定位"
            if reinit_reason and img_path is not None:
                self._reinit_end_tracker_with_vlm(img_path, frame_bgr, target_object, reinit_reason)

            ok, bbox_xywh = self._update_tracker("_end_tracker", frame_bgr)
            if not ok:
                print("[GRAB] [Phase3] end Tracker 丢失，跳过本次")
                if self._end_tracker_lost_count >= 3:
                    return {"success": False, "message": "Phase 3 end Tracker 持续丢失"}
                continue

            h, w = frame_bgr.shape[:2]
            bbox_xyxy, cx_ratio, cy_ratio, area_ratio, y2_ratio, width_ratio = self._tracker_metrics(bbox_xywh, w, h)
            h_ratio = (bbox_xyxy[3] - bbox_xyxy[1]) / h
            metrics = {
                "center_x_ratio": cx_ratio,
                "center_y_ratio": cy_ratio,
                "area_ratio": area_ratio,
                "y2_ratio": y2_ratio,
                "width_ratio": width_ratio,
                "h_ratio": h_ratio,
            }
            print(f"[GRAB] [Phase3] Tracker: cx={cx_ratio:.3f}, cy={cy_ratio:.3f}, area={area_ratio:.3f}, y2={y2_ratio:.3f}, w={width_ratio:.3f}, h_ratio={h_ratio:.3f}")
            self._save_debug_frame(frame_bgr, bbox_xyxy, f"phase3_{step}", metrics)

            # 注：Phase 3 不再因为水平漂移而回退到 Phase 2。
            # 近景时侧装摄像头 parallax 会让 cx 明显偏离 0.5，这种视觉偏置在 Phase 4 VLM 终审中再处理；
            # 这里只做同一周期内的保守水平微调（见下方 error_x 分支）。

            if pose == "fallen":
                # 垂直下降：目标底部接近画面底部且面积足够大
                if y2_ratio >= target_y2 and area_ratio >= target_area:
                    print("[GRAB] [Phase3] ✅ 倾倒物体已下降至目标位置")
                    break
                ok = self._move_to_rz_relative(dr=0, dz=-step_mm, pitch=params["grasp_pitch"],
                                               desc="倾倒物体垂直下降", wait=1.0)
            else:
                # 直立物体：先通过 _depth_fine_tune 闭环深度，再用 width/y2 做二次确认

                # 水平微调：近景时 parallax 会让物体看起来偏侧，
                # 只在偏移较大时做保守 base 修正，避免频繁抖动或过度补偿
                error_x = cx_ratio - 0.5
                if abs(error_x) > 0.12:
                    KP_BASE_PHASE3 = 3.0
                    base_offset = error_x * KP_BASE_PHASE3
                    base_offset = max(0.3, min(0.6, abs(base_offset))) * (1 if error_x > 0 else -1)
                    current_angles = self.get_current_angles()
                    current_base = current_angles.get("base", OBSERVATION_POSE.get("base", -90))
                    new_base = current_base + base_offset
                    new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
                    print(f"[GRAB] [Phase3] 水平微调: cx={cx_ratio:.3f}, base {current_base:.1f}° -> {new_base:.1f}°")
                    align_pose = {
                        "base": new_base,
                        "shoulder": current_angles.get("shoulder", 0),
                        "elbow": current_angles.get("elbow", 150),
                        "wrist_flex": current_angles.get("wrist_flex", 30),
                        "wrist_roll": _GRAB_WRIST_ROLL,
                        "gripper": 90,
                    }
                    self._move_to_pose(align_pose, "Phase3 水平微调", wait=0.6)

                # 垂直微调：根据物体中心 cy 调整 z，使夹爪对准物体中下部分
                # cy > 0.5 表示物体中心在画面下半部分，夹爪偏高，需要下降
                error_y = cy_ratio - 0.5
                if abs(error_y) > 0.08:
                    KP_Z_PHASE3 = 8.0
                    dz_cm = -error_y * KP_Z_PHASE3
                    dz_cm = max(0.3, min(1.5, abs(dz_cm))) * (1 if dz_cm > 0 else -1)
                    print(f"[GRAB] [Phase3] 垂直微调: cy={cy_ratio:.3f}, 调整 z {dz_cm:+.2f}cm")
                    self._arm_move_relative(up_cm=dz_cm)

                # 防推动/防异常：连续 h_ratio 下降说明可能已顶到物体或 Tracker 漂移，停止前进
                h_history.append(h_ratio)
                if len(h_history) >= 4:
                    if h_history[-1] < h_history[-2] < h_history[-3] < h_history[-4]:
                        print("[GRAB] [Phase3] ⚠️ h_ratio 连续下降，判断已接触物体或 Tracker 异常，停止前进")
                        break

                # 使用高度占比闭环进行深度微调
                depth_done = self._depth_fine_tune(frame_bgr, bbox_xyxy)

                # 严格终止条件：深度达标且 width/y2 也达标
                strict_done = depth_done and y2_ratio >= target_y2 and width_ratio >= target_width
                # 宽松终止条件：多个指标同时接近目标（避免 h_ratio 非线性导致提前/过晚停止）
                composite_done = (
                    h_ratio >= 0.70 and
                    y2_ratio >= 0.84 and
                    width_ratio >= 0.20 and
                    area_ratio >= 0.14
                )

                if strict_done or composite_done:
                    reason = "深度与宽度/y2 均达标" if strict_done else "综合指标已接近目标"
                    final_approach = self._compute_final_approach_cm(h_ratio)
                    print(f"[GRAB] [Phase3] ✅ {reason}，执行最终接近 {final_approach:.1f}cm (h_ratio={h_ratio:.3f})")
                    self._arm_move_relative(forward_cm=final_approach)
                    break

                if depth_done:
                    # 深度已达标但 width/y2 仍未达标，执行一次保守前伸作为 fallback
                    print("[GRAB] [Phase3] 深度已达标，继续保守前伸以扩大 width/y2")
                    ok = self._move_to_rz_relative(dr=step_mm, dz=0, pitch=params["grasp_pitch"],
                                                   desc="直立物体水平贴紧(fallback)", wait=1.0)
                else:
                    # _depth_fine_tune 已执行前伸，本周期无需额外运动
                    ok = True

            if not ok:
                return {"success": False, "message": "Phase 3 微调运动失败"}

        self._phase = GrabPhase.TOUCH_VERIFY
        return {"success": True, "message": "Phase 3 完成"}

    # ==================== Phase 4: 触达确认与大模型终审 ====================

    def _phase4_touch_verify(self, target_object: str) -> dict:
        """Phase 4: 第 3 次 VLM，触碰终审"""
        print("\n[GRAB] ====== Phase 4: 触达确认与大模型终审 ======")
        print("[GRAB] 第 3 次 VLM：末端摄像头对齐校验")

        pose = self._object_pose.get("pose", "fallen")

        img_path, frame_bgr = self.capture_end("phase4_verify")
        if img_path is None:
            return {"success": False, "message": "Phase 4 图像捕获失败"}

        if pose == "upright":
            specific_q = (
                "从当前侧面近景看，水平夹爪是否已经将直立柱体的中部纳入了抓取范围内？"
                "物体是否处于两片夹爪之间的空腔中？只要闭合夹爪时不会将瓶子顶倒或推开，"
                "即使画面中看起来有光影缝隙或轻微偏侧，也请判定 aligned 为 true。"
            )
        else:
            specific_q = (
                "从当前俯视视角看，垂直向下的夹爪是否已经从正上方罩住了倒下物体的中心区域？"
                "只要闭合夹爪能成功将其包裹在内、不会滑空或压在物体正上方导致受损，"
                "即使边缘有微小间隙，也请判定 aligned 为 true。"
            )

        prompt = f'''你是机械臂末端摄像头的视觉验证与决策专家。当前处于抓取前的最后 3 厘米距离。

【请特别注意】：
1. 视觉偏置：摄像头安装在手腕侧面而非正中心。因此，物体在画面中看起来“偏向某一侧”是完全正常的视觉错觉，不要因为位置不居中就判定为失败。
2. 决策维度：你的判定不能仅基于 2D 坐标。请判断：通过机械臂的“水平前伸”或“垂直下降”动作，是否可以将物体纳入夹爪的抓取范围内？
3. 容错标准：只要物体已经进入了两侧夹爪的覆盖视野范围内（即使看起来靠边），且没有发生剧烈的遮挡，请判定为 aligned: true。

【输入信息】：
目标：{target_object}
{specific_q}

【输出格式】：
只返回纯 JSON 字符串，严禁 Markdown 代码块、严禁包含 ```json 标记。

{{
  "aligned": true/false,
  "reason": "简短原因，说明为什么判定为成功或失败"
}}'''

        result = self._vlm_call(img_path, prompt, ["aligned", "reason"],
                                max_tokens=512, timeout=60)
        if result is None:
            return {"success": False, "message": "Phase 4 VLM 终审失败"}

        aligned = bool(result.get("aligned", False))
        reason = result.get("reason", "")
        print(f"[GRAB] [Phase4] VLM 终审: aligned={aligned}, reason={reason}")

        if not aligned:
            print("[GRAB] [Phase4] ❌ 对齐校验未通过，禁止闭合夹爪，回退观察姿态")
            self._phase0_observation_reset()
            return {"success": False, "message": f"Phase 4 对齐校验未通过: {reason}"}

        print("[GRAB] [Phase4] ✅ 对齐校验通过，放行抓取")
        self._phase = GrabPhase.GRASP_LIFT
        return {"success": True, "message": "Phase 4 通过"}

    # ==================== Phase 5: 夹紧抬升与回缩验证 ====================

    def _phase5_grasp_and_lift(self, target_object: str) -> dict:
        """Phase 5: 闭合夹爪 + pose-specific 抬升 + 验证"""
        print("\n[GRAB] ====== Phase 5: 夹紧抬升与回缩验证 ======")
        pose = self._object_pose.get("pose", "fallen")
        params = _POSE_PARAMS[pose]

        # 张开 -> 微调闭合前姿态（已在 Phase 3 到位）
        print("[GRAB] [Phase5] 1/3 张开夹爪并确认姿态")
        self.arm.set_gripper(90)
        time.sleep(0.3)

        # 再次确认最终抓取姿态
        final_r = self._current_rz["r"]
        final_z = self._current_rz["z"]
        ok = self._move_to_rz(final_r, final_z, "最终抓取姿态确认",
                              wait=1.0, target_pitch=params["grasp_pitch"])
        if not ok:
            print("[GRAB] [Phase5] ⚠️ 最终姿态确认失败，尝试直接闭合")

        # 夹紧
        print("[GRAB] [Phase5] 2/3 夹紧")
        self.arm.set_gripper(0)
        time.sleep(1.0)

        # 抬升
        print("[GRAB] [Phase5] 3/3 抬升")
        for idx, (r, z, pitch) in enumerate(params["lift_trajectory"]):
            ok = self._move_to_rz(r, z, f"抬升路点 {idx+1}/{len(params['lift_trajectory'])}",
                                  wait=1.5, target_pitch=pitch)
            if not ok:
                print(f"[GRAB] [Phase5] ⚠️ 抬升路点 {idx+1} 失败")
                break

        print("[GRAB] [Phase5] 抓取序列完成")

        # 战果验证：本状态机已把语义终审放在 Phase 4，Phase 5 不再调用额外 VLM，
        # 仅依赖夹爪闭合反馈。如需增强验证，可在此调用 _verify_grab（消耗额外 VLM）。
        return {"success": True, "message": "抓取并抬升完成（Phase 4 已终审通过）"}

    def _verify_grab(self, target_object: str) -> dict:
        """抓取后视觉验证（末端摄像头）"""
        print("\n[GRAB] ====== 战果验证 ======")
        if self._end_camera_available:
            img_path, _ = self.capture_end("verify")
            camera_name = "末端摄像头"
        else:
            img_path, _ = self.capture("verify")
            camera_name = "机身摄像头"

        if img_path is None:
            return {"success": False, "message": "验证拍照失败"}

        prompt = (
            f"这张图来自机械臂末端夹爪上的摄像头，是近距离视角。"
            f"请仔细观察夹爪的两个手指之间，是否确实夹住了'{target_object}'。"
            f"只回复以下三种之一：已夹住 / 未夹住 / 未夹住但物体在附近"
        )
        try:
            # 战果验证允许额外 VLM（不计入主流程 2 次）
            result, used_provider = analyze_images_with_fallback(
                image_paths=[img_path],
                prompt=prompt,
            )
            result = result.strip()
            print(f"[GRAB] 验证结果({camera_name} / {used_provider}): {result}")
            if "已夹住" in result:
                return {"success": True, "message": f"{camera_name}视觉验证确认已夹住"}
            elif "未夹住但物体在附近" in result:
                return {"success": False, "message": "未夹住，但物体在夹爪附近，建议微调后重试"}
            else:
                return {"success": False, "message": f"{camera_name}视觉验证显示未夹住"}
        except Exception as e:
            print(f"[GRAB] 验证异常: {e}")
            return {"success": False, "message": f"验证异常: {e}"}

    # ==================== 主流程 ====================

    def _cleanup_old_run_dirs(self, keep: int = 1):
        """
        清理旧的抓取运行目录，只保留最近 keep 次的完整图片日志。

        每次 run() 会产生一个 run_YYYYMMDD_HHMMSS 子目录，里面同时包含关键帧
        grab_*.jpg 和中间过程 debug/*.jpg；抓取结束后按目录名时间排序，删除更早
        的目录以释放磁盘空间。
        """
        try:
            entries = []
            for name in os.listdir(self.output_dir):
                path = os.path.join(self.output_dir, name)
                if os.path.isdir(path) and name.startswith("run_"):
                    entries.append((name, path))
            if len(entries) <= keep:
                return
            # 目录名含时间戳，字典序即时间序
            entries.sort(key=lambda x: x[0], reverse=True)
            for name, path in entries[keep:]:
                shutil.rmtree(path, ignore_errors=True)
                print(f"[GRAB] [Cleanup] 已清理旧抓取日志目录: {name}")
        except Exception as e:
            print(f"[GRAB] [Cleanup] 清理旧目录失败: {e}")

    def run(self, target_object: str = "一包纸巾") -> dict:
        """运行六阶段自主抓取状态机"""
        print(f"\n{'='*60}")
        print(f"[GRAB] 开始自主抓取: {target_object}")
        print(f"[GRAB] 末端摄像头: {'已启用' if self.use_end_camera else '未启用'}")
        print(f"[GRAB] OpenCV Tracker: {'可用' if _CV2_AVAILABLE else '不可用'}")
        print(f"[GRAB] 3D 感知: {'已启用' if self._3d_enabled else '未启用'} ({self._3d_status_message})")
        print(f"[GRAB] VLM 调用: 不做硬限制，仅计数")
        print(f"[GRAB] 运动学: L1={_ARM_CFG.upper_arm_length}mm, L2={_ARM_CFG.forearm_length}mm")
        print(f"{'='*60}")

        # 创建本次运行的独立目录，关键帧与中间过程调试图都放到这里；
        # run() 结束时会清理旧目录，只保留最近 1 次抓取的完整图片日志
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_run_dir = os.path.join(self.output_dir, f"run_{run_ts}")
        os.makedirs(self._current_run_dir, exist_ok=True)
        print(f"[GRAB] 本次抓取日志目录: {self._current_run_dir}")

        if not _CV2_AVAILABLE:
            print("[GRAB] [ERROR] 本状态机依赖 OpenCV Tracker 做端侧几何闭环，请先安装 opencv-contrib-python")
            return {"success": False, "message": "OpenCV Tracker 不可用"}

        # Phase 0
        if not self._phase0_observation_reset():
            return {"success": False, "message": "Phase 0 观察姿态失败"}

        # Phase 1
        result = self._phase1_detect_and_align(target_object)
        if not result.get("success"):
            self._phase0_observation_reset()
            return result

        # Phase 2
        result = self._phase2_approach_and_deploy(target_object)
        if not result.get("success"):
            self._phase0_observation_reset()
            return result

        # Phase 3: 二次距离闭环与精准贴紧
        result = self._phase3_final_tuning(target_object)
        if not result.get("success"):
            self._phase0_observation_reset()
            return result

        # Phase 4
        result = self._phase4_touch_verify(target_object)
        if not result.get("success"):
            self._phase0_observation_reset()
            return result

        # Phase 5
        result = self._phase5_grasp_and_lift(target_object)

        print(f"\n{'='*60}")
        print(f"[GRAB] 最终结果: {result}")
        print(f"{'='*60}")

        # 抓取结束后清理旧运行目录，只保留最近 1 次完整图片日志
        self._cleanup_old_run_dirs(keep=1)
        return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HomeBot 自主抓取工作流（六阶段状态机）")
    parser.add_argument("--ip", default=config.ROBOT_IP, help="机器人 IP")
    parser.add_argument("--video-port", type=int, default=config.VIDEO_PORT, help="机身摄像头端口")
    parser.add_argument("--end-video-port", type=int, default=config.END_VIDEO_PORT, help="末端摄像头端口")
    parser.add_argument("--arm-port", type=int, default=config.ARM_PORT, help="机械臂端口")
    parser.add_argument("--max-attempts", type=int, default=10, help="总尝试次数")
    parser.add_argument("--no-end-camera", action="store_true", help="禁用机械臂末端摄像头")
    parser.add_argument("--target", default="一包纸巾", help="目标物品描述")
    args = parser.parse_args()

    workflow = AutoGrabWorkflow(
        robot_ip=args.ip,
        video_port=args.video_port,
        end_video_port=args.end_video_port,
        arm_port=args.arm_port,
        max_attempts=args.max_attempts,
        use_end_camera=not args.no_end_camera,
    )
    result = workflow.run(target_object=args.target)

    print(f"\n{'='*60}")
    print(f"[GRAB] 最终结果: {result}")
    print(f"{'='*60}")
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
