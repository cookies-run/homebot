#!/usr/bin/env python3
"""
Auto Grab Workflow - LLM 引导的自主抓取（运动学版）

正确流程：
    1. 机械臂回到"观察姿态"（rest_position，摄像头与桌面平齐）
    2. 底盘粗定位：只调 base（旋转）+ 底盘前进/后退，机械臂保持不动！
    3. 机械臂展开接近：使用逆运动学精确前伸，每次移动后视觉验证
    4. 下降、夹紧、抬起

核心改进（方案 B）：
    - 引入 ArmKinematics 运动学系统
    - 用 (r, z) 坐标控制末端位置，代替固定关节步进
    - 自动计算 wrist_flex = 180 - shoulder - elbow，保持末端方向一致
    - 实时追踪末端在空间中的实际位置

用法:
    python auto_grab_workflow.py              # 默认抓取纸巾
    python auto_grab_workflow.py --target "一个苹果"

要求:
    - 机器人摄像头已启动并发布到 ZeroMQ (默认端口 5560)
    - 机械臂服务已启动 (默认端口 5557)
    - VLM 视觉分析可用 (MiniMax 优先)

双摄像头配置:
    - 身上摄像头：默认端口 5560
    - 末端摄像头：默认端口 5561（可选，用于精对准）
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_subscriber import VideoSubscriber
from arm_control import HomeBotArmController
from chassis_control import HomeBotChassisController

# 视觉分析客户端优先级：MiniMax -> MiMo -> 火山引擎
_AVAILABLE_VLM = []
try:
    from minimax_vision_client import analyze_images as _minimax_analyze
    _AVAILABLE_VLM.append(("minimax", _minimax_analyze))
except ImportError:
    pass
try:
    from mimo_vision_client import analyze_images as _mimo_analyze
    _AVAILABLE_VLM.append(("mimo", _mimo_analyze))
except ImportError:
    pass
try:
    from volcengine_vision_client import analyze_images as _volcengine_analyze
    _AVAILABLE_VLM.append(("volcengine", _volcengine_analyze))
except ImportError:
    pass

VLM_PROVIDER = _AVAILABLE_VLM[0][0] if _AVAILABLE_VLM else None
if not _AVAILABLE_VLM:
    print("[WARN] 未找到视觉分析客户端，抓取功能将不可用")


def _is_quota_error(provider: str, error: Exception) -> bool:
    """判断是否为 Token Plan / 用量上限类错误，需要触发 fallback"""
    if provider != "minimax":
        return False
    err_str = str(error).lower()
    return any(k in err_str for k in ["2056", "token plan", "用量上限", "quota exceeded", "rate limit"])


def analyze_images_with_fallback(
    image_paths: list,
    prompt: str,
    max_tokens: int = 256,
    reasoning_effort: str = "low",
    timeout: int = 60,
) -> tuple[str, str]:
    """
    按优先级调用 VLM，当 MiniMax 达到 Token Plan 用量上限时自动回退到 MiMo / 火山引擎。

    Returns:
        (text_result, provider_name)
    """
    if not _AVAILABLE_VLM:
        raise RuntimeError("没有可用的视觉分析客户端")

    last_err = None
    for name, fn in _AVAILABLE_VLM:
        try:
            print(f"[GRAB] 尝试 VLM provider: {name}")
            if name == "volcengine":
                result = fn(
                    image_paths=image_paths,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )
            else:
                result = fn(
                    image_paths=image_paths,
                    prompt=prompt,
                    timeout=timeout,
                )
            print(f"[GRAB] VLM({name}) 调用成功")
            return result, name
        except Exception as e:
            err_str = str(e)
            print(f"[GRAB] VLM({name}) 调用失败: {err_str}")
            last_err = e
            if _is_quota_error(name, e):
                print(f"[GRAB] VLM({name}) 触发 quota 上限 fallback，尝试下一个 provider")
                continue
            # 非 quota 类错误不再继续浪费其他 provider 额度
            break
    raise last_err or RuntimeError("所有 VLM provider 均失败")

import robot_config as config

# 引入 YOLO（本地视觉伺服，Phase 2 像素闭环）
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# 加载机械臂关节限位和运动学
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../software/src'))
from configs.config import get_config as _get_arm_config
_ARM_CFG = _get_arm_config().arm
_JOINT_LIMITS = _ARM_CFG.joint_limits

# 引入原始代码的运动学系统
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../software/src/hal/arm'))
from Kinematics import ArmKinematics


# ========== 姿态定义 ==========
# 观察姿态：使用 config 中的 rest_position（复位姿态）
# 用户确认：复位后的角度能看清桌面和目标物体，摄像头与桌面平齐
# 底盘粗定位阶段机械臂必须保持此姿态不动！
_OBSERVATION_POSE = dict(_ARM_CFG.rest_position)
# 张开夹爪，准备观察；wrist_roll 提前设为 -90°，后续阶段无需再旋转
OBSERVATION_POSE = {**_OBSERVATION_POSE, "gripper": 90, "wrist_roll": -90}

# ========== 成功抓取姿态（基于实测） ==========
# 实测可成功夹住纸巾的姿态参数，作为接近和抓取的基准
_GRAB_SHOULDER = 20          # 大臂向上 20°
_GRAB_ELBOW = 167            # 小臂相对大臂 167°（反向折叠，末端低且靠后）
_GRAB_WRIST_FLEX = 5         # 夹爪略微上仰 5°
_GRAB_WRIST_ROLL = -90       # 夹爪侧向旋转 90°（关键！侧向夹取）

# 末端摄像头精调姿态（让末端摄像头能看到前方目标）
_END_CAM_SHOULDER = 20
_END_CAM_ELBOW = 135
_END_CAM_WRIST_FLEX = 180 - _END_CAM_SHOULDER - _END_CAM_ELBOW  # 25°

# 抓取展开姿态（从末端观察逐步过渡到抓取）
_APPROACH_SHOULDER = 20
_APPROACH_ELBOW_START = 135  # 末端观察
_APPROACH_ELBOW_END = 160    # 预抓取（接近桌面但未接触）


class AutoGrabWorkflow:
    """
    自主抓取工作流（运动学版）

    正确流程:
        Phase 0: 机械臂回到观察姿态
        Phase 1: 底盘粗定位（只调 base + 底盘移动，机械臂不动）
        Phase 2: 机械臂展开接近（运动学精确前伸 + 视觉闭环）
        Phase 3: 下降、夹紧、抬起
        Phase 4: 视觉验证
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
        self._end_camera_available = False  # 运行时检测末端摄像头是否可用

        self.arm = HomeBotArmController(robot_ip=self.robot_ip, robot_port=self.arm_port)
        self.chassis = HomeBotChassisController(
            ip=self.robot_ip, port=config.CHASSIS_PORT
        )

        # ---- 运动学系统 ----
        self._kin = ArmKinematics(
            L1=_ARM_CFG.upper_arm_length,
            L2=_ARM_CFG.forearm_length,
        )
        # 当前末端位置 (r=水平距离mm, z=垂直高度mm)
        self._current_rz = {"r": 0.0, "z": 0.0}

        # 缓存关节角度
        self._last_aligned_angles = None
        # 缓存 wrist_roll（防止 motion service 状态缺失导致重置为 0）
        self._last_wrist_roll = 0

        # 持久化档位状态（Phase 2 低空闭环单向降档锁）
        self.current_gear = "coarse"
        self.output_dir = os.path.join(os.path.dirname(__file__), "grab_captures")
        os.makedirs(self.output_dir, exist_ok=True)

        # ---- 粗定位步进（底盘 + base 旋转）----
        self.base_step = 2           # base 旋转步进 — 减小避免矫枉过正（视差补偿需更精细）
        self.chassis_step_cm = 6     # 底盘移动步进（厘米）— 加大步长，更快接近目标

        # ---- 运动学接近参数 ----
        self.extend_step_mm = 30     # 每次前伸距离 (mm)
        self.lower_step_mm = 20      # 下降距离 (mm)
        self.lift_height_mm = 30     # 抬起高度 (mm)

        # ---- 精微调步进 ----
        self.fine_base_step = 1      # 减小到1°，避免矫枉过正（用户反馈偏右3°就碰到了）
        self.fine_r_step = 2         # 水平微调 (mm)
        self.fine_z_step = 2         # 垂直微调 (mm)

        # ---- 抓取前伸深度补偿（Deep Bite）----
        # 视觉对准的是物体表面，抓取需要夹爪前伸跨越边缘。
        # 在最终对准后、下落前，底盘额外前进此距离，让夹爪掌心越过物体前边缘。
        # 建议范围：2.5 ~ 4.5 cm（根据物体大小调节）
        self.grasp_forward_compensation_cm = 3.5

        # ---- 加载 YOLO 模型（Phase 2 像素伺服）----
        self._init_yolo()

    # ==================== YOLO 视觉伺服 ====================

    def _init_yolo(self):
        """加载本地 YOLO 模型（用于 Phase 2 像素闭环伺服）"""
        self.yolo_model = None
        if not _YOLO_AVAILABLE:
            print("[GRAB] [WARN] ultralytics 未安装，Phase 2 将回退到 VLM")
            return

        # 优先 yolo26n.pt（新且轻量），其次 yolo11n.pt
        candidates = [
            os.path.join(os.path.dirname(__file__), '../../../software/models/yolo26n.pt'),
            os.path.join(os.path.dirname(__file__), '../../../software/models/yolo11n.pt'),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    self.yolo_model = YOLO(path)
                    print(f"[GRAB] YOLO 模型已加载: {path}")
                    return
                except Exception as e:
                    print(f"[GRAB] [WARN] 加载 YOLO 模型失败 {path}: {e}")
                    continue
        print("[GRAB] [WARN] 未找到可用 YOLO 模型，Phase 2 将回退到 VLM")

    def _analyze_with_yolo(self, image_path: str, target_object: str = None) -> dict | None:
        """使用本地 YOLO 进行视觉伺服分析

        Args:
            image_path: 图像文件路径
            target_object: 目标物体描述（YOLO 通用模型不依赖此参数，仅用于日志）

        Returns:
            {
                "cx": float, "cy": float,      # 目标中心绝对像素坐标
                "w": float, "h": float,        # 边界框宽高
                "img_w": int, "img_h": int,    # 画面总分辨率
                "confidence": float,            # 置信度
            }
            未检测到返回 None
        """
        import cv2

        if self.yolo_model is None:
            return None

        img = cv2.imread(image_path)
        if img is None:
            print(f"[GRAB] [YOLO] 无法读取图像: {image_path}")
            return None
        img_h, img_w = img.shape[:2]

        try:
            results = self.yolo_model(image_path, verbose=False)
        except Exception as e:
            print(f"[GRAB] [YOLO] 推理异常: {e}")
            return None

        if not results or len(results) == 0:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # 筛选检测框：过滤过大（背景）和过小（噪声）的框
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        valid_candidates = []
        for idx, (box, conf_val) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            area_ratio = (bw * bh) / (img_w * img_h)
            # 过滤掉占满画面的背景框和极小的噪声
            if area_ratio > 0.85 or area_ratio < 0.005:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # 计算到画面中心的距离（优先选画面中心的物体）
            center_dist = ((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2) ** 0.5
            valid_candidates.append({
                "idx": idx, "conf": float(conf_val), "area_ratio": area_ratio,
                "cx": cx, "cy": cy, "w": bw, "h": bh, "center_dist": center_dist,
            })

        if not valid_candidates:
            print(f"[GRAB] [YOLO] 无有效检测框（共 {len(confs)} 个被过滤）")
            return None

        # 优先选置信度高且靠近画面中心的（近距离伺服时目标通常在中心附近）
        valid_candidates.sort(key=lambda x: (x["center_dist"] / max(img_w, img_h) - x["conf"] * 0.5))
        best = valid_candidates[0]

        print(f"[GRAB] [YOLO] 检测到目标: conf={best['conf']:.2f}, "
              f"bbox=({best['w']:.0f}x{best['h']:.0f}), "
              f"中心=({best['cx']:.0f},{best['cy']:.0f}), 画面={img_w}x{img_h}, "
              f"面积占比={best['area_ratio']:.3f}")

        return {
            "cx": float(best["cx"]),
            "cy": float(best["cy"]),
            "w": float(best["w"]),
            "h": float(best["h"]),
            "img_w": img_w,
            "img_h": img_h,
            "confidence": best["conf"],
        }

    # ==================== 运动学辅助 ====================

    def _refresh_arm_position(self) -> dict:
        """从实际关节角度计算当前末端位置 (r, z)

        模仿原始 mcp_server.py 中的 _refresh_arm_position()
        """
        angles = self.get_current_angles()
        shoulder = angles.get("shoulder", 0)
        elbow = angles.get("elbow", 150)
        r, z = self._kin.forward_kinematics(shoulder, elbow)
        self._current_rz = {"r": r, "z": z}
        print(f"[GRAB] [运动学] 当前末端位置: r={r:.1f}mm, z={z:.1f}mm "
              f"(shoulder={shoulder:.1f}°, elbow={elbow:.1f}°)")
        return self._current_rz

    def _move_to_rz(self, target_r: float, target_z: float, desc: str, wait: float = 1.5, fixed_shoulder: float = None) -> bool:
        """用逆运动学移动到指定 (r, z)，自动计算 wrist_flex 保持末端方向

        Args:
            fixed_shoulder: 如果指定，固定 shoulder 角度，只解 elbow。
                            用于已知 shoulder 的抓取姿态（如 shoulder=20°）。

        关键修正：限位 clamp 后重新计算实际 (r,z)，避免假设已到达目标。
        """
        import math

        lim = _JOINT_LIMITS
        L1 = _ARM_CFG.upper_arm_length
        L2 = _ARM_CFG.forearm_length

        if fixed_shoulder is not None:
            # 固定 shoulder，只解 elbow
            shoulder_raw = fixed_shoulder
            dx = target_r - L1 * math.cos(math.radians(shoulder_raw))
            dy = target_z - L1 * math.sin(math.radians(shoulder_raw))
            dist = math.hypot(dx, dy)
            if dist > L2 + 0.1 or dist < abs(L2 - L1) - 0.1:
                print(f"[GRAB] [ERR] {desc}: 目标({target_r:.0f}, {target_z:.0f}) 在固定 shoulder={shoulder_raw}° 时不可达，"
                      f"需要小臂长度={dist:.1f}mm，实际 L2={L2}mm")
                return False
            abs_angle = math.degrees(math.atan2(dy, dx))
            elbow_raw = abs_angle - shoulder_raw
            # 规范到 [-180, 180] 范围
            while elbow_raw > 180:
                elbow_raw -= 360
            while elbow_raw < -180:
                elbow_raw += 360
        else:
            # 检查可达性
            if not self._kin.is_reachable(target_r, target_z):
                min_r, max_r = self._kin.get_workspace_radius()
                print(f"[GRAB] [ERR] {desc}: 目标({target_r:.0f}, {target_z:.0f}) 不可达，"
                      f"工作空间 r∈[{min_r:.0f}, {max_r:.0f}]")
                return False

            # 逆运动学计算 shoulder/elbow
            angles = self._kin.inverse_kinematics(target_r, target_z, elbow_up=True)
            if angles is None:
                print(f"[GRAB] [ERR] {desc}: 逆运动学无解，目标({target_r:.0f}, {target_z:.0f})")
                return False
            shoulder_raw, elbow_raw = angles

        # 自动计算 wrist_flex
        wrist_flex_raw = self._kin.compute_wrist_flex(shoulder_raw, elbow_raw, target_orientation=0.0)

        # 限位
        shoulder = self.clamp(shoulder_raw, lim["shoulder"][0], lim["shoulder"][1])
        elbow = self.clamp(elbow_raw, lim["elbow"][0], lim["elbow"][1])
        wrist_flex = self.clamp(wrist_flex_raw, lim["wrist_flex"][0], lim["wrist_flex"][1])

        # 用 clamp 后的角度重新计算实际可达位置（关键！）
        actual_r, actual_z = self._kin.forward_kinematics(shoulder, elbow)

        # 如果 clamp 导致严重偏离，打印警告
        if abs(shoulder - shoulder_raw) > 5 or abs(elbow - elbow_raw) > 5:
            print(f"[GRAB] [WARN] {desc}: 关节限位介入，原始 shoulder={shoulder_raw:.1f}° "
                  f"-> clamp={shoulder:.1f}°，实际将到达 r={actual_r:.1f}, z={actual_z:.1f}")

        # 保持当前 base 不变；wrist_roll 使用本地缓存（防止 motion service 缺失该字段）
        base = self.get_current_angles().get("base", 0)

        pose = {
            "base": base,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist_flex": wrist_flex,
            "wrist_roll": self._last_wrist_roll,
            "gripper": 90,
        }

        print(f"[GRAB] [运动学] {desc}: target(r={target_r:.1f}, z={target_z:.1f}) -> "
              f"实际(r={actual_r:.1f}, z={actual_z:.1f}) "
              f"shoulder={shoulder:.1f}°, elbow={elbow:.1f}°, wrist_flex={wrist_flex:.1f}°")

        ok = self._move_to_pose(pose, desc, wait)
        if ok:
            # 用实际位置更新，而不是假设到达 target
            self._current_rz = {"r": actual_r, "z": actual_z}
        return ok

    def _move_to_rz_relative(self, dr: float = 0, dz: float = 0, desc: str = "相对移动", wait: float = 1.5) -> bool:
        """基于当前位置相对移动 (dr, dz)"""
        self._refresh_arm_position()
        target_r = self._current_rz["r"] + dr
        target_z = self._current_rz["z"] + dz
        # 确保不碰到地面
        target_z = max(20.0, target_z)
        return self._move_to_rz(target_r, target_z, desc, wait)

    # ==================== 图像捕获 ====================

    def _capture_from(self, port: int, label: str) -> str:
        """从指定端口捕获图像"""
        subscriber = VideoSubscriber(self.robot_ip, port)
        try:
            print(f"[GRAB] 正在捕获图像 ({label}, 端口 {port})...")
            frame = subscriber.wait_for_frame(timeout_seconds=5.0)
            if frame is None:
                print(f"[GRAB] [ERROR] 图像捕获超时 (端口 {port})")
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.output_dir, f"grab_{label}_{timestamp}.jpg")
            with open(path, "wb") as f:
                f.write(frame)
            print(f"[GRAB] 图像已保存: {path}")
            return path
        finally:
            subscriber.close()

    def capture(self, label: str = "") -> str:
        """订阅身上摄像头"""
        return self._capture_from(self.video_port, label)

    def capture_end(self, label: str = "") -> str:
        """订阅末端摄像头"""
        return self._capture_from(self.end_video_port, f"end_{label}")

    def _check_end_camera(self) -> bool:
        """检测末端摄像头是否可用（增加重试，容忍偶发丢帧）"""
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

    # ==================== VLM 分析 ====================

    def analyze(self, image_path: str, target_object: str = "一包纸巾") -> str:
        """调用 VLM 视觉分析，用于 Phase 1 底盘粗定位（只评估前后距离 + 可见性）"""
        prompt = f"""你是机器人的高精度视觉分析助手。这张图片来自机器人机身主摄像头，用于底盘极其贴近的粗定位（Phase 1）。
由于日常场景中，承载物体的桌面（载物台）高度可能比机器人高，也可能比机器人矮，请结合“物体边缘截断”与“背景遮挡”进行综合距离判定。

Your core task is: Locate "{target_object}" and judge the distance category strictly based on the following robust physical cues.

⚠️ 核心过滤原则：
- 画面中可能存在机械臂本身的结构或外壳遮挡，请在分析时【完全忽略机械臂】，只盯着目标物体与桌面的几何关系。

📏 视差自适应距离比例尺：

1. 【太远】：物体在画面中是个“微型小方块”（高度占比 < 15%），能看到大面积的房间远景、地面或宏观环境。
2. 【较远】：物体轮廓完整，高度占比在 [15% - 40%] 之间，周围环境或完整的桌子外形依然清晰。
3. 【适中】：物体高度占比在 [40% - 60%] 之间，底盘已驶近桌沿，物体和桌面开始占据画面中心。
4. 【较近】：物体非常巨大（占比 60% - 80%），纹理极清晰。但此时【物体整体依然完全在画面内，没有被任何视场边缘截断】，且隐约还能看到桌子两侧或下方的远景。
5. 【太近】（停止线/极度接近，物理距离 < 5cm）：
   只要满足以下任意一条，必须判定为【太近】：
   - 【物体截断】：物体的顶部、底部或侧边，由于离镜头过近，已经【超出了画面的边缘（看不全完整形体了）】（如顶部直接顶天出框）。
   - 【视野堵死】：物体与其下方的粗大桌面边缘（桌沿）组合起来，呈现压倒性的局部大特写，后方和下方的房间远景/地面已被 100% 完全遮挡，视觉深度锁死。

请先在脑海中严格按照上述【截断与堵死】原则进行几何推算，然后【仅严格按以下固定格式输出两行文本，不要包含任何推理过程、Markdown标记或额外解释】：

- 距离：<太远/较远/适中/较近/太近/未知>
- 可见性：<看到了/未找到>

格式示例：
- 距离：太近
- 可见性：看到了

如果没找到目标物体，回复：
- 距离：未知
- 可见性：未找到
"""
        try:
            result, used_provider = analyze_images_with_fallback(
                image_paths=[image_path],
                prompt=prompt,
            )
            print(f"[GRAB] VLM({used_provider}) 分析结果: {result.strip()}")
            return result.strip()
        except Exception as e:
            print(f"[GRAB] [ERROR] 图像分析失败: {e}")
            return None

    def analyze_alignment(self, image_path: str, target_object: str, use_end_camera: bool = False) -> str:
        """分析夹爪与目标的相对对准情况（抓取前微调用）

        Args:
            use_end_camera: 是否使用末端摄像头（夹爪视角）
        """
        if use_end_camera:
            # 低空精调 Prompt：夹爪末端相机近距离俯视视角
            prompt = f"""你是一个高精度的具身智能机器人末端相机视觉感知专家。你的任务是通过观察画面中的【机械臂左爪】与【目标物体】的相对空间几何关系，评估抓取对准状态。

### 🔍 核心物体识别
1. 【机械臂左爪】：画面左下角出现的白色塑料结构（夹爪的左侧指尖）。
2. 【目标物体】：当前机器人需要抓取的物体，具体表现为：{target_object}

### 🚨 核心对准推理逻辑（针对倾斜物体的左爪内侧基准线法则）
请在图像中，沿【机械臂左爪】的【右侧内边沿】向上发射一条【虚拟的垂直探测线】。

如果目标物体相对于相机发生了倾斜（不平行），其左边缘会呈现为一条斜线。请严格按下述规则判定：

1. **中心/已对准（Centered - 允许微小倾斜容错 🎯）**：
   - 现象：目标物体【离夹爪最近的左前角（底部拐角）】刚好紧贴在探测线上，或者探测线虽然切入了物体，但【切入的横向深度极浅（小于物体总体宽度的5%）】。
   - 物理含义：此时夹爪下落刚好能擦着物体的左前边缘进去，右侧空间也足够，属于可安全抓取的黄金核心区。

2. **偏左（Slight Left）**：
   - 现象：虚拟探测线【大幅度切入了】目标物体的身体内部，目标物体的左前拐角或大面积包装【明显漏在了探测线的左边】。
   - 物理含义：下落必撞，必须向左微调。

3. **偏右（Slight Right）**：
   - 现象：目标的任何部分都在探测线的右侧，且目标的左前拐角与探测线之间【有明显的肉眼可见空当】。
   - 物理含义：太靠右，必须向右微调。

### 📐 垂直距离判定（Y轴位置）
- 目标物体在视野中上部：垂直距离：较远
- 目标物体在视野正中部：垂直距离：适中
- 目标物体已极度放大，且与左爪在视觉上发生上下重叠：垂直距离：太近

### 🧠 链式思考（CoT）推理步骤
在输出最终标签前，请在心中执行以下三步硬性比对：
1. 找到左爪的右侧内边沿，向上拉出垂直线。
2. 检查：是否有任何目标的身体部分漏在了这条线的左边？如果有，直接输出【水平偏差：偏左】，拒绝承认是对准！
3. 如果完全在右边，检查左边缘是否紧贴这条线？紧贴则为【中心】，有空当则为【偏右】。

### 格式化输出规范
请严格按照以下键值对格式返回，不要有任何多余的解释、寒暄或Markdown标记：
水平偏差：<中心/偏左/偏右/极左/极右>
垂直距离：<太远/较远/适中/较近/太近/已接触或超出>
对准状态：<已对准/未对准> （注意：只有水平偏差为"中心"时，对准状态才为"已对准"）"""
        else:
            # 机身摄像头 prompt：中距离，从外部观察夹爪和目标
            prompt = f"""你是机器人的视觉分析助手。请观察图片，重点关注机械臂末端夹爪与目标物体"{target_object}"的相对位置。

请判断目标物体相对于夹爪中心的位置：
- 水平方向：极左 / 偏左 / 中心 / 偏右 / 极右
- 垂直方向：极高 / 偏高 / 中心 / 偏低 / 极低
- 距离：太远（画面边缘）/ 较远 / 适中（正前方）/ 较近（几乎接触）/ 已接触或超出
- 可抓取：可以 / 不可（被遮挡/太远/角度不对）

如果夹爪已经正对目标物体正上方且距离适中，回复：已对准

否则回复格式示例：
水平：偏左，垂直：中心，距离：适中，可抓取：可以
"""
        try:
            result, used_provider = analyze_images_with_fallback(
                image_paths=[image_path],
                prompt=prompt,
            )
            print(f"[GRAB] VLM({used_provider}) 对准分析: {result.strip()}")
            return result.strip()
        except Exception as e:
            print(f"[GRAB] [ERROR] 对准分析失败: {e}")
            return None

    def parse_analysis(self, text: str) -> dict:
        """从 LLM 回复文本中解析结构化方位信息"""
        if not text:
            return None

        text_lower = text.lower()
        if "未找到" in text or "没找到" in text or "not found" in text_lower:
            return {"found": False}

        def extract(label: str, options: list) -> str:
            for opt in options:
                if opt in text:
                    return opt
            return "中心"

        # 注意：必须先匹配带方向修饰词（如"中心偏右"里的"偏右"），再兜底"中心"
        horizontal = extract("水平", ["极左", "偏左", "偏右", "极右", "中心"])
        vertical = extract("垂直", ["极高", "偏高", "偏低", "极低", "中心"])
        size = extract("大小", ["很小", "较小", "适中", "较大", "极大"])
        # 兼容对准分析中的"距离"/"垂直距离"字段以及 Phase 1 粗定位词汇
        distance_raw = extract(
            "距离",
            ["太远", "较远", "适中", "较近", "已接触或超出", "太近"]
        )
        distance = distance_raw  # 提示词与内部档位已统一，无需额外映射
        reachable = "不可" not in text and ("可以" in text or "可抓取" in text)
        aligned = "已对准" in text
        # 解析可见性（Phase 1 粗定位新 Prompt）
        visible = "未找到" not in text and ("看到了" in text or "看到" in text or reachable)

        return {
            "found": True,
            "horizontal": horizontal,
            "vertical": vertical,
            "size": size,
            "distance": distance,
            "reachable": reachable,
            "aligned": aligned,
            "visible": visible,
        }

    def is_aligned_for_chassis(self, info: dict) -> bool:
        """判断底盘粗定位是否完成：目标在水平偏右/中心 + 距离适中

        视差 Hack：主摄像头在机器人正中间，机械臂底座在左侧。
        当主摄像头看到目标在"中心"时，对左侧的机械臂来说目标其实在右侧。
        因此必须让目标在主摄像头画面中位于"偏右"（甚至"极右"），
        这样左侧的机械臂朝前伸出时才能正对目标。
        兼容"中心"是为了避免过度震荡，但主推"偏右"。
        """
        return (
            info["horizontal"] in ("偏右", "中心")
            and info["size"] in ("适中", "较大")
        )

    def is_aligned_for_grab(self, info: dict) -> bool:
        """判断机械臂是否已对准可抓取：水平+垂直中心 + 距离适中/较近"""
        return (
            info.get("aligned", False)
            or (
                info["horizontal"] == "中心"
                and info["vertical"] == "中心"
                and info.get("distance", info["size"]) in ("适中", "较近", "较大")
            )
        )

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

    def _move_to_pose(self, pose: dict, desc: str, wait: float = 1.5) -> bool:
        """发送关节角度指令并等待到位"""
        lim = _JOINT_LIMITS
        target = {}
        current = self.get_current_angles()
        for name in ["base", "shoulder", "elbow", "wrist_flex", "wrist_roll", "gripper"]:
            if name == "wrist_roll":
                # 特殊处理 wrist_roll：优先使用 pose 值，其次用本地缓存，最后回退到当前状态
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
            return True
        else:
            print(f"[GRAB] [ERR] {desc} 失败")
            return False

    # ==================== Phase 0: 回到观察姿态 ====================

    def _move_to_observation_pose(self) -> bool:
        """机械臂回到观察姿态，准备底盘粗定位"""
        print("\n[GRAB] ====== Phase 0: 机械臂回到观察姿态 ======")
        p = OBSERVATION_POSE
        print(f"[GRAB] 观察姿态: base={p['base']}, shoulder={p['shoulder']}, "
              f"elbow={p['elbow']}, wrist_flex={p['wrist_flex']} (复位角度，摄像头与桌面平齐)")
        ok = self._move_to_pose(OBSERVATION_POSE, "机械臂已回到观察姿态", wait=2.0)
        if ok:
            # 同步运动学位置
            self._refresh_arm_position()
        return ok

    # ==================== Phase 1: 底盘粗定位 ====================

    def _chassis_coarse_alignment(self, target_object: str) -> dict:
        """底盘粗定位闭环：base 锁死，只根据机身主摄像头距离前后移动底盘

        放行条件：
        - 当 VLM 明确返回"太近"时，视为已到达最佳逼近点，底盘停机放行进入 Phase 2。
        - 当检测到【较近 -> 太近】的状态跃迁时，视为越过最佳点，
          后退 5cm 建立安全垫后破圈放行。
        "太远"/"较远"/"适中"/"较近" 使用基于物理距离的变步长继续逼近；
        初始状态即"太近"/"已接触或超出" 温柔后退，拒绝放行。
        视觉达到收敛时底盘原地锁定，直接丝滑切换到 Phase 2 的低空相机闭环精对准。

        Returns:
            最后一次解析的 info 字典
        """
        print("\n[GRAB] ====== Phase 1: 底盘粗定位（base 锁死，只移动底盘）======")
        print("[GRAB] 原则: base 保持观察姿态不动，只根据距离调整底盘前后位置")
        print("[GRAB] 放行条件: 距离=太近，或探测到【较近 -> 太近】过度接近")
        print("[GRAB] 纯净策略: 基于物理距离的变步长逼近 + 过度接近保护，零尾端补偿")

        # base 锁死为观察姿态角度（默认 -90°）
        current_base = OBSERVATION_POSE.get("base", -90)
        print(f"[GRAB] [底盘粗定位] base 锁死={current_base:.0f}°")
        last_info = None
        last_distance = None  # 状态历史记忆，捕捉 较近 -> 太近 的过渡
        attempts = min(8, self.max_attempts)  # 最多 8 次，给足推进空间

        for attempt in range(1, attempts + 1):
            print(f"\n[GRAB] [底盘粗定位] --- 第 {attempt}/{attempts} 次 ---")

            img_path = self.capture(f"底盘粗定位_{attempt}")
            if img_path is None:
                continue

            analysis = self.analyze(img_path, target_object)
            if analysis is None:
                continue

            info = self.parse_analysis(analysis)
            if not info or not info["found"]:
                print(f"[GRAB] [底盘粗定位] 未找到目标，继续尝试")
                continue

            print(
                f"[GRAB] [底盘粗定位] 目标状态: "
                f"距离={info['distance']}, 可见性={info.get('visible', True)}"
            )
            last_info = info
            dist = info["distance"]

            # base 锁死：保持当前观察姿态的 base 角度
            target = dict(OBSERVATION_POSE)
            target["base"] = current_base
            resp = self.arm.set_joint_angles(target)
            if resp and resp.success:
                print("[GRAB] [底盘粗定位] 观察姿态保持（base 锁死）")
                time.sleep(0.5)
            else:
                print(f"[GRAB] [ERR] 观察姿态保持失败")
                continue

            # 过度接近保护：检测到 较近 -> 太近 的状态跃迁
            if dist == "太近":
                if last_distance == "较近":
                    print(
                        "[GRAB] [底盘粗定位] 探测到【较近 -> 太近】过度接近！"
                        "底盘后退 2.0cm 建立安全垫，然后放行 Phase 2！"
                    )
                    self.chassis.backward_cm(2.0)
                    time.sleep(0.5)
                    return info
                else:
                    print(
                        f"[GRAB] [底盘粗定位] 初始状态即太近（last={last_distance}），"
                        f"底盘后退 5.0cm 重新调整"
                    )
                    self.chassis.backward_cm(5.0)
                    time.sleep(0.5)
                    last_distance = "太近"
                    continue

            # 已接触或超出：后退 5cm，重新调整
            if dist == "已接触或超出":
                print("[GRAB] [底盘粗定位] 距离=已接触或超出，后退 5.0cm 重新调整")
                self.chassis.backward_cm(5.0)
                time.sleep(0.5)
                last_distance = dist
                continue

            # 基于物理距离的变步长推进
            # 太远: >1.5m 微型小点 (<10%) 大步冲锋
            # 较远: 0.8-1.5m 小方块 (10%-25%) 中步逼近
            # 适中: 0.4-0.8m 中等轮廓 (25%-40%) 精细控制
            # 较近: 0.2-0.4m 明显放大 (40%-50%) 小步逼近，直到进入"太近"
            dist_move_map = {
                "太远": 30,
                "较远": 10,
                "适中": 4,
                "较近": 3,
            }
            chassis_move = dist_move_map.get(dist, 0)
            if chassis_move > 0:
                print(
                    f"[GRAB] [底盘粗定位] base 锁死={current_base:.0f}°，"
                    f"距离={dist}，底盘前进 {chassis_move}cm"
                )
                self.chassis.forward_cm(chassis_move)
                time.sleep(0.5)
            else:
                print(f"[GRAB] [底盘粗定位] 未知距离 {dist}，底盘保持不动")

            # 更新状态历史记忆
            last_distance = dist

        print("[GRAB] [底盘粗定位] 达到最大尝试次数，按最后状态放行进入 Phase 2")
        return last_info if last_info else {
            "found": True,
            "horizontal": "中心",
            "vertical": "中心",
            "size": "适中",
            "distance": "适中",
            "reachable": True,
            "aligned": False,
            "visible": True,
        }

    # ==================== Phase 2: 机械臂展开接近（运动学版） ====================

    def _fine_tune_rz(self, info: dict) -> bool:
        """根据 VLM 反馈用运动学微调末端位置

        根据水平/垂直/距离偏差，微调 r/z 坐标
        """
        h = info["horizontal"]
        v = info["vertical"]
        d = info.get("distance", info.get("size", "适中"))

        # 先刷新当前位置
        self._refresh_arm_position()
        target_r = self._current_rz["r"]
        target_z = self._current_rz["z"]

        # 水平偏差：通过 base 旋转微调（不改变 r/z）
        # 画面与物理方向一致（flip_horizontal=False）
        base_offset = 0
        if h == "极左":
            base_offset = -self.fine_base_step * 2
        elif h == "偏左":
            base_offset = -self.fine_base_step
        elif h == "极右":
            base_offset = self.fine_base_step * 2
        # "偏右"和"中心"不调整 base（已对准）

        if base_offset != 0:
            current = self.get_current_angles()
            new_base = current.get("base", 0) + base_offset
            new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
            print(f"[GRAB] [运动学微调] 水平偏差({h}) -> base 调整 {base_offset:+.0f}°")
            self._move_to_pose({
                "base": new_base,
                "shoulder": _APPROACH_SHOULDER,
                "elbow": current.get("elbow", _APPROACH_ELBOW_START),
                "wrist_flex": 180 - _APPROACH_SHOULDER - current.get("elbow", _APPROACH_ELBOW_START),
                "wrist_roll": _GRAB_WRIST_ROLL,
                "gripper": 90,
            }, "base 微调", wait=1.0)

        # 垂直偏差：微调 z（上下移动）
        z_offset = 0
        if v == "极高":
            z_offset = self.fine_z_step * 2
        elif v == "偏高":
            z_offset = self.fine_z_step
        elif v == "偏低":
            z_offset = -self.fine_z_step
        elif v == "极低":
            z_offset = -self.fine_z_step * 2

        # 距离偏差：微调 r（前后移动）
        r_offset = 0
        if d in ("太远", "很小"):
            r_offset = self.fine_r_step * 2
        elif d in ("较远", "较小"):
            r_offset = self.fine_r_step
        elif d in ("较近", "已接触或超出", "较大", "极大"):
            r_offset = -self.fine_r_step

        if r_offset != 0 or z_offset != 0:
            target_r += r_offset
            target_z += z_offset
            target_z = max(20.0, target_z)  # 不低于 20mm
            print(f"[GRAB] [运动学微调] 垂直={v}, 距离={d} -> "
                  f"r 调整 {r_offset:+.0f}mm, z 调整 {z_offset:+.0f}mm")
            return self._move_to_rz(target_r, target_z, "运动学微调", wait=1.2, fixed_shoulder=_APPROACH_SHOULDER)

        return True

    def _capture_for_approach(self, label: str, step: int) -> tuple[str | None, bool]:
        """抓取接近阶段拍照：优先使用末端摄像头，回退到机身摄像头

        Returns:
            (image_path, used_end_camera)
        """
        if self._end_camera_available:
            img_path = self.capture_end(f"{label}_{step}")
            if img_path is not None:
                return img_path, True
            print(f"[GRAB] [接近] 末端摄像头捕获失败，回退到机身摄像头")
        # 回退到机身摄像头
        img_path = self.capture(f"{label}_{step}")
        return img_path, False

    def _analyze_with_retry(self, img_path: str, target_object: str, use_end_camera: bool, retries: int = 3) -> str | None:
        """调用 VLM 对准分析，带指数退避重试（解决偶发 API 超时）"""
        for i in range(retries):
            analysis = self.analyze_alignment(img_path, target_object, use_end_camera=use_end_camera)
            if analysis is not None:
                return analysis
            wait = min(2 ** i, 8)
            print(f"[GRAB] VLM API 失败，{wait}s 后重试 ({i + 1}/{retries})...")
            time.sleep(wait)
        print("[GRAB] [ERROR] VLM API 多次重试后仍失败")
        return None

    def _align_base_with_end_camera(self, target_object: str, base_angle: float) -> float:
        """用末端摄像头（夹爪视角）精调 base，确保夹爪正对目标

        在主摄像头完成底盘粗定位后，夹爪视角下目标可能仍有水平偏差。
        这里先把手臂小幅展开到 elbow=135°（末端摄像头能看到前方），
        然后根据末端摄像头反馈迭代微调 base。

        Returns:
            精调后的 base 角度
        """
        if not self._end_camera_available:
            return base_angle

        print("\n[GRAB] [末端精调] 用夹爪视角精调 base 角度...")
        current_base = base_angle

        # 先移动到小幅展开姿态，让末端摄像头能看到目标
        # 使用 shoulder=20°（实测成功姿态），elbow=135° 保证末端摄像头朝前下方
        # wrist_roll=-90° 是关键：末端摄像头安装在夹爪上，roll 角度决定画面方向
        self._move_to_pose({
            "base": current_base,
            "shoulder": _END_CAM_SHOULDER,
            "elbow": _END_CAM_ELBOW,
            "wrist_flex": _END_CAM_WRIST_FLEX,
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 90,
        }, f"末端精调姿态 (shoulder={_END_CAM_SHOULDER}°, elbow={_END_CAM_ELBOW}°, wrist_roll=-90°)", wait=2.0)
        self._refresh_arm_position()

        for attempt in range(1, 6):
            print(f"[GRAB] [末端精调] --- 第 {attempt}/5 次 ---")
            img_path = self.capture_end(f"base精调_{attempt}")
            if img_path is None:
                continue

            analysis = self._analyze_with_retry(img_path, target_object, use_end_camera=True)
            if analysis is None:
                continue

            if "已对准" in analysis:
                print("[GRAB] [末端精调] ✅ VLM 确认已对准，进入低空闭环二次确认")
                _, final_base = self._low_alt_fine_tune_loop(target_object, current_base)
                return final_base

            info = self.parse_analysis(analysis)
            if not info or not info["found"]:
                print("[GRAB] [末端精调] 未找到目标，跳过本次")
                continue

            h = info["horizontal"]
            print(f"[GRAB] [末端精调] 水平={h}, 垂直={info['vertical']}, "
                  f"距离={info.get('distance', info.get('size', '适中'))}, 可抓取={info['reachable']}")

            # 目标必须在画面中心才算对准（用户要求：目标在中间更好抓取）
            if h == "中心":
                print(f"[GRAB] [末端精调] ✅ 水平=中心，进入低空闭环二次确认")
                _, final_base = self._low_alt_fine_tune_loop(target_object, current_base)
                return final_base

            # 水平偏差 -> 调整 base（步长减小到1-2°，避免矫枉过正）
            # 画面与物理方向一致（flip_horizontal=False）
            h_map = {"极左": -2, "偏左": -1, "偏右": 1, "极右": 2}
            offset = h_map.get(h, 0)
            if offset == 0:
                print(f"[GRAB] [末端精调] 水平={h}，无需 base 调整，进入低空闭环二次确认")
                _, final_base = self._low_alt_fine_tune_loop(target_object, current_base)
                return final_base

            new_base = current_base + offset
            new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
            print(f"[GRAB] [末端精调] 水平={h} -> base {current_base:.0f}° -> {new_base:.0f}°")
            self._move_to_pose({
                "base": new_base,
                "shoulder": _END_CAM_SHOULDER,
                "elbow": _END_CAM_ELBOW,
                "wrist_flex": _END_CAM_WRIST_FLEX,
                "wrist_roll": _GRAB_WRIST_ROLL,
                "gripper": 90,
            }, f"base 精调到 {new_base:.0f}°", wait=1.5)
            current_base = new_base

        print(f"[GRAB] [末端精调] 达到最大尝试次数，进入低空闭环")
        _, final_base = self._low_alt_fine_tune_loop(target_object, current_base)
        return final_base

    def _arm_approach_with_vision_vlm(self, target_object: str, base_angle: float = 0) -> bool:
        """机械臂展开接近目标（VLM 视觉闭环 - 回退路径）

        修正：高空斜视视差会导致 VLM 误判"已对准"。
        因此先用末端摄像头精调 base，一旦 VLM 认为对准，立即进入低空闭环精调，
        把剩余的深度死区（约 11cm）100% 交给底盘移动 + 夹爪低空视角闭环来蚕食。
        """
        print("\n[GRAB] ====== Phase 2: 机械臂展开接近（VLM + 低空闭环） ======")

        # 检测末端摄像头
        self._check_end_camera()
        if self._end_camera_available:
            print("[GRAB] [接近] 将使用末端摄像头做视觉闭环，并在对准后进入低空闭环")
        else:
            print("[GRAB] [接近] ⚠️ 末端摄像头不可用，使用机身摄像头")

        # 同步当前运动学位置
        self._refresh_arm_position()

        # ---- 末端摄像头可用：精调 base 后直接进入低空闭环 ----
        # 高空斜视视差会导致 VLM 误判"已对准"，所有末端摄像头路径最终都进入低空闭环，
        # 由底盘移动 + 夹爪低空视角闭环蚕食剩余的深度死区（约 11cm）。
        if self._end_camera_available:
            base_angle = self._align_base_with_end_camera(target_object, base_angle)
            return True

        # 确保 base 正确，同时过渡到 shoulder=20° 的接近姿态
        current = self.get_current_angles()
        if abs(current.get("base", 0) - base_angle) > 1 or current.get("shoulder", 0) != _APPROACH_SHOULDER:
            print(f"[GRAB] [接近] 调整 base 到 {base_angle:.0f}°，shoulder 到 {_APPROACH_SHOULDER}°")
            self._move_to_pose({
                "base": base_angle,
                "shoulder": _APPROACH_SHOULDER,
                "elbow": current.get("elbow", _APPROACH_ELBOW_START),
                "wrist_flex": 180 - _APPROACH_SHOULDER - current.get("elbow", _APPROACH_ELBOW_START),
                "wrist_roll": _GRAB_WRIST_ROLL,
                "gripper": 90,
            }, "base 对准并过渡到 shoulder=20°", wait=1.5)
            self._refresh_arm_position()

        # ---- 修正：在 shoulder=20° 约束下（实测成功姿态），
        # elbow 从 135° 增大到 167° 时，末端高度逐渐降低（z 从 ~90mm 降到 ~23mm），
        # 水平位置从肩膀前方移到肩膀后方。这是正确的接近方向！
        #
        # 运动学校验（L1=116, L2=135, shoulder=20°）:
        #   elbow=150°: r≈-17mm, z≈63mm  (接近观察姿态)
        #   elbow=135°: r≈8mm,  z≈87mm   (末端观察，摄像头朝前下)
        #   elbow=160°: r≈-22mm, z≈44mm  (预抓取，接近桌面)
        #   elbow=167°: r≈-25mm, z≈23mm  (最终抓取姿态，桌面高度)
        #
        # 策略：固定 shoulder=20°，elbow 从 135° 逐步增大到 167° 完成接近和下降。

        approach_elbow = _APPROACH_ELBOW_START
        print(f"\n[GRAB] [接近] --- 末端观察姿态 (shoulder={_APPROACH_SHOULDER}°, elbow={approach_elbow}°, wrist_roll=-90°) ---")
        target = {
            "base": base_angle,
            "shoulder": _APPROACH_SHOULDER,
            "elbow": approach_elbow,
            "wrist_flex": 180 - _APPROACH_SHOULDER - approach_elbow,
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 90,
        }
        ok = self._move_to_pose(target, "末端观察姿态", wait=2.5)
        if not ok:
            print("[GRAB] [接近] [WARN] 姿态设定失败，尝试当前位置抓取")
        self._refresh_arm_position()

        chassis_forward_count = 0
        MAX_CHASSIS_FORWARD = 8  # 最多前进 8 次（用户反馈整体还能再前进 5cm，给足余量）
        for attempt in range(1, 8):
            print(f"\n[GRAB] [接近] --- 末端验证 第 {attempt}/7 次 ---")

            # Phase 2 强制使用末端摄像头，不回退主摄像头
            img_path = self.capture_end(f"接近_{attempt}")
            if img_path is None:
                print("[GRAB] [接近] 末端摄像头捕获失败，跳过本次")
                continue

            print("[GRAB] [接近] 使用 末端摄像头 分析")

            analysis = self._analyze_with_retry(img_path, target_object, use_end_camera=True)
            if analysis is None:
                continue

            if "已对准" in analysis:
                print(f"[GRAB] [接近] ✅ VLM(末端摄像头) 确认已对准")
                return True

            info = self.parse_analysis(analysis)
            if not info or not info["found"]:
                print("[GRAB] [接近] 未找到目标，继续尝试")
                continue

            dist = info.get("distance", info.get("size", "适中"))
            h = info["horizontal"]
            print(
                f"[GRAB] [接近] 状态: 水平={h}, 垂直={info['vertical']}, "
                f"距离={dist}, 可抓取={info['reachable']}"
            )

            # 水平偏移 -> 先微调 base（目标必须在画面中心）
            if h != "中心":
                offset_map = {"极左": -2, "偏左": -1, "偏右": 1, "极右": 2}
                offset = offset_map.get(h, 0)
                if offset != 0:
                    new_base = base_angle + offset
                    new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
                    print(f"[GRAB] [接近] 水平未对准({h}) -> base {base_angle:.0f}° -> {new_base:.0f}°")
                    self._move_to_pose({
                        "base": new_base,
                        "shoulder": _APPROACH_SHOULDER,
                        "elbow": approach_elbow,
                        "wrist_flex": 180 - _APPROACH_SHOULDER - approach_elbow,
                        "wrist_roll": _GRAB_WRIST_ROLL,
                        "gripper": 90,
                    }, f"base 微调至 {new_base:.0f}°", wait=1.5)
                    base_angle = new_base
                    continue  # 重新拍照验证

            # 判断是否已对准且可抓取 -> 也进入低空二次精调确认（避免高空斜视视差）
            if self.is_aligned_for_grab(info) and info["reachable"]:
                print(f"[GRAB] [接近] VLM 认为已对准，进入低空二次精调确认")
                ok, final_base = self._low_alt_fine_tune_loop(target_object, base_angle)
                return ok

            # 距离适中/较近/已接触/较近 -> 直接进入低空二次精调，禁用机械臂 r/z 调整
            if dist in ("较近", "已接触或超出", "适中"):
                print(f"[GRAB] [接近] 距离{dist}，切换到低空二次精调（禁用机械臂 r/z 调整）")
                ok, final_base = self._low_alt_fine_tune_loop(target_object, base_angle)
                return ok

            # 距离还远 -> 底盘前进（不再继续减小 elbow）
            if dist in ("太远", "较远"):
                if chassis_forward_count < MAX_CHASSIS_FORWARD:
                    forward_cm = 5 if dist == "太远" else 3
                    print(f"[GRAB] [接近] 距离仍{dist}，底盘前进 {forward_cm}cm "
                          f"({chassis_forward_count + 1}/{MAX_CHASSIS_FORWARD})")
                    self.chassis.forward_cm(forward_cm)
                    time.sleep(1.0)
                    chassis_forward_count += 1
                else:
                    print(f"[GRAB] [接近] 距离仍{dist}，底盘已前进 {MAX_CHASSIS_FORWARD} 次，尝试抓取")
                    return True
                continue

        print("[GRAB] [接近] 达到最大尝试次数，尝试抓取")
        return True

    def _fine_tune_loop(self, target_object: str, base_angle: float) -> str:
        """微调循环：最多 6 次，每次拍照后调整 base/r/z

        Returns:
            "aligned" — 已对准
            "ok"      — 未完全对准但距离适中，可以尝试抓取
            "too_far" — 距离太远，建议继续展开机械臂
        """
        for fine_attempt in range(1, 7):
            print(f"[GRAB] [运动学微调] --- 第 {fine_attempt}/6 次 ---")

            # Phase 2 强制使用末端摄像头，不回退主摄像头
            img_path = self.capture_end(f"接近微调_{fine_attempt}")
            if img_path is None:
                print("[GRAB] [运动学微调] 末端摄像头捕获失败，跳过本次")
                continue

            print("[GRAB] [运动学微调] 使用 末端摄像头 分析")
            analysis = self._analyze_with_retry(img_path, target_object, use_end_camera=True)
            if analysis and "已对准" in analysis:
                print(f"[GRAB] [运动学微调] ✅ VLM(末端摄像头) 确认已对准")
                return "aligned"

            info = self.parse_analysis(analysis) if analysis else None
            if not info or not info["found"]:
                continue

            dist = info.get("distance", info.get("size", "适中"))
            print(
                f"[GRAB] [运动学微调] 状态: 水平={info['horizontal']}, "
                f"垂直={info['vertical']}, 距离={dist}"
            )
            if self.is_aligned_for_grab(info) and info["reachable"]:
                print(f"[GRAB] [运动学微调] ✅ 已对准")
                return "aligned"

            # 如果微调中发现距离太远，建议继续展开
            if dist in ("太远", "较远"):
                print("[GRAB] [运动学微调] 距离仍远，建议继续展开机械臂")
                return "too_far"

            # 关键改动：微调阶段如果距离是"适中/较近"但还不够近，
            # 也触发底盘小步前进（而不是只调 arm，arm 每次只动3mm对4-5cm偏差没用）
            if dist in ("适中", "较近"):
                print("[GRAB] [运动学微调] 距离适中/较近，底盘前进 2cm 补距离")
                self.chassis.forward_cm(2)
                time.sleep(0.8)
                return "too_far"  # 返回 too_far 让外层循环继续拍照验证

            self._fine_tune_rz(info)

        return "ok"

    # ==================== Phase 2: 低空末端相机闭环精对准 ====================

    def _arm_approach_with_vision(self, target_object: str, base_angle: float = 0) -> tuple[bool, float]:
        """Phase 2: 低空末端相机闭环精对准（方案 B 核心）

        进入 Phase 2 的第一步，立刻将机械臂下探到低空预备姿态：
        shoulder=0, elbow=145, wrist_flex=35（夹爪水平俯视桌面）。
        这样可以彻底消除高空斜视视差（约 11cm 误差）。
        然后在低空姿态下启用末端相机闭环，水平偏差调 base，深度偏差调底盘。

        Returns:
            (success, final_base_angle)
        """
        print("\n[GRAB] ====== Phase 2: 低空末端相机闭环精对准 ======")

        self._check_end_camera()
        if not self._end_camera_available:
            print("[GRAB] [接近] ⚠️ 末端摄像头不可用，无法执行低空闭环")
            return False, base_angle

        print("[GRAB] [接近] 末端摄像头可用，立即切换到低空预备姿态")

        # 立刻驱动机械臂下探到低空预备姿态
        # shoulder=0, elbow=145, wrist_flex=180-145=35，夹爪水平俯视桌面
        low_alt_ready_pose = {
            "base": base_angle,
            "shoulder": 0,
            "elbow": 145,
            "wrist_flex": 35,
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 90,
        }
        self._move_to_pose(low_alt_ready_pose, "低空预备姿态 shoulder=0, elbow=145, wrist_flex=35", wait=2.0)

        # 启动末端相机闭环精调
        ok, final_base = self._low_alt_fine_tune_loop(target_object, base_angle)
        return ok, final_base

    def _low_alt_fine_tune_loop(self, target_object: str, base_angle: float) -> tuple[bool, float]:
        """低空末端相机闭环精调：One-way Gear Lock 单向降档 + 精调沉淀锁策略

        核心改进：
        - 引入持久化档位 self.current_gear，进入 Phase 2 初始化为 coarse。
        - 单向降档：仅当 coarse 时，若目标进入精调战区（距离适中/较近/已接触且水平偏差不大），
          一次性降档至 fine 并永久锁定，严禁因单帧 VLM 闪烁反弹回 coarse。
        - 步长完全由 self.current_gear 决定，不再每帧动态计算临时档位。
        - 安全底线：仅当目标丢失，或发生极左/极右完全反向的严重过冲时，才允许重置回 coarse。
        - 粗调降档首帧锁（Transit Lock）：上一轮是 coarse 且本轮满足放行条件时，
          强制用 fine 步长沉淀一轮，不直接放行。
        - 死区破局：仅在 fine 档下，连续 2 次水平状态相同可一次性脉冲放大 1.5 倍，
          但 base 上限 2.0°，绝不膨胀回粗调档。
        - 放行：已对准，或近距离时水平=中心且距离∈{较近,已接触或超出}。

        Returns:
            (success, final_base_angle)
        """
        print("\n[GRAB] [低空闭环] 进入低空末端相机闭环精调（One-way Gear Lock 单向降档）")
        print("[GRAB] [低空闭环] 粗调档: 远距离/大偏差 -> base=4.0°, chassis=5.0cm")
        print("[GRAB] [低空闭环] 精调档: 接近容错窗口 -> base=1.0°, chassis=1.5cm")
        print("[GRAB] [低空闭环] 单向降档锁: coarse -> fine 一次性切换，严禁无组织升档")
        print("[GRAB] [低空闭环] 放行: 已对准 或 (近距离时水平=中心 且 距离∈{较近,已接触或超出})")

        # 低空预备姿态：shoulder=0°, elbow=145°, wrist_flex=35°
        low_alt_pose = {
            "base": base_angle,
            "shoulder": 0,
            "elbow": 145,
            "wrist_flex": 35,
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 90,
        }
        print("[GRAB] [低空闭环] 保持低空预备姿态 shoulder=0°, elbow=145°, wrist_flex=35°")
        self._move_to_pose(low_alt_pose, "低空预备姿态", wait=1.5)

        current_base = base_angle
        self.current_gear = "coarse"  # 每次进入 Phase 2 都重置为 coarse
        MAX_ATTEMPTS = 12
        STALL_THRESHOLD = 2

        # 档位参数表
        GEAR_PARAMS = {
            "coarse": {"base_step": 4.0, "chassis_step": 5.0},
            "fine":   {"base_step": 1.0, "chassis_step": 1.5},
        }
        FINE_BASE_MAX = 2.0   # 精调档死区破局上限
        FINE_CHASSIS_MAX = 2.0  # 精调档死区破局上限

        # 状态追踪
        last_horizontal = None
        last_gear = "fine"  # 上一轮实际档位，用于粗调降档首帧锁
        stall_count = 0

        # 水平状态到方向符号的映射
        def horizontal_sign(h: str) -> int:
            return {"极左": -1, "偏左": -1, "中心": 0, "偏右": 1, "极右": 1}.get(h, 0)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"\n[GRAB] [低空闭环] --- 第 {attempt}/{MAX_ATTEMPTS} 次 ---")

            img_path = self.capture_end(f"低空闭环_{attempt}")
            if img_path is None:
                print("[GRAB] [低空闭环] 末端摄像头捕获失败，跳过本次")
                continue

            analysis = self._analyze_with_retry(img_path, target_object, use_end_camera=True)
            if analysis is None:
                continue

            if "已对准" in analysis:
                print("[GRAB] [低空闭环] ✅ VLM 确认已对准，退出精调")
                return True, current_base

            info = self.parse_analysis(analysis)
            if not info or not info["found"]:
                print("[GRAB] [低空闭环] 未找到目标，继续尝试")
                # 安全底线：目标丢失时允许重置回 coarse 以扩大搜索视野
                if self.current_gear == "fine":
                    print("[GRAB] [低空闭环] ⚠️ 目标丢失，重置为 coarse 档以扩大搜索视野")
                    self.current_gear = "coarse"
                continue

            h = info["horizontal"]
            dist = info.get("distance", "适中")
            aligned = info.get("aligned", False)

            # 安全底线：方向完全相反的严重过冲才允许重置回 coarse
            if (
                self.current_gear == "fine"
                and last_horizontal in ("极左", "极右")
                and h in ("极左", "极右")
                and last_horizontal != h
            ):
                print(
                    f"[GRAB] [低空闭环] ⚠️ 检测到方向完全相反的严重过冲"
                    f"({last_horizontal} -> {h})，重置为 coarse 档"
                )
                self.current_gear = "coarse"

            # ---------- One-way Gear Lock：单向降档，严禁无组织升档 ----------
            if self.current_gear == "coarse":
                if dist in ("适中", "较近", "已接触或超出") and h in ("中心", "偏左", "偏右"):
                    self.current_gear = "fine"
                    print(
                        "[GRAB] [低空闭环] 🔒 触发降档锁：已成功切入精调战区，"
                        "锁定 fine 档，后续严禁反弹回粗调大步长。"
                    )

            # ---------- 步长完全由持久档位决定，杜绝单帧闪烁导致的档位震荡 ----------
            base_step = GEAR_PARAMS[self.current_gear]["base_step"]
            chassis_step = GEAR_PARAMS[self.current_gear]["chassis_step"]

            # ---------- 近距离零横向容忍 + 粗调降档首帧锁 ----------
            close_distance = dist in ("较近", "已接触或超出", "太近")
            # 路径 A：VLM 明确已对准
            # 路径 B：近距离时水平必须绝对居中（视场小、机械容错极低）
            release_ok = aligned or (h == "中心" and close_distance)

            # 粗调降档首帧锁：上一轮是 coarse 且本轮满足放行条件，
            # 强制锁定 fine 档做一次沉淀微调，绝不直接放行。
            if release_ok and last_gear == "coarse":
                print(
                    "[GRAB] [低空闭环] 虽进入近距离/已对准，但处于粗调降档首帧，"
                    "触发[Transit Lock]，强制锁定 fine 档微调，暂不放行。"
                )
                self.current_gear = "fine"
                base_step = GEAR_PARAMS["fine"]["base_step"]
                chassis_step = GEAR_PARAMS["fine"]["chassis_step"]
            elif release_ok:
                print(
                    f"[GRAB] [低空闭环] ✅ 满足放行条件: 水平={h}, 距离={dist}, "
                    f"已对准={aligned}，档位={self.current_gear}，退出精调"
                )
                return True, current_base
            elif self.current_gear == "coarse":
                print(
                    f"[GRAB] [低空闭环] 🎯 当前[粗调档]: 远距离/大偏差，"
                    f"采用大步长奔袭 (base={base_step:.1f}°, chassis={chassis_step:.1f}cm)"
                )
            else:
                # 已在 fine 档：近距离但水平仍有偏差，持续精调
                if close_distance and h != "中心":
                    print(
                        f"[GRAB] [低空闭环] 虽进入近距离，但水平仍有偏差({h})，"
                        f"当前 fine 档微调，暂不放行。"
                    )
                else:
                    print(
                        f"[GRAB] [低空闭环] 🔍 当前[精调档]: 持续精细对准 "
                        f"(base={base_step:.2f}°, chassis={chassis_step:.1f}cm)"
                    )

            # ---------- 精调档死区破局（一次性脉冲，有上限，不转粗调档） ----------
            if self.current_gear == "fine" and h == last_horizontal and h != "中心":
                stall_count += 1
                print(
                    f"[GRAB] [低空闭环] 精调停滞: 连续 {stall_count}/"
                    f"{STALL_THRESHOLD} 次 {h}"
                )
                if stall_count >= STALL_THRESHOLD:
                    old_base = base_step
                    old_chassis = chassis_step
                    base_step = min(base_step * 1.5, FINE_BASE_MAX)
                    chassis_step = min(chassis_step * 1.5, FINE_CHASSIS_MAX)
                    print(
                        f"[GRAB] [低空闭环] ⚡ 精调死区破局脉冲: "
                        f"base {old_base:.2f}° -> {base_step:.2f}° (上限 {FINE_BASE_MAX:.1f}°), "
                        f"chassis {old_chassis:.1f}cm -> {chassis_step:.1f}cm (上限 {FINE_CHASSIS_MAX:.1f}cm)"
                    )
                    stall_count = 0  # 脉冲后重置，避免连续膨胀
            else:
                if stall_count > 0:
                    # 精调阶段一旦发生方向反转，说明当前步长已经越过最优解，
                    # 立即将 fine 档 base_step 减半，实现 Step-Halving 超精细收敛。
                    if self.current_gear == "fine":
                        last_sign = horizontal_sign(last_horizontal)
                        sign = horizontal_sign(h)
                        if last_sign != 0 and sign != 0 and last_sign != sign:
                            old_fine_step = GEAR_PARAMS["fine"]["base_step"]
                            new_fine_step = max(0.25, old_fine_step * 0.5)
                            GEAR_PARAMS["fine"]["base_step"] = new_fine_step
                            print(
                                f"[GRAB] [低空闭环] 🔄 检测到震荡横跳！"
                                f"base_step {old_fine_step:.2f}° -> {new_fine_step:.2f}°，"
                                f"主动衰减步长进行超精细收敛"
                            )
                    print(
                        f"[GRAB] [低空闭环] 水平状态变化 {last_horizontal} -> {h}，"
                        f"精调停滞计数清零，恢复精调档"
                    )
                stall_count = 0

            last_horizontal = h

            print(
                f"[GRAB] [低空闭环] 当前: 水平={h}, 距离={dist}, "
                f"档位={self.current_gear}, base_step={base_step:.2f}°, chassis_step={chassis_step:.1f}cm"
            )

            # 记录本轮实际档位，供下一轮 Transit Lock 判断
            last_gear = self.current_gear

            # ---------- 1. 水平偏差：微调 base ----------
            if h != "中心" and base_step > 0:
                direction = horizontal_sign(h)
                offset = direction * base_step
                new_base = current_base + offset
                new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
                print(
                    f"[GRAB] [低空闭环] 水平偏差({h}) -> base {current_base:.2f}° -> "
                    f"{new_base:.2f}° (步长={base_step:.2f}°)"
                )
                self._move_to_pose({
                    "base": new_base,
                    "shoulder": 0,
                    "elbow": 145,
                    "wrist_flex": 35,
                    "wrist_roll": _GRAB_WRIST_ROLL,
                    "gripper": 90,
                }, "低空闭环 base 微调", wait=1.0)
                current_base = new_base
                continue

            # ---------- 2. 深度/距离偏差：底盘微蹭 ----------
            # 使用当前档位的 chassis_step 作为基准，按距离远近微调
            if dist in ("太远", "较远"):
                move_cm = chassis_step
            elif dist == "适中":
                move_cm = max(chassis_step * 0.6, 1.0)
            elif dist == "较近":
                move_cm = max(chassis_step * 0.4, 0.8)
            else:  # 已接触或超出
                move_cm = 0.0

            move_cm = min(move_cm, chassis_step)  # 不超过当前档位步长

            if move_cm > 0:
                print(f"[GRAB] [低空闭环] 距离={dist}，底盘前进 {move_cm:.1f}cm")
                self.chassis.forward_cm(move_cm)
                time.sleep(0.8)
                continue
            else:
                print("[GRAB] [低空闭环] 距离已接触或超出，退出精调")
                return True, current_base

        print("[GRAB] [低空闭环] 达到最大尝试次数，退出精调")
        return True, current_base

    def _pre_grasp_validation_and_nudge(self, target_object: str, base_angle: float) -> tuple[bool, float]:
        """Phase 3 前置：终极校验锁 + Hail Mary 盲推

        在真正下压抓取前，强制做最后一次末端相机确认：
        - 若目标彻底偏离水平范围（极左/极右）或完全未找到，直接保护退出，避免空挥撞飞。
        - 否则，只要物体尚未与车体高度重合/顶到车体，一律执行 4cm 定量盲推，
          用底盘强制闭合低空视角下仍存在的约 4cm 物理盲区。

        Returns:
            (should_grasp, final_base_angle)
        """
        print("\n[GRAB] ====== Phase 3 前置：终极校验锁 + Hail Mary 盲推 ======")
        print("[GRAB] [终极校验] 策略：未顶到车体前，底盘一律前进 4cm 强喂夹爪")

        # 先回到低空预备姿态，确保末端相机视野正确
        self._move_to_pose({
            "base": base_angle,
            "shoulder": 0,
            "elbow": 145,
            "wrist_flex": 35,
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 90,
        }, "终极校验姿态 shoulder=0, elbow=145", wait=1.5)

        current_base = base_angle

        img_path = self.capture_end("终极校验")
        if img_path is None:
            print("[GRAB] [终极校验] ⚠️ 末端摄像头捕获失败，无法判断，保守放行抓取")
            return True, current_base

        analysis = self._analyze_with_retry(img_path, target_object, use_end_camera=True)
        if analysis is None:
            print("[GRAB] [终极校验] ⚠️ VLM 分析失败，保守放行抓取")
            return True, current_base

        info = self.parse_analysis(analysis)
        if not info or not info["found"]:
            print("[GRAB] [终极校验] ❌ 未找到目标物体，触发硬件保护退出，禁止空挥")
            return False, current_base

        h = info["horizontal"]
        dist = info.get("distance", "适中")
        print(f"[GRAB] [终极校验] 状态: 水平={h}, 距离={dist}")

        # 终极硬件保护拦截：目标彻底偏离水平范围
        if h in ("极左", "极右"):
            print(f"[GRAB] [终极校验] ❌ 水平严重偏离({h})，目标已脱离夹爪范围，触发硬件保护退出")
            return False, current_base

        # Hail Mary 盲推补偿（永不言弃）
        # 致命修正：低空视角下的"很近"仍有约 4cm 物理空隙，必须盲推！
        # 只有当物体已顶到车体/高度重合（已接触或超出）时才跳过。
        if dist == "已接触或超出":
            print(f"[GRAB] [终极校验] 距离={dist}，物体已顶到车体/高度重合，无需盲推，直接放行")
        else:
            nudge_cm = 4.0
            print(
                f"[GRAB] [终极校验] 距离={dist} 触发 Hail Mary 盲推！"
                f"物体处于盲区前沿，底盘强制突击前进 {nudge_cm:.1f}cm 强喂夹爪！"
            )

            # 水平残余补偿：边走边切向中心
            if h in ("偏左", "偏右"):
                h_sign = {"偏左": -1, "偏右": 1}.get(h, 0)
                base_offset = h_sign * 0.8  # 0.5°~1.0° 保守补偿
                new_base = current_base + base_offset
                new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
                print(
                    f"[GRAB] [终极校验] 水平残余补偿({h})："
                    f"base {current_base:.1f}° -> {new_base:.1f}°"
                )
                self._move_to_pose({
                    "base": new_base,
                    "shoulder": 0,
                    "elbow": 145,
                    "wrist_flex": 35,
                    "wrist_roll": _GRAB_WRIST_ROLL,
                    "gripper": 90,
                }, "Hail Mary 水平残余补偿", wait=0.5)
                current_base = new_base

            # 底盘突击前进 4cm，完成最后致命一击
            self.chassis.forward_cm(nudge_cm)
            time.sleep(0.8)

        print("[GRAB] [终极校验] ✅ 通过校验，无条件放行抓取")
        return True, current_base

    def _yolo_approach_loop(self, target_object: str, base_angle: float, retry_after_adjust: bool = False) -> tuple[bool, float]:
        """YOLO 像素闭环伺服控制（精细化调优版）

        固定 shoulder=20°, wrist_roll=-90°，在末端观察姿态（elbow=135°）下，
        通过 base 旋转（水平）和底盘移动（距离）进行闭环对准。
        垂直方向通过 _move_to_rz_relative 微调 z。

        控制逻辑（P-Controller + 动态增益 + 硬截断 + 防抖 + 反向回退）：
            err_x = cx - img_w/2   (水平像素误差)
            err_y = cy - img_h/2   (垂直像素误差)
            area_ratio = (w*h)/(img_w*img_h)  (面积占比)

            base_offset  = Kp_base(area_ratio) * err_x   (水平对准，动态增益)
            chassis_move = Kp_dist * area_err            (距离调整)
            dz           = Kp_z   * err_y                (垂直微调)

        注意：画面与物理方向一致（flip_horizontal=False）。
              err_x > 0（目标在画面右侧）= 物理右侧，需 base 减小（右转）对准。
              因此 base_offset = -Kp * err_x（反向）。

        Returns:
            (success, final_base_angle)
        """
        # 初始化到末端观察姿态
        self._move_to_pose({
            "base": base_angle,
            "shoulder": _END_CAM_SHOULDER,
            "elbow": _END_CAM_ELBOW,
            "wrist_flex": _END_CAM_WRIST_FLEX,
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 90,
        }, "YOLO 伺服初始姿态", wait=2.0)
        self._refresh_arm_position()

        # P-Controller 参数（经验值，可根据实际标定调整）
        KP_BASE = 0.06           # deg/pixel，水平误差到 base 角度
        KP_CHASSIS = 300.0       # cm*pixel_ratio，面积误差到底盘移动
        TARGET_AREA_RATIO = 0.25 # 目标占画面 25% 视为距离合适
        KP_Z = 0.08              # mm/pixel，垂直误差到 z 调整

        THRESH_X = 15            # 水平像素阈值
        THRESH_Y = 20            # 垂直像素阈值
        THRESH_AREA = 0.25       # 面积误差相对阈值

        MAX_BASE_STEP = 4.0      # 单次最大 base 调整角度（保留但不用于截断，改为硬截断 1.5°）
        MAX_CHASSIS_STEP = 5     # 单次最大底盘移动 cm
        MAX_Z_STEP = 8.0         # 单次最大 z 调整 mm

        # 新增：防抖等待时间（消除运动模糊）
        STABILIZATION_SLEEP = 1.0  # 秒

        det_none_count = 0
        last_base_offset = 0.0   # 记录上一次 base 调整量，用于反向回退

        for attempt in range(1, 16):
            print(f"\n[GRAB] [YOLO] --- 第 {attempt}/15 次 ---")
            img_path = self.capture_end(f"yolo_servo_{attempt}")
            if img_path is None:
                time.sleep(0.3)
                continue

            det = self._analyze_with_yolo(img_path, target_object)
            if det is None:
                det_none_count += 1
                print(f"[GRAB] [YOLO] 未检测到显著物体 (连续 {det_none_count} 次)")

                # === 反向回退机制（Anti-Lost Backtracking）===
                # 如果这是连续第一次丢失，且上一步做了较大的 base 调整，
                # 说明我们可能走过头了，往回退半步尝试把物体拉回视野
                if det_none_count == 1 and abs(last_base_offset) > 1.0:
                    backtrack = -last_base_offset * 0.5
                    new_base = base_angle + backtrack
                    new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
                    print(f"[GRAB] [YOLO] 触发反向回退: 上一步 base_offset={last_base_offset:+.1f}°，"
                          f"回退 {backtrack:+.1f}° -> {new_base:.1f}°")
                    self._move_to_pose({
                        "base": new_base,
                        "shoulder": _END_CAM_SHOULDER,
                        "elbow": _END_CAM_ELBOW,
                        "wrist_flex": _END_CAM_WRIST_FLEX,
                        "wrist_roll": _GRAB_WRIST_ROLL,
                        "gripper": 90,
                    }, "YOLO 反向回退", wait=STABILIZATION_SLEEP)
                    base_angle = new_base
                    last_base_offset = backtrack  # 更新记录为回退量
                    continue  # 重新拍照验证

                # 连续 2 次都未检测到，提前调整 elbow 后重试
                if det_none_count >= 2 and not retry_after_adjust:
                    print("[GRAB] [YOLO] 连续 2 次未检测到，提前中断并调整 elbow")
                    break
                continue

            # 成功检测到，重置丢失计数和上一次调整量
            det_none_count = 0
            last_base_offset = 0.0

            cx, cy = det["cx"], det["cy"]
            w, h = det["w"], det["h"]
            img_w, img_h = det["img_w"], det["img_h"]
            conf = det["confidence"]
            area_ratio = (w * h) / (img_w * img_h)

            err_x = cx - img_w / 2.0
            err_y = cy - img_h / 2.0

            print(f"[GRAB] [YOLO] 检测: conf={conf:.2f}, bbox=({w:.0f}x{h:.0f}), "
                  f"中心=({cx:.0f},{cy:.0f}), 画面={img_w}x{img_h}")
            print(f"[GRAB] [YOLO] 误差: err_x={err_x:+.0f}px, err_y={err_y:+.0f}px, "
                  f"面积占比={area_ratio:.3f} (目标={TARGET_AREA_RATIO:.3f})")

            # 1. 水平对准（优先级最高）
            if abs(err_x) > THRESH_X:
                # === 动态控制增益（Adaptive Kp）===
                # 目标越远（面积小）灵敏度越高，目标越近（面积大）进入"微雕模式"
                if area_ratio < 0.15:
                    kp = KP_BASE  # 0.06，保持原有灵敏度
                elif area_ratio >= 0.25:
                    kp = KP_BASE * 0.4  # 0.024，贴脸状态衰减为 0.4 倍
                else:
                    # 0.15 ~ 0.25 之间线性插值
                    kp = KP_BASE * (1.0 - 0.6 * (area_ratio - 0.15) / 0.10)

                base_offset = -kp * err_x

                # === 硬幅值限制（Clamping）===
                # 单次 Base 关节的调整角度绝对不超过 1.5°
                base_offset = max(-1.5, min(1.5, base_offset))

                # 保留最大步长限制作为安全兜底
                base_offset = max(-MAX_BASE_STEP, min(MAX_BASE_STEP, base_offset))

                new_base = base_angle + base_offset
                new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
                print(f"[GRAB] [YOLO] 水平调整: err_x={err_x:+.0f}px, kp={kp:.4f}, "
                      f"base_offset={base_offset:+.1f}° ({base_angle:.1f}° -> {new_base:.1f}°)")
                self._move_to_pose({
                    "base": new_base,
                    "shoulder": _END_CAM_SHOULDER,
                    "elbow": _END_CAM_ELBOW,
                    "wrist_flex": _END_CAM_WRIST_FLEX,
                    "wrist_roll": _GRAB_WRIST_ROLL,
                    "gripper": 90,
                }, "YOLO 水平微调", wait=STABILIZATION_SLEEP)
                base_angle = new_base
                last_base_offset = base_offset  # 记录本次调整量，用于可能的回退
                continue

            # 2. 距离调整（通过面积占比判断远近）
            area_err = TARGET_AREA_RATIO - area_ratio
            if abs(area_err) > TARGET_AREA_RATIO * THRESH_AREA:
                if area_err > 0:
                    # 面积太小 -> 太远 -> 底盘前进
                    forward_mm = abs(area_err) * KP_CHASSIS
                    forward_cm = max(1, min(MAX_CHASSIS_STEP, int(forward_mm / 10)))
                    print(f"[GRAB] [YOLO] 距离调整(远): 面积占比={area_ratio:.3f} < "
                          f"{TARGET_AREA_RATIO*(1-THRESH_AREA):.3f}, 底盘前进 {forward_cm}cm")
                    self.chassis.forward_cm(forward_cm)
                else:
                    # 面积太大 -> 太近 -> 底盘后退
                    backward_mm = abs(area_err) * KP_CHASSIS * 0.5
                    backward_cm = max(1, min(MAX_CHASSIS_STEP, int(backward_mm / 10)))
                    print(f"[GRAB] [YOLO] 距离调整(近): 面积占比={area_ratio:.3f} > "
                          f"{TARGET_AREA_RATIO*(1+THRESH_AREA):.3f}, 底盘后退 {backward_cm}cm")
                    self.chassis.backward_cm(backward_cm)
                # === 防抖等待 ===
                time.sleep(STABILIZATION_SLEEP)
                continue

            # 3. 垂直微调（优先级最低）
            if abs(err_y) > THRESH_Y:
                # err_y > 0: 目标在画面下半部分 -> 需要下降（z 减小）
                dz = -KP_Z * err_y
                dz = max(-MAX_Z_STEP, min(MAX_Z_STEP, dz))
                print(f"[GRAB] [YOLO] 垂直调整: err_y={err_y:+.0f}px -> dz={dz:+.1f}mm")
                self._move_to_rz_relative(dr=0, dz=dz, desc="YOLO 垂直微调", wait=STABILIZATION_SLEEP)
                continue

            # 所有指标满足 -> 收敛，再做一次最终底盘接近补偿（避免停太远）
            print(f"[GRAB] [YOLO] ✅ 像素伺服收敛 (attempt={attempt})")
            print(f"[GRAB] [YOLO]    err_x={err_x:.0f}px, err_y={err_y:.0f}px, "
                  f"area_ratio={area_ratio:.3f}")
            # 最终补偿：像素对准后通常还有 5-10cm 物理距离，前进 4cm 补差距
            print("[GRAB] [YOLO] 最终底盘补偿：前进 4cm")
            self.chassis.forward_cm(4)
            time.sleep(0.5)
            return True, base_angle

        print("[GRAB] [YOLO] 达到最大尝试次数，未收敛")
        # 如果绝大部分尝试都未检测到物体，尝试降低 elbow 10° 后重试一次
        if det_none_count >= 10 and not retry_after_adjust:
            adjusted_elbow = _END_CAM_ELBOW - 10
            adjusted_wrist_flex = 180 - _END_CAM_SHOULDER - adjusted_elbow
            print(f"[GRAB] [YOLO] 多次未检测到物体 ({det_none_count}/15)，"
                  f"尝试将 elbow 从 {_END_CAM_ELBOW}° 降至 {adjusted_elbow}° 后重试")
            self._move_to_pose({
                "base": base_angle,
                "shoulder": _END_CAM_SHOULDER,
                "elbow": adjusted_elbow,
                "wrist_flex": adjusted_wrist_flex,
                "wrist_roll": _GRAB_WRIST_ROLL,
                "gripper": 90,
            }, "elbow 下降 10° 后重试 YOLO", wait=1.5)
            return self._yolo_approach_loop(target_object, base_angle, retry_after_adjust=True)
        return False, base_angle

    # ==================== Phase 3: 抓取 ====================

    def _final_alignment_before_grasp(self, target_object: str, base_angle: float) -> float:
        """抓取前最终对准：用末端摄像头确认夹爪是否正对目标，偏了就调 base 1-2°

        用户反馈：偏右3°就会直接碰到目标，因此最终对准步长必须很小（1°）。
        Returns:
            精调后的 base 角度
        """
        if not self._end_camera_available:
            return base_angle

        print("\n[GRAB] [最终对准] 抓取前用末端摄像头做最后确认...")
        current_base = base_angle

        for attempt in range(1, 4):
            print(f"[GRAB] [最终对准] --- 第 {attempt}/3 次 ---")
            img_path = self.capture_end(f"最终对准_{attempt}")
            if img_path is None:
                continue

            analysis = self._analyze_with_retry(img_path, target_object, use_end_camera=True)
            if analysis is None:
                continue

            if "已对准" in analysis:
                print("[GRAB] [最终对准] ✅ 已对准")
                return current_base

            info = self.parse_analysis(analysis)
            if not info or not info["found"]:
                continue

            h = info["horizontal"]
            dist = info.get("distance", info.get("size", "适中"))
            print(f"[GRAB] [最终对准] 水平={h}, 垂直={info['vertical']}, 距离={dist}")

            # 如果距离还远，底盘前进补偿（用户反馈通常还差 10cm 左右，增大补偿）
            if dist in ("太远", "较远"):
                print("[GRAB] [最终对准] 距离仍远，底盘前进 5cm")
                self.chassis.forward_cm(5)
                time.sleep(0.6)
                continue
            elif dist == "适中":
                print("[GRAB] [最终对准] 距离适中，底盘前进 3cm")
                self.chassis.forward_cm(3)
                time.sleep(0.5)
                continue

            # 目标必须在画面中心才算对准（用户要求：目标在中间更好抓取）
            if h == "中心":
                print(f"[GRAB] [最终对准] ✅ 水平=中心，目标在画面正中")
                return current_base

            # 步长只有1°，避免矫枉过正
            # 画面与物理方向一致（flip_horizontal=False）
            h_map = {"极左": -2, "偏左": -1, "偏右": 1, "极右": 2}
            offset = h_map.get(h, 0)
            if offset == 0:
                return current_base

            new_base = current_base + offset
            new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
            print(f"[GRAB] [最终对准] 水平={h} -> base {current_base:.0f}° -> {new_base:.0f}° (步长1°)")
            self._move_to_pose({
                "base": new_base,
                "shoulder": _GRAB_SHOULDER,
                "elbow": _APPROACH_ELBOW_END,
                "wrist_flex": 180 - _GRAB_SHOULDER - _APPROACH_ELBOW_END,
                "wrist_roll": _GRAB_WRIST_ROLL,
                "gripper": 90,
            }, f"最终对准 base={new_base:.0f}°", wait=1.2)
            current_base = new_base

        print(f"[GRAB] [最终对准] 达到最大尝试次数，base={current_base:.0f}°")
        return current_base

    def _grasp_and_lift(self, target_object: str, base_angle: float = None) -> bool:
        """执行抓取：保持低空姿态 → 顺滑下压 → 夹紧 → 抬起

        前置条件：Phase 2 低空末端相机闭环已完成，机械臂处于 shoulder=0, elbow=145
        的低空预备姿态，底盘已将误差蹭完。
        本阶段只做简单的 elbow 顺滑增大下压，保持 base 不动。
        target_object 参数保留仅用于接口兼容，本阶段不再使用。
        """
        print("\n[GRAB] ====== Phase 3: 抓取 ======")

        current_base = base_angle
        if current_base is None:
            current = self.get_current_angles()
            current_base = current.get("base", 0)

        # 1/4 张开夹爪（保持低空预备姿态 shoulder=0, elbow=145, wrist_flex=35）
        print("[GRAB] [抓取] 1/4 张开夹爪")
        self._move_to_pose({
            "base": current_base,
            "shoulder": 0,
            "elbow": 145,
            "wrist_flex": 35,
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 90,
        }, "低空抓取姿态 - 张开夹爪", wait=1.0)

        # 2/4 顺滑下压（elbow 145° -> 167°），保持 base 不动
        # wrist_flex = 180 - elbow 保持夹爪水平切入
        print("[GRAB] [抓取] 2/4 顺滑下压 (elbow=145° -> 167°)")
        self._move_to_pose({
            "base": current_base,
            "shoulder": 0,
            "elbow": 167,
            "wrist_flex": 180 - 0 - 167,  # 13°
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 90,
        }, "顺滑下压 elbow=167°", wait=1.2)

        # 3/4 夹紧
        print("[GRAB] [抓取] 3/4 夹紧")
        self.arm.set_gripper(0)
        time.sleep(1.0)

        # 4/4 抬起（减小 elbow，保持 base 不动）
        print("[GRAB] [抓取] 4/4 抬起")
        self._move_to_pose({
            "base": current_base,
            "shoulder": 0,
            "elbow": 130,
            "wrist_flex": 180 - 0 - 130,  # 50°
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 0,
        }, "抬起 elbow=130°", wait=1.5)

        print("[GRAB] [抓取] 抓取序列完成")
        return True

    # ==================== Phase 4: 验证 ====================

    def _verify_grab(self, target_object: str) -> dict:
        """抓取后拍照验证（使用末端摄像头，夹爪视角）"""
        print(f"\n[GRAB] ====== Phase 4: 抓取后视觉验证（末端摄像头） ======")

        # 优先使用末端摄像头，回退到机身摄像头
        if self._end_camera_available:
            img_path = self.capture_end("verify")
            camera_name = "末端摄像头"
        else:
            img_path = self.capture("verify")
            camera_name = "机身摄像头"

        if img_path is None:
            return {"success": False, "message": "验证拍照失败"}

        # 末端摄像头视角：近距离看夹爪和目标
        prompt = (
            f"这张图来自机械臂末端夹爪上的摄像头，是近距离视角。"
            f"请仔细观察夹爪的两个手指之间，是否确实夹住了'{target_object}'。"
            f"只回复以下三种之一：已夹住 / 未夹住 / 未夹住但物体在附近"
        )
        try:
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

    def run(self, target_object: str = "一包纸巾") -> dict:
        """
        运行自主抓取主循环（运动学版）

        流程:
            Phase 0: 机械臂回到观察姿态
            Phase 1: 底盘粗定位（只调 base + 底盘，机械臂不动）
            Phase 2: 机械臂展开接近（运动学精确前伸 + 视觉闭环）
            Phase 3: 下降、夹紧、抬起
            Phase 4: 视觉验证
        """
        print(f"\n{'='*60}")
        print(f"[GRAB] 开始自主抓取: {target_object}")
        print(f"[GRAB] 末端摄像头: {'已启用' if self.use_end_camera else '未启用'}")
        print(f"[GRAB] 运动学: L1={_ARM_CFG.upper_arm_length}mm, L2={_ARM_CFG.forearm_length}mm")
        print(f"{'='*60}")

        # Phase 0: 机械臂回到观察姿态
        if not self._move_to_observation_pose():
            return {"success": False, "message": "机械臂无法回到观察姿态"}

        # Phase 1: 底盘粗定位
        info = self._chassis_coarse_alignment(target_object)
        if info is None:
            return {"success": False, "message": "底盘粗定位失败，未找到目标"}

        # Phase 1 不再检查 reachable，即使"太近"也放行进入 Phase 2
        base_angle = self.get_current_angles().get("base", 0)

        # Phase 2: 低空末端相机闭环精对准
        approached, final_base = self._arm_approach_with_vision(target_object, base_angle=base_angle)
        if not approached:
            return {"success": False, "message": "机械臂接近失败"}

        # Phase 3 前置：终极校验锁 + Hail Mary 盲推
        should_grasp, final_base = self._pre_grasp_validation_and_nudge(target_object, final_base)
        if not should_grasp:
            return {"success": False, "message": "终极校验触发硬件保护，目标脱离夹爪范围"}

        # Phase 3: 抓取
        grasped = self._grasp_and_lift(target_object, base_angle=final_base)
        if not grasped:
            return {"success": False, "message": "抓取序列执行失败"}

        # Phase 4: 视觉验证
        return self._verify_grab(target_object)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HomeBot 自主抓取工作流（运动学版）")
    parser.add_argument("--ip", default=config.ROBOT_IP, help="机器人 IP")
    parser.add_argument("--video-port", type=int, default=config.VIDEO_PORT, help="身上摄像头端口")
    parser.add_argument("--end-video-port", type=int, default=config.END_VIDEO_PORT, help="末端摄像头端口")
    parser.add_argument("--arm-port", type=int, default=config.ARM_PORT, help="机械臂端口")
    parser.add_argument("--max-attempts", type=int, default=10, help="总尝试次数")
    parser.add_argument("--no-end-camera", action="store_true", help="禁用机械臂末端摄像头（默认已启用）")
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
