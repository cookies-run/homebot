#!/usr/bin/env python3
"""
Auto Grab Workflow - LLM 引导的自主抓取（运动学版，YOLO 加速版）

正确流程：
    1. 机械臂回到"观察姿态"（rest_position，摄像头与桌面平齐）
    2. 底盘粗定位：只调 base（旋转）+ 底盘前进/后退，机械臂保持不动！
    3. 机械臂展开接近：使用逆运动学精确前伸，每次移动后视觉验证
    4. 下降、夹紧、抬起

核心改进（方案 B + YOLO 加速）：
    - 引入 ArmKinematics 运动学系统
    - 用 (r, z) 坐标控制末端位置，代替固定关节步进
    - 自动计算 wrist_flex = 180 - shoulder - elbow，保持末端方向一致
    - 实时追踪末端在空间中的实际位置
    - Phase 1/2 优先使用本地 YOLO 检测，显著降低 VLM 调用次数，提升速度

用法:
    python grab_optimized.py              # 默认抓取纸巾
    python grab_optimized.py --target "一个苹果"

要求:
    - 机器人摄像头已启动并发布到 ZeroMQ (默认端口 5560)
    - 机械臂服务已启动 (默认端口 5557)
    - VLM 视觉分析可用 (MiniMax 优先，Phase 3/4 仍需使用)
    - 本地 YOLO 模型已下载（Phase 1/2 加速用）

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

# 引入 YOLO（本地视觉伺服，Phase 1/2 像素闭环）
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
    自主抓取工作流（运动学版，YOLO 加速版）

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

        # ---- 加载 YOLO 模型（Phase 1/2 像素伺服）----
        self._init_yolo()

    # ==================== YOLO 视觉伺服 ====================

    def _init_yolo(self):
        """加载本地 YOLO 模型（用于 Phase 1/2 像素闭环伺服）

        加载顺序：
        1. YOLO-World（开放词汇，可检测任意文本描述物体）
        2. COCO 预训练 YOLO（速度快，固定 80 类）
        如果都失败，Phase 1/2 将回退到 VLM。
        """
        self.yolo_model = None
        self.yolo_world_model = None
        self._yolo_world_last_classes = None

        if not _YOLO_AVAILABLE:
            print("[GRAB] [WARN] ultralytics 未安装，Phase 1/2 将回退到 VLM")
            return

        # ---- 1. 尝试加载 YOLO-World（开放词汇，支持文本提示）----
        # l = large（精度优先），m = medium（平衡），s = small（速度优先）
        # 经测试 s 版对纸巾等小目标召回率偏低，默认优先使用 l 版
        models_dir = os.path.join(os.path.dirname(__file__), '../../../software/models')
        world_candidates = [
            ("yolov8l-worldv2.pt", os.path.join(models_dir, "yolov8l-worldv2.pt")),
            ("yolov8m-worldv2.pt", os.path.join(models_dir, "yolov8m-worldv2.pt")),
            ("yolov8s-worldv2.pt", os.path.join(models_dir, "yolov8s-worldv2.pt")),
        ]
        for name, path in world_candidates:
            try:
                print(f"[GRAB] [YOLO] 正在加载 YOLO-World 模型: {name}")
                if os.path.exists(path):
                    self.yolo_world_model = YOLO(path)
                else:
                    # 如果 models 目录没有，让 ultralytics 自动下载到当前目录
                    self.yolo_world_model = YOLO(name)
                print(f"[GRAB] [YOLO] YOLO-World 模型已加载: {name}")
                break
            except Exception as e:
                print(f"[GRAB] [WARN] 加载 YOLO-World 模型失败 {name}: {e}")
                continue

        # ---- 2. 兜底加载 COCO 预训练 YOLO ----
        coco_candidates = [
            os.path.join(os.path.dirname(__file__), '../../../software/models/yolo26n.pt'),
            os.path.join(os.path.dirname(__file__), '../../../software/models/yolo11n.pt'),
        ]
        for path in coco_candidates:
            if os.path.exists(path):
                try:
                    self.yolo_model = YOLO(path)
                    print(f"[GRAB] YOLO 模型已加载: {path}")
                    return
                except Exception as e:
                    print(f"[GRAB] [WARN] 加载 YOLO 模型失败 {path}: {e}")
                    continue

        if self.yolo_world_model is None and self.yolo_model is None:
            print("[GRAB] [WARN] 未找到可用 YOLO 模型，Phase 1/2 将回退到 VLM")

    # ==================== YOLO 目标检测（新增） ====================

    def _build_yolo_world_classes(self, target_object: str) -> list[str]:
        """把目标描述转成 YOLO-World 可识别的英文提示词列表"""
        cn_to_en = {
            "纸巾": [
                "a pack of tissues",
                "white tissue box",
                "paper napkin",
                "facial tissue",
                "tissue",
                "paper",
            ],
            "纸": ["paper", "tissue", "napkin", "paper sheet"],
            "苹果": ["apple", "red apple", "green apple"],
            "瓶子": ["bottle", "plastic bottle", "glass bottle"],
            "杯子": ["cup", "mug", "paper cup", "plastic cup"],
            "球": ["ball", "sports ball"],
            "香蕉": ["banana", "yellow banana"],
            "橘子": ["orange", "mandarin orange", "tangerine"],
            "手机": ["cell phone", "mobile phone", "smartphone"],
            "遥控器": ["remote", "remote control", "TV remote"],
            "钥匙": ["key", "keys"],
            "笔": ["pen", "ballpoint pen"],
            "书": ["book", "notebook"],
            "盒子": ["box", "cardboard box", "package box"],
            "袋子": ["bag", "plastic bag", "paper bag"],
            "零食": ["snack", "snack bag", "food package"],
        }

        raw = target_object.strip().lower()
        for prefix in ["一个", "一包", "一张", "面前的", "这个", "那个"]:
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()

        # 中文目标优先用映射；否则把原始字符串也当作提示词
        prompts = cn_to_en.get(raw, [raw])
        # 去重并保持顺序
        seen = set()
        unique = []
        for p in prompts:
            if p and p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _yolo_detect_target(self, image_path: str, target_object: str) -> tuple | None:
        """
        使用本地 YOLO 检测目标物体，返回归一化信息。
        优先使用 YOLO-World（开放词汇），失败再回退到 COCO YOLO。

        Returns:
            (target_bbox, center_x_ratio, area_ratio) 或 None
            target_bbox: (x1, y1, x2, y2) 像素坐标
            center_x_ratio: 目标中心 x 占画面宽度的比例（0~1）
            area_ratio: 目标框面积占画面总面积的比例
        """
        import PIL.Image as Image

        if self.yolo_world_model is None and self.yolo_model is None:
            return None

        try:
            with Image.open(image_path) as img_obj:
                img_w, img_h = img_obj.size
        except Exception as e:
            print(f"[GRAB] [YOLO] 无法读取图像: {image_path}, {e}")
            return None

        # 通用过滤阈值（YOLO-World 和 COCO YOLO 共用）
        conf_threshold = 0.15
        min_area_ratio = 0.001
        max_area_ratio = 0.95

        # ---------- 1. 优先尝试 YOLO-World（开放词汇，支持任意文本目标） ----------
        if self.yolo_world_model is not None:
            try:
                classes = self._build_yolo_world_classes(target_object)
                # 只在类别变化时才调用 set_classes，避免重复开销
                if self._yolo_world_last_classes != classes:
                    print(f"[GRAB] [YOLO-World] 设置检测类别: {classes}")
                    self.yolo_world_model.set_classes(classes)
                    self._yolo_world_last_classes = classes

                results = self.yolo_world_model(image_path, verbose=False)
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        best = self._pick_best_yolo_box(
                            boxes, img_w, img_h,
                            conf_threshold=conf_threshold,
                            min_area_ratio=min_area_ratio,
                            max_area_ratio=max_area_ratio,
                        )
                        if best is not None:
                            print(f"[GRAB] [YOLO-World] 检测到目标: conf={best['conf']:.2f}, "
                                  f"center_x_ratio={best['cx_ratio']:.3f}, area_ratio={best['area_ratio']:.3f}")
                            return best["bbox"], best["cx_ratio"], best["area_ratio"]
                        print("[GRAB] [YOLO-World] 未通过过滤条件")
            except Exception as e:
                print(f"[GRAB] [YOLO-World] 推理异常: {e}")

        # ---------- 2. 回退到 COCO YOLO（固定 80 类） ----------
        if self.yolo_model is None:
            return None

        print("[GRAB] [YOLO-World] 未检测到目标，回退到 COCO YOLO")
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

        best = self._pick_best_yolo_box(
            boxes, img_w, img_h,
            conf_threshold=conf_threshold,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            target_object=target_object,
        )
        if best is None:
            print(f"[GRAB] [YOLO] 无有效检测框（共 {len(boxes)} 个被过滤）")
            return None

        print(f"[GRAB] [YOLO] COCO 匹配: {best.get('class_name', 'unknown')} conf={best['conf']:.2f}, "
              f"center_x_ratio={best['cx_ratio']:.3f}, area_ratio={best['area_ratio']:.3f}")
        return best["bbox"], best["cx_ratio"], best["area_ratio"]

    def _pick_best_yolo_box(
        self,
        boxes,
        img_w: int,
        img_h: int,
        conf_threshold: float,
        min_area_ratio: float,
        max_area_ratio: float,
        target_object: str | None = None,
    ) -> dict | None:
        """从 YOLO 检测框中筛选并挑选最佳框。

        对于 COCO YOLO，会尝试按类别名匹配 target_object；
        匹配失败时 fallback 到最高置信度框。
        """
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # COCO 模型才有 cls，YOLO-World 的 classes 由 set_classes 决定
        has_cls = hasattr(boxes, "cls") and boxes.cls is not None
        cls_ids = boxes.cls.cpu().numpy().astype(int) if has_cls else [None] * len(confs)

        target_classes = []
        if target_object and self.yolo_model is not None:
            raw = target_object.strip().lower()
            for prefix in ["一个", "一包", "一张", "面前的", "这个", "那个"]:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):].strip()
            cn_to_en = {
                "纸巾": ["tissue", "paper", "toilet paper", "book"],
                "纸": ["tissue", "paper", "toilet paper", "book"],
                "苹果": ["apple"],
                "瓶子": ["bottle"],
                "杯子": ["cup"],
                "球": ["sports ball"],
                "香蕉": ["banana"],
                "橘子": ["orange"],
                "手机": ["cell phone"],
            }
            target_classes = cn_to_en.get(raw, [raw])

        valid_boxes = []
        for idx, (box, conf_val, cls_id) in enumerate(zip(xyxy, confs, cls_ids)):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            area_ratio = (bw * bh) / (img_w * img_h)
            class_name = ""
            if cls_id is not None and self.yolo_model is not None:
                class_name = self.yolo_model.names.get(cls_id, "").lower()

            if conf_val < conf_threshold:
                print(f"[GRAB] [YOLO] raw #{idx}: {class_name} conf={conf_val:.2f} area={area_ratio:.4f} -> 置信度低于 {conf_threshold}")
                continue
            if area_ratio > max_area_ratio or area_ratio < min_area_ratio:
                print(f"[GRAB] [YOLO] raw #{idx}: {class_name} conf={conf_val:.2f} area={area_ratio:.4f} -> 面积超出 [{min_area_ratio}, {max_area_ratio}]")
                continue

            cx_ratio = ((x1 + x2) / 2.0) / img_w
            valid_boxes.append({
                "idx": idx,
                "conf": float(conf_val),
                "area_ratio": area_ratio,
                "cx_ratio": cx_ratio,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "class_name": class_name,
            })
            print(f"[GRAB] [YOLO] raw #{idx}: {class_name} conf={conf_val:.2f} area={area_ratio:.4f} -> 有效")

        if not valid_boxes:
            return None

        # COCO 模型优先按类别名匹配
        if target_classes:
            matched = []
            for b in valid_boxes:
                for tc in target_classes:
                    if tc in b["class_name"] or b["class_name"] in tc:
                        matched.append(b)
                        break
            if matched:
                best = max(matched, key=lambda x: x["conf"])
                print(f"[GRAB] [YOLO] 类别匹配成功: {best['class_name']} conf={best['conf']:.2f}")
                return best

        # fallback：选最高置信度框
        best = max(valid_boxes, key=lambda x: x["conf"])
        print(f"[GRAB] [YOLO] 无类别匹配，fallback 最高置信度: {best.get('class_name', 'unknown')} conf={best['conf']:.2f}")
        return best

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

    # ==================== Phase 1: 底盘粗定位（YOLO 加速版） ====================

    def _chassis_coarse_alignment(self, target_object: str) -> dict:
        """底盘粗定位闭环：base 锁死，优先使用 YOLO 快速距离估计，回退到 VLM。

        放行条件：
        - 当 YOLO area_ratio >= 0.45 或 VLM 明确返回"太近"时，视为已到达最佳逼近点，底盘停机放行进入 Phase 2。
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
        print("[GRAB] 放行条件: YOLO area_ratio>=0.45 或 距离=太近，或探测到【较近 -> 太近】过度接近")
        print("[GRAB] 纯净策略: 基于物理距离的变步长逼近 + 过度接近保护，零尾端补偿")

        # base 锁死为观察姿态角度（默认 -90°）
        current_base = OBSERVATION_POSE.get("base", -90)
        print(f"[GRAB] [底盘粗定位] base 锁死={current_base:.0f}°")
        last_info = None
        last_distance = None  # 状态历史记忆，捕捉 较近 -> 太近 的过渡
        attempts = min(6, self.max_attempts)  # 最多 6 次

        for attempt in range(1, attempts + 1):
            print(f"\n[GRAB] [底盘粗定位] --- 第 {attempt}/{attempts} 次 ---")

            img_path = self.capture(f"底盘粗定位_{attempt}")
            if img_path is None:
                continue

            # ====== YOLO 快速检测优先 ======
            yolo_result = self._yolo_detect_target(img_path, target_object)
            if yolo_result is not None:
                target_bbox, center_x_ratio, area_ratio = yolo_result
                print(f"[GRAB] [底盘粗定位] YOLO 检测成功: area_ratio={area_ratio:.3f}, center_x_ratio={center_x_ratio:.3f}")

                # 基于面积占比的距离判断（base 保持锁死）
                if area_ratio < 0.15:
                    print(f"[GRAB] [底盘粗定位] YOLO: area_ratio={area_ratio:.3f} < 0.15，目标太远，底盘前进 20cm")
                    self.chassis.forward_cm(20)
                    time.sleep(0.5)
                    last_distance = "太远"
                    continue
                elif area_ratio < 0.45:
                    print(f"[GRAB] [底盘粗定位] YOLO: area_ratio={area_ratio:.3f} 在 [0.15, 0.45)，底盘前进 10cm")
                    self.chassis.forward_cm(10)
                    time.sleep(0.5)
                    last_distance = "较远"
                    continue
                else:
                    print(f"[GRAB] [底盘粗定位] YOLO: area_ratio={area_ratio:.3f} >= 0.45，目标已足够近，放行 Phase 2")
                    return {
                        "found": True,
                        "horizontal": "中心",
                        "vertical": "中心",
                        "distance": "较近",
                        "reachable": True,
                        "aligned": False,
                        "visible": True,
                    }

            # ====== YOLO 失败，回退到 VLM ======
            print("[GRAB] [底盘粗定位] YOLO 未检测到目标，回退到 VLM 分析")
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

            # 基于物理距离的变步长推进（更激进的版本）
            dist_move_map = {
                "太远": 20,
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

    # ==================== Phase 2: 机械臂展开接近（YOLO 像素伺服版） ====================

    def _arm_approach_with_vision(self, target_object: str, base_angle: float = 0) -> tuple[bool, float]:
        """Phase 2: 低空末端相机闭环精对准（YOLO 像素伺服版）

        进入 Phase 2 的第一步，立刻将机械臂下探到低空预备姿态：
        shoulder=0, elbow=145, wrist_flex=35（夹爪水平俯视桌面）。
        然后在低空姿态下启用末端相机闭环，优先使用 YOLO 做像素级水平对准，
        回退到 VLM 分析。

        Returns:
            (success, final_base_angle)
        """
        print("\n[GRAB] ====== Phase 2: 低空末端相机闭环精对准（YOLO 像素伺服） ======")

        self._check_end_camera()
        if not self._end_camera_available:
            print("[GRAB] [接近] ⚠️ 末端摄像头不可用，无法执行低空闭环")
            return False, base_angle

        print("[GRAB] [接近] 末端摄像头可用，立即切换到低空预备姿态")

        # 立刻驱动机械臂下探到低空预备姿态
        low_alt_ready_pose = {
            "base": base_angle,
            "shoulder": 0,
            "elbow": 145,
            "wrist_flex": 35,
            "wrist_roll": _GRAB_WRIST_ROLL,
            "gripper": 90,
        }
        self._move_to_pose(low_alt_ready_pose, "低空预备姿态 shoulder=0, elbow=145, wrist_flex=35", wait=2.0)

        # YOLO 像素伺服闭环
        KP_BASE = 8.0  # 度每全帧误差
        max_attempts = 4
        current_base = base_angle
        last_error_x = None

        for attempt in range(1, max_attempts + 1):
            print(f"\n[GRAB] [Phase2] --- 第 {attempt}/{max_attempts} 次 ---")

            img_path = self.capture_end(f"phase2_{attempt}")
            if img_path is None:
                print("[GRAB] [Phase2] 末端摄像头捕获失败，跳过本次")
                continue

            # ====== YOLO 像素伺服优先 ======
            yolo_result = self._yolo_detect_target(img_path, target_object)
            if yolo_result is not None:
                target_bbox, center_x_ratio, area_ratio = yolo_result
                error_x = center_x_ratio - 0.5
                print(f"[GRAB] [Phase2] YOLO 检测成功: center_x_ratio={center_x_ratio:.3f}, error_x={error_x:+.3f}, area_ratio={area_ratio:.3f}")

                # 死区判定：误差在 ±5% 以内视为已对准
                if abs(error_x) <= 0.05:
                    print("[GRAB] [Phase2] 水平已对准（死区），放行抓取")
                    return True, current_base

                # 过零点检测：误差符号变化说明已越过中心，视为基本对准
                if last_error_x is not None and (error_x * last_error_x < 0):
                    print("[GRAB] [Phase2] 过零点检测触发，判定为基本对准，放行抓取")
                    return True, current_base

                # 计算 base 偏移量（P-Controller，带最小步长限制）
                base_offset = -error_x * KP_BASE
                # 钳制绝对值到 [0.25, 2.0] 度，避免过冲和死区
                if abs(base_offset) < 0.25:
                    base_offset = 0.25 if base_offset >= 0 else -0.25
                if abs(base_offset) > 2.0:
                    base_offset = 2.0 if base_offset >= 0 else -2.0

                new_base = current_base + base_offset
                new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
                print(f"[GRAB] [Phase2] YOLO 水平调整: error_x={error_x:+.3f}, base_offset={base_offset:+.2f}° ({current_base:.1f}° -> {new_base:.1f}°)")
                self._move_to_pose({
                    "base": new_base,
                    "shoulder": 0,
                    "elbow": 145,
                    "wrist_flex": 35,
                    "wrist_roll": _GRAB_WRIST_ROLL,
                    "gripper": 90,
                }, "Phase2 YOLO base 微调", wait=1.0)
                current_base = new_base
                last_error_x = error_x
                continue

            # ====== YOLO 失败，回退到 VLM ======
            print("[GRAB] [Phase2] YOLO 未检测到目标，回退到 VLM 对准分析")
            analysis = self.analyze_alignment(img_path, target_object, use_end_camera=True)
            if analysis is None:
                print("[GRAB] [Phase2] VLM 分析失败，继续尝试")
                continue

            if "已对准" in analysis or self.is_aligned_for_grab(self.parse_analysis(analysis)):
                print("[GRAB] [Phase2] VLM 确认已对准，放行抓取")
                return True, current_base

            info = self.parse_analysis(analysis)
            if not info or not info["found"]:
                print("[GRAB] [Phase2] VLM 未找到目标，继续尝试")
                continue

            h = info["horizontal"]
            print(f"[GRAB] [Phase2] VLM 状态: 水平={h}, 垂直={info['vertical']}, 距离={info.get('distance', '适中')}")

            # VLM 水平偏差调整 base
            if h != "中心":
                offset_map = {"极左": -2, "偏左": -1, "偏右": 1, "极右": 2}
                offset = offset_map.get(h, 0)
                if offset != 0:
                    new_base = current_base + offset
                    new_base = self.clamp(new_base, _JOINT_LIMITS["base"][0], _JOINT_LIMITS["base"][1])
                    print(f"[GRAB] [Phase2] VLM 水平调整({h}): base {current_base:.1f}° -> {new_base:.1f}°")
                    self._move_to_pose({
                        "base": new_base,
                        "shoulder": 0,
                        "elbow": 145,
                        "wrist_flex": 35,
                        "wrist_roll": _GRAB_WRIST_ROLL,
                        "gripper": 90,
                    }, "Phase2 VLM base 微调", wait=1.0)
                    current_base = new_base
                continue

        # 循环结束未返回：保守放行，不失败 Phase 2
        print("[GRAB] [Phase2] 达到最大尝试次数，保守放行进入 Phase 3")
        return True, current_base

    # ==================== Phase 3: 抓取（与原版完全一致） ====================

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

    # ==================== Phase 4: 验证（与原版完全一致） ====================

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
        运行自主抓取主循环（运动学版，YOLO 加速版）

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

    parser = argparse.ArgumentParser(description="HomeBot 自主抓取工作流（运动学版，YOLO 加速版）")
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
