# -*- coding: utf-8 -*-
"""系统配置 - 机器人硬件与系统相关配置

包含摄像头、机械臂、底盘、ZeroMQ、日志、语音引擎、手柄、人体跟随、电池等
硬件/系统层配置。AI 模型(LLM/TTS/Vision)配置见 configs.ai_config。
"""
from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """摄像头配置

    macOS 上推荐使用 device_name 或 unique_id（最稳定），VisionService 会
    自动使用 AVFoundation 原生驱动，绕过 OpenCV 易变的整数索引。

    示例设备名称（通过 `python -m services.vision_service --list-cameras` 查看）：
        "1080P USB Camera"  -> 外接 USB 摄像头 (1920x1080)
        "USB摄像头"         -> 末端/机械臂摄像头 (1280x720)
        "FaceTime高清相机"  -> 笔记本自带摄像头

    unique_id 是硬件级稳定标识，插拔不变；优先级：unique_id > device_name > device_id。
    """
    device_id: int = 0   # OpenCV 设备索引（Linux/Windows 使用；macOS 仅作 fallback）
    device_name: str = "1080P USB Camera"  # 按名称查找摄像头（主摄像头），非空时优先于 device_id
    unique_id: str = ""    # macOS AVFoundation 稳定硬件标识（最优先）
    device_path: str = ""  # Windows DirectShow/MSMF 稳定设备路径；macOS 也可复用为 uniqueID 的替代。
                           # 匹配优先级：device_path > unique_id > device_name > device_id
    width: int = 1920     # 摄像头原始分辨率
    height: int = 1080
    fps: int = 30


@dataclass
class ArmConfig:
    """机械臂配置"""
    serial_port: str = "auto"  # 与底盘共用串口，设为 "auto" 自动检测
    baudrate: int = 1000000
    # 舵机ID映射 (1-6号关节)
    base_id: int = 1
    shoulder_id: int = 2
    elbow_id: int = 3
    wrist_flex_id: int = 4
    wrist_roll_id: int = 5
    gripper_id: int = 6
    # 连杆长度 (mm) 人工设置，AI勿动
    upper_arm_length: float = 116.0  # 大臂长度 (L1) - 对应 CAD 图 shoulder→elbow
    forearm_length: float = 135.0    # 小臂长度 (L2) - 对应 CAD 图 elbow→wrist_flex
    # 关节角度限制 (度) 人工设置，AI勿动
    joint_limits: dict = field(default_factory=lambda: {
        "base": (-180, 180),
        "shoulder": (0, 180),
        "elbow": (0, 180),
        "wrist_flex": (-90, 90),
        "wrist_roll": (-180, 180),
        "gripper": (0, 90),
    })
    # 默认速度/加速度
    default_speed: int = 1000
    default_acc: int = 50
    # 休息位置/待机位置 (度) - 服务启动时自动恢复到此位置 人工设置，AI勿动
    rest_position: dict = field(default_factory=lambda: {
        "base": -90,         # J1: 基座旋转
        "shoulder": 0,   # J2: 肩关节（自然下垂）
        "elbow": 150,       # J3: 肘关节
        "wrist_flex": 30,   # J4: 腕关节屈伸
        "wrist_roll": 0,   # J5: 腕关节旋转
        "gripper": 45,     # J6: 夹爪（半开）
    })


@dataclass
class ChassisConfig:
    """底盘配置 - 从机器人配置文件读取"""
    # 串口配置（Windows: COM3, Linux: /dev/ttyUSB0, macOS: /dev/tty.usbmodemxxx）
    # 设为 "auto" 自动检测，或硬编码具体路径
    serial_port: str = "auto"
    baudrate: int = 1000000
    
    # 舵机ID映射
    left_front_id: int = 9
    right_front_id: int = 8
    rear_id: int = 7
    
    # 物理参数
    wheel_radius: float = 0.08      # 轮子半径 (m)
    chassis_radius: float = 0.18     # 底盘半径 (m)
    
    # 运动限制
    max_linear_speed: float = 0.5    # 最大线速度 (m/s)
    max_angular_speed: float = 1.0   # 最大角速度 (rad/s)
    default_wheel_speed: int = 3250  # 舵机最大速度
    
    # ZeroMQ地址
    service_addr: str = "tcp://*:5556"


@dataclass
class ZMQConfig:
    """ZeroMQ网络配置"""
    chassis_service_addr: str = "tcp://*:5556"
    arm_service_addr: str = "tcp://*:5557"      # 机械臂服务地址
    vision_pub_addr: str = "tcp://*:5560"
    speech_service_addr: str = "tcp://*:5570"   # 语音服务地址（备用）
    wakeup_pub_addr: str = "tcp://*:5571"       # 唤醒+ASR PUB地址


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"


@dataclass
class SpeechConfig:
    """语音引擎配置"""
    # 模型路径
    wakeup_model_path: str = "models/wakeup"
    asr_model_path: str = "models/asr"
    cache_dir: str = "cache"
    
    # ASR模型文件
    asr_encoder_file: str = "encoder.int8.onnx"
    asr_decoder_file: str = "decoder.onnx"
    asr_joiner_file: str = "joiner.int8.onnx"
    
    # 唤醒模型文件
    wakeup_encoder_file: str = "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
    wakeup_decoder_file: str = "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
    wakeup_joiner_file: str = "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
    wakeup_keyword_file: str = "keywords.txt"
    
    # 音频参数
    sample_rate: int = 16000
    channels: int = 1
    mic_index: int = 1  # 整数索引，兼容旧配置；若 mic_name 为空则使用此索引
    mic_name: str = "1080P USB Camera-Audio"  # 设备名称（优先），避免插拔后索引变化
    
    # 唤醒词配置
    wakeup_keyword: str = "你好小白"
    wakeup_sensitivity: float = 0.2
    
    # ASR监听超时（秒）
    listen_timeout: float = 1.5


@dataclass
class GamepadConfig:
    """游戏手柄控制配置 - 同时控制底盘和机械臂"""
    
    # ========== 底盘控制参数 ==========
    max_linear_speed: float = 0.5          # 最大线速度 (m/s)
    max_angular_speed: float = 1.0         # 最大角速度 (rad/s)
    trigger_deadzone: float = 0.1          # 扳机键死区
    left_stick_deadzone: float = 0.15      # 左摇杆死区
    
    # ========== 机械臂控制参数 ==========
    arm_base_step: float = 3.0             # 基座关节步进 (度/帧)
    arm_elbow_step: float = 2.0            # 肘关节步进 (度/帧)
    arm_shoulder_step: float = 2.0         # 肩关节步进 (度/帧)
    arm_wrist_flex_step: float = 3.0       # 腕屈伸步进 (度/次)
    arm_wrist_roll_step: float = 3.0       # 腕旋转步进 (度/帧)
    arm_gripper_open: float = 90.0         # 夹爪打开角度
    arm_gripper_close: float = 0.0         # 夹爪关闭角度
    arm_speed: int = 800                   # 机械臂运动速度
    right_stick_deadzone: float = 0.15     # 右摇杆死区
    
    # ========== 通信配置 ==========
    chassis_service_addr: str = "tcp://localhost:5556"
    arm_service_addr: str = "tcp://localhost:5557"
    
    # ========== 轮询配置 ==========
    polling_interval: float = 0.02         # 50Hz (20ms)


@dataclass
class HumanFollowConfig:
    """人体跟随配置（YOLO26版）"""
    # 模型配置
    model_path: str = "models/yolo26n.pt"       # YOLO26 nano (~5.3MB)
    conf_threshold: float = 0.5               # 检测置信度阈值
    
    # 跟踪配置
    max_tracking_age: int = 30                # 最大丢失帧数
    min_iou_threshold: float = 0.3            # IoU匹配阈值
    target_selection: str = "center"          # 目标选择策略: center/largest/closest
    
    # 推理优化（边缘设备）
    inference_size: int = 320                 # 输入分辨率 320x320
    use_half_precision: bool = False          # FP16半精度推理（需GPU支持）
    
    # 跟随控制配置
    target_distance: float = 1.0              # 目标距离（米）
    target_width_ratio: float = 0.4          # 1米处人体占画面宽度比例（0.25=25%）
    target_height_ratio: float = 1.0          # 1米处人体占画面高度比例（1.0=100%）
    kp_linear: float = 0.8                    # 线速度P系数（归一化误差后）
    kp_angular: float = 1.5                   # 角速度P系数（归一化误差后）
    max_linear_speed: float = 0.5             # 最大线速度 (m/s)
    max_angular_speed: float = 2.0            # 最大角速度 (rad/s)
    dead_zone_x: float = 0.15                 # 水平死区（比例值，0.15=15%画面宽度）
    dead_zone_area: float = 0.1               # 面积死区（相对值）
    
    # 安全配置
    timeout_ms: int = 1000                    # 通信超时
    stop_on_lost: bool = True                 # 丢失目标时是否停止
    search_on_lost: bool = False               # 丢失时是否旋转搜索
    lost_patience: int = 30                   # 丢失容忍帧数（约2秒@30fps）
    
    # ZeroMQ配置
    chassis_service_addr: str = "tcp://localhost:5556"
    vision_sub_addr: str = "tcp://localhost:5560"


@dataclass
class BatteryConfig:
    """电池监测配置"""
    # 用于读取电压的舵机ID列表（按优先级排序）
    servo_ids: list = field(default_factory=lambda: [1])  # 默认使用ID 1
    
    # 电压阈值配置 (3S锂电池)
    full_voltage: float = 12.6     # 满电电压 (V)
    low_voltage: float = 10.5      # 低电量阈值 (V)
    critical_voltage: float = 9.5  # 严重低电量阈值 (V)
    min_voltage: float = 9.0       # 最低工作电压 (V)
    
    # 发布配置
    publish_interval: float = 5.0  # 电压信息发布间隔 (秒)
    pub_addr: str = "tcp://*:5555"  # 电池状态PUB地址
