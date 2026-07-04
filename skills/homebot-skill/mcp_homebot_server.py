"""
HomeBot MCP Server
Model Context Protocol 服务器，封装 HomeBot 机器人控制技能
支持底盘控制、机械臂控制、视觉查询功能
使用 FastMCP 简化开发
"""

import asyncio
import json
from mcp.server.fastmcp import FastMCP
from pydantic import Field

# 导入 HomeBot 模块
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/scripts")

# 细粒度确定性技能（查找 / 接近 / 递送）。
# 任务拆解与整体编排上移到 openclaw，Python 侧仅暴露每一步的原子能力，
# 并在每一步返回结构化反馈，由 openclaw 决定下一步。
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../software/src")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/scripts/delivery")
    from adapters import VisionAdapter, ChassisAdapter, ArmAdapter
    from search_skill import SearchSkill
    from approach_skill import ApproachSkill
    from place_skill import PlaceSkill
    _SKILLS_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] 递送技能导入失败: {e}")
    _SKILLS_AVAILABLE = False

from scripts.chassis_control import HomeBotChassisController
from scripts.arm_control import HomeBotArmController
from scripts.what_does_robot_see_workflow import WhatDoesRobotSeeWorkflow
# 使用 grab_optimized 中的五阶段状态机抓取工作流
from scripts.grab_optimized import AutoGrabWorkflow
from scripts.robot_config import ROBOT_IP, CHASSIS_PORT, ARM_PORT, VIDEO_PORT, END_VIDEO_PORT


# -----------------------------------------------------------------------------
# 机器人可达性检测
# -----------------------------------------------------------------------------
SKIP_REACHABILITY_CHECK = os.getenv("HOMEBOT_SKIP_REACHABILITY_CHECK", "false").lower() in ("1", "true", "yes")


async def check_robot_reachable(
    ip: str = ROBOT_IP,
    ports: list[int] | None = None,
    timeout: float = 2.0,
) -> tuple[bool, str]:
    """检测机器人指定端口是否可达。

    只要有一个端口能建立 TCP 连接，即认为机器人服务在线。
    返回 (是否可达, 提示信息)。不可达时提示信息会包含当前 IP 和修改方法。
    """
    if ports is None:
        ports = [CHASSIS_PORT, ARM_PORT, VIDEO_PORT]

    print(f"[HomeBot] 正在检测机器人可达性: ip={ip}, ports={ports}, timeout={timeout}s")

    async def _try_connect(port: int) -> tuple[int, bool]:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port), timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            print(f"[HomeBot] 端口 {port} 可连接")
            return port, True
        except asyncio.TimeoutError:
            print(f"[HomeBot] 端口 {port} 连接超时")
            return port, False
        except OSError as e:
            print(f"[HomeBot] 端口 {port} 连接失败: {e}")
            return port, False
        except Exception as e:
            print(f"[HomeBot] 端口 {port} 检测异常: {e}")
            return port, False

    results = await asyncio.gather(*(_try_connect(p) for p in ports))
    reachable_ports = [port for port, ok in results if ok]
    if reachable_ports:
        print(f"[HomeBot] 可达端口: {reachable_ports}")
        return True, ""

    tips = (
        f"❌ 无法连接到机器人服务（当前 IP: {ip}，检测端口: {ports}）。\n"
        f"请确认以下问题后再试：\n"
        f"1. 机器人已开机并连接到同一局域网；\n"
        f"2. 机器人后台服务已启动（运动、机械臂、视觉服务）；\n"
        f"3. 当前 IP 是否正确（当前默认值: {ip}）。\n\n"
        f"如需修改 IP，可通过以下方式之一：\n"
        f"  • 设置环境变量：export HOMEBOT_IP=你的机器人IP\n"
        f"  • 在 OpenClaw / Claude Code MCP 配置的 env 中设置 HOMEBOT_IP\n"
        f"  • 修改 scripts/robot_config.py 中的 ROBOT_IP 默认值\n\n"
        f"如果确定网络已通但仍被误拦，可设置 HOMEBOT_SKIP_REACHABILITY_CHECK=1 跳过检测。\n\n"
        f"也可先用 ping {ip} 测试网络连通性。"
    )
    return False, tips


# 缓存最近一次成功检测结果，避免同一轮对话中重复检测；失败不缓存，方便用户修复后重试
_last_reachable: bool | None = None
_last_check_ip: str | None = None


async def ensure_robot_reachable(ports: list[int] | None = None) -> tuple[bool, str]:
    """带缓存的可达性检测，供工具入口统一调用。"""
    if SKIP_REACHABILITY_CHECK:
        print("[HomeBot] 已跳过机器人可达性检测（HOMEBOT_SKIP_REACHABILITY_CHECK=1）")
        return True, ""

    global _last_reachable, _last_check_ip
    if _last_reachable is True and _last_check_ip == ROBOT_IP:
        print("[HomeBot] 使用缓存的可达性检测结果")
        return True, ""
    reachable, msg = await check_robot_reachable(ports=ports)
    if reachable:
        _last_reachable = True
        _last_check_ip = ROBOT_IP
    else:
        # 失败不缓存，方便服务恢复后直接重试
        _last_reachable = False
        _last_check_ip = None
    return reachable, msg

# 优先 MiniMax VLM，回退火山引擎
try:
    from scripts.minimax_vision_client import analyze_images as _vlm_analyze
    VLM_PROVIDER = "minimax"
except ImportError:
    try:
        from scripts.volcengine_vision_client import analyze_images as _vlm_analyze
        VLM_PROVIDER = "volcengine"
    except ImportError:
        VLM_PROVIDER = None

# 创建 FastMCP 服务器
mcp = FastMCP("homebot")

# 初始化控制器（自动连接）
chassis_controller = HomeBotChassisController(f"tcp://{ROBOT_IP}:{CHASSIS_PORT}")
arm_controller = HomeBotArmController(f"tcp://{ROBOT_IP}:{ARM_PORT}")


@mcp.tool()
async def chassis_forward(cm: float = Field(description="前进的距离，单位厘米，例如 10 表示前进10厘米")) -> str:
    """让机器人底盘前进指定距离"""
    reachable, msg = await ensure_robot_reachable(ports=[CHASSIS_PORT])
    if not reachable:
        return msg
    result = chassis_controller.forward_cm(cm)
    return f"✅ 机器人前进 {cm} 厘米完成\n执行结果: {result}"


@mcp.tool()
async def chassis_backward(cm: float = Field(description="后退的距离，单位厘米，例如 10 表示后退10厘米")) -> str:
    """让机器人底盘后退指定距离"""
    reachable, msg = await ensure_robot_reachable(ports=[CHASSIS_PORT])
    if not reachable:
        return msg
    result = chassis_controller.backward_cm(cm)
    return f"✅ 机器人后退 {cm} 厘米完成\n执行结果: {result}"


@mcp.tool()
async def chassis_left(degrees: float = Field(description="左转的角度，单位度数，例如 90 表示左转90度")) -> str:
    """让机器人底盘左转指定角度"""
    reachable, msg = await ensure_robot_reachable(ports=[CHASSIS_PORT])
    if not reachable:
        return msg
    result = chassis_controller.left_deg(degrees)
    return f"✅ 机器人左转 {degrees} 度完成\n执行结果: {result}"


@mcp.tool()
async def chassis_right(degrees: float = Field(description="右转的角度，单位度数，例如 90 表示右转90度")) -> str:
    """让机器人底盘右转指定角度"""
    reachable, msg = await ensure_robot_reachable(ports=[CHASSIS_PORT])
    if not reachable:
        return msg
    result = chassis_controller.right_deg(degrees)
    return f"✅ 机器人右转 {degrees} 度完成\n执行结果: {result}"


@mcp.tool()
async def chassis_stop() -> str:
    """紧急停止机器人底盘所有运动"""
    reachable, msg = await ensure_robot_reachable(ports=[CHASSIS_PORT])
    if not reachable:
        return msg
    result = chassis_controller.stop()
    return f"✅ 机器人紧急停止完成\n执行结果: {result}"


@mcp.tool()
async def arm_move_joint(
    joint_name: str = Field(description="关节名称，可选值: base(基座), shoulder(肩关节), elbow(肘关节), wrist_flex(手腕俯仰), wrist_roll(手腕翻滚), gripper(夹爪)"),
    angle: float = Field(description="目标角度，单位度数")
) -> str:
    """移动机械臂指定关节到目标角度"""
    reachable, msg = await ensure_robot_reachable(ports=[ARM_PORT])
    if not reachable:
        return msg
    result = arm_controller.set_joint_angle(joint_name, angle)
    return f"✅ 机械臂 {joint_name} 移动到 {angle} 度完成\n执行结果: {result.success if result else False}"


@mcp.tool()
async def arm_get_positions() -> str:
    """获取机械臂所有关节当前角度位置"""
    reachable, msg = await ensure_robot_reachable(ports=[ARM_PORT])
    if not reachable:
        return msg
    result = arm_controller.get_status()
    if result and result.joint_states:
        return f"✅ 获取机械臂当前位置完成\n当前各关节角度: {json.dumps(result.joint_states, indent=2, ensure_ascii=False)}"
    elif result:
        return f"✅ 获取成功，但未返回关节角度\n响应: {result.message}"
    else:
        return "❌ 获取机械臂位置失败"


@mcp.tool()
async def arm_stop() -> str:
    """停止机械臂所有运动"""
    reachable, msg = await ensure_robot_reachable(ports=[ARM_PORT])
    if not reachable:
        return msg
    result = arm_controller.send_command({}, source="emergency", priority=4)
    return f"✅ 机械臂停止完成\n执行结果: {result.success if result else False}"


@mcp.tool()
async def robot_what_does_robot_see() -> str:
    """捕获机器人摄像头最新画面，并用AI分析描述场景"""
    reachable, msg = await ensure_robot_reachable(ports=[VIDEO_PORT])
    if not reachable:
        return msg
    workflow = WhatDoesRobotSeeWorkflow()
    result = workflow.capture_and_analyze()
    if result["success"]:
        analysis_text = result["analysis"] if result["analysis"] else "图像捕获成功，但未进行分析"
        provider = getattr(workflow, 'VLM_PROVIDER', 'unknown')
        return f"✅ 视觉分析完成 ({provider})\n图像路径: {result['image_path']}\n场景描述: {analysis_text}"
    else:
        return f"❌ 视觉分析失败: 捕获图像未成功"


@mcp.tool()
async def auto_grab(
    target: str = Field(description="要抓取的目标物品描述，例如'一包纸巾'、'一个红色苹果'、'桌上的矿泉水瓶'"),
    use_end_camera: bool = Field(default=True, description="是否使用机械臂末端摄像头进行精对准。抓取类命令建议开启（默认 true）")
) -> str:
    """自主抓取/拿起/拿取指定物品。

    当用户发出抓取、拿起、拿取、取物、捡起等命令时调用此工具。
    机器人会主动观察画面、识别目标位置、调整底盘与机械臂姿态，然后执行抓取。
    不需要指定具体关节角度或底盘距离，只需描述要抓什么。

    典型触发语句:
        - "帮我拿那包纸巾"
        - "抓取桌上的红色苹果"
        - "把那个蓝色的盒子拿起来"
        - "捡起地上的玩具"
        - "取一下桌角的遥控器"
    """
    reachable, msg = await ensure_robot_reachable(ports=[VIDEO_PORT, ARM_PORT, CHASSIS_PORT])
    if not reachable:
        return msg
    try:
        workflow = AutoGrabWorkflow(
            robot_ip=ROBOT_IP,
            video_port=VIDEO_PORT,
            end_video_port=END_VIDEO_PORT,
            arm_port=ARM_PORT,
            max_attempts=8,
            use_end_camera=use_end_camera,
        )
        result = workflow.run(target_object=target)
        if result.get("success"):
            return f"✅ 已抓取: {target}。验证结果：{result.get('message', '')}"
        else:
            return f"❌ 抓取失败: {target}。原因：{result.get('message', '未知')}"
    except Exception as e:
        return f"❌ 抓取异常: {e}"


# 细粒度技能实例（惰性初始化，跨工具复用同一视觉订阅与底盘/机械臂适配器）
_vision_adapter = None
_search_skill = None
_approach_skill = None
_place_skill = None
_chassis_adapter = None
_arm_adapter = None

# 面积-距离标定基准：目标框归一化面积约 0.08 时距离约 30cm。
# 期望距离 D 处的停止面积按平方反比缩放 area = 0.08 * (30/D)^2。
# 这是基于框面积的粗估，非真实深度；后续手眼外参标定后会替换。
_AREA_AT_30CM = 0.08


def _get_vision_adapter() -> "VisionAdapter | None":
    """获取共享视觉订阅适配器。"""
    global _vision_adapter
    if _vision_adapter is None and _SKILLS_AVAILABLE:
        try:
            _vision_adapter = VisionAdapter(f"tcp://{ROBOT_IP}:{VIDEO_PORT}")
        except Exception as e:
            print(f"[WARNING] 视觉适配器初始化失败: {e}")
    return _vision_adapter


def _get_search_skill() -> "SearchSkill | None":
    """获取搜索技能实例（复用共享视觉订阅）。"""
    global _search_skill
    if _search_skill is None and _SKILLS_AVAILABLE:
        va = _get_vision_adapter()
        if va is not None:
            _search_skill = SearchSkill(va, provider="minimax")
    return _search_skill


def _get_approach_skill() -> "ApproachSkill | None":
    """获取接近技能实例（底盘速度 PID + 共享视觉订阅）。"""
    global _approach_skill, _chassis_adapter
    if _approach_skill is None and _SKILLS_AVAILABLE:
        try:
            va = _get_vision_adapter()
            if _chassis_adapter is None:
                _chassis_adapter = ChassisAdapter(f"tcp://{ROBOT_IP}:{CHASSIS_PORT}")
            if va is not None:
                _approach_skill = ApproachSkill(_chassis_adapter, va)
        except Exception as e:
            print(f"[WARNING] 接近技能初始化失败: {e}")
    return _approach_skill


def _get_place_skill() -> "PlaceSkill | None":
    """获取递送/放置技能实例（机械臂适配器）。"""
    global _place_skill, _arm_adapter
    if _place_skill is None and _SKILLS_AVAILABLE:
        try:
            if _arm_adapter is None:
                _arm_adapter = ArmAdapter(f"tcp://{ROBOT_IP}:{ARM_PORT}")
            _place_skill = PlaceSkill(_arm_adapter, _get_vision_adapter())
        except Exception as e:
            print(f"[WARNING] 递送技能初始化失败: {e}")
    return _place_skill


@mcp.tool()
async def search_target(
    target: str = Field(description="要寻找的单个目标描述，例如'一包纸巾'、'穿黑色上衣的人'、'桌上的矿泉水瓶'")
) -> str:
    """旋转扫描寻找单个目标，返回其在画面中的位置(bbox)。

    机器人先看当前画面是否有该目标；找到则记录位置并返回；找不到则左转 120°
    再看，如此循环最多旋转 3 次(转满一圈回到起始朝向，最多分析 4 帧)。一次只找
    一个目标——需要同时定位抓取目标和递送目标时，请分两次调用本工具。

    返回 JSON 字段：found(是否找到)、bbox(归一化 xyxy 位置)、height_cm(估计高度)、
    pose(姿态)、scene_description(模型对整张画面的描述)、graspable(是否可抓)、
    views_used(实际搜索画面数)、rotations_used(扫描旋转次数)、alignments_used(找到目标后
    微调对准次数)。找到目标后会尽量让机身正面正对目标。found=false 时由调用方决定是否移动
    到别处再搜。

    典型触发语句:
        - "帮我找一下那包纸巾"
        - "找找穿黑色上衣的人在哪"
        - "看看附近有没有矿泉水"
    """
    reachable, msg = await ensure_robot_reachable(ports=[VIDEO_PORT, CHASSIS_PORT])
    if not reachable:
        return msg
    skill = _get_search_skill()
    if skill is None:
        return "❌ 搜索技能不可用"
    try:
        result = skill.search_with_scan(
            target,
            rotate_fn=lambda: chassis_controller.left_deg(120),
            rotate_deg=120,
            max_rotations=3,
            align_fn=lambda deg: chassis_controller.left_deg(deg) if deg > 0 else chassis_controller.right_deg(-deg),
            camera_hfov_deg=60.0,
            align_threshold=0.1,
            max_align_attempts=3,
        )
        return f"{'✅ 已找到' if result.get('found') else '❌ 未找到'}目标: {target}\n" + \
            json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ 搜索异常: {e}"


@mcp.tool()
async def approach_target(
    target: str = Field(description="要接近的单个目标描述，例如'一包纸巾'、'穿黑色上衣的人'"),
    distance_cm: float = Field(default=40.0, description="期望停靠距离(厘米)，默认 40。注意：当前为基于目标框面积的反推估计，非真实深度；后续手眼外参标定后精度会提升。")
) -> str:
    """接近单个目标到指定距离（初步定位，为后续精细抓取/递送让出空间）。

    工作方式：先在当前朝向用 VLM 定位一次拿到初始位置，然后基于底盘速度 PID
    朝目标前进，过程中周期性重定位使目标框随接近而放大，直到估计距离达到
    distance_cm 或超时。本工具不做抓取，只负责把机器人开到目标正前方约
    distance_cm 处。

    调用前提：目标应大致在当前画面内（通常先调用 search_target 定位/转向）。
    若当前画面找不到目标，会直接返回失败并建议先 search_target。

    返回 JSON：success(是否到位)、message、final_distance_cm(估计最终距离)。

    典型编排：search_target(抓取目标) → approach_target(抓取目标) → auto_grab(抓取)
             → search_target(递送目标) → approach_target(递送目标) → deliver
    """
    reachable, msg = await ensure_robot_reachable(ports=[VIDEO_PORT, CHASSIS_PORT])
    if not reachable:
        return msg
    if not _SKILLS_AVAILABLE:
        return "❌ 接近技能不可用"
    search = _get_search_skill()
    approach = _get_approach_skill()
    if search is None or approach is None:
        return "❌ 接近技能初始化失败（视觉/底盘适配器不可用）"
    try:
        loc = search.search(target)
        if not loc.get("found"):
            return (
                f"❌ 接近失败：当前画面未找到目标「{target}」。"
                f"建议先调用 search_target 旋转扫描定位后再接近。\n"
                + json.dumps(loc, ensure_ascii=False)
            )
        bbox = loc["bbox"]
        # 依据期望距离设置面积停止阈值（平方反比缩放），并让最终距离估算与之一致
        approach.approach_distance_cm = distance_cm
        approach.target_area_at_30cm = _AREA_AT_30CM * (30.0 / max(distance_cm, 1.0)) ** 2
        result = approach.approach(
            tuple(bbox),
            timeout_s=30.0,
            relocate_fn=lambda: (search.search(target) or {}).get("bbox"),
        )
        status = "✅ 已接近" if result.get("success") else "❌ 未到位"
        return f"{status}目标「{target}」(目标距离 {distance_cm:.0f}cm)\n" + \
            json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ 接近异常: {e}"


@mcp.tool()
async def deliver(
    target: str = Field(default="", description="递送目标描述（可选，用于日志/未来手部确认），例如'穿黑色上衣的人'、'餐桌上'")
) -> str:
    """递送/放置：将夹爪伸到目标前方并松开夹爪，随后机械臂复位。

    作为递送流程的最后一步，在已用 approach_target 接近递送目标之后调用。
    本工具假设机器人已接近到位，只负责机械臂伸出 → 打开夹爪释放 → 复位，
    不再移动底盘。

    返回 JSON：success(是否成功)、message。

    典型触发语句:
        - "把它递过去"
        - "放到这里"
        - "松开夹爪把东西给他"
    """
    reachable, msg = await ensure_robot_reachable(ports=[ARM_PORT])
    if not reachable:
        return msg
    if not _SKILLS_AVAILABLE:
        return "❌ 递送技能不可用"
    place = _get_place_skill()
    if place is None:
        return "❌ 递送技能初始化失败（机械臂适配器不可用）"
    try:
        result = place.execute()
        ok = result.get("success")
        who = f"给「{target}」" if target else ""
        return f"{'✅ 已递送' if ok else '❌ 递送失败'}{who}\n" + \
            json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ 递送异常: {e}"



if __name__ == "__main__":
    mcp.run()