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

# 尝试导入递送智能体（依赖 software/src 中的应用层运行时）
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../software/src")
    from applications.delivery_agent import DeliveryAgent, DeliveryAgentConfig
    _DELIVERY_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] 递送智能体导入失败: {e}")
    _DELIVERY_AVAILABLE = False

from scripts.chassis_control import HomeBotChassisController
from scripts.arm_control import HomeBotArmController
from scripts.what_does_robot_see_workflow import WhatDoesRobotSeeWorkflow
# 使用 grab_optimized 中的五阶段状态机抓取工作流
from scripts.grab_optimized import AutoGrabWorkflow
from scripts.robot_config import ROBOT_IP, CHASSIS_PORT, ARM_PORT, VIDEO_PORT


# -----------------------------------------------------------------------------
# 机器人可达性检测
# -----------------------------------------------------------------------------
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

    async def _try_connect(port: int) -> bool:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port), timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    results = await asyncio.gather(*(_try_connect(p) for p in ports))
    if any(results):
        return True, ""

    tips = (
        f"❌ 无法连接到机器人服务（当前 IP: {ip}）。\n"
        f"请确认以下问题后再试：\n"
        f"1. 机器人已开机并连接到同一局域网；\n"
        f"2. 机器人后台服务已启动（运动、机械臂、视觉服务）；\n"
        f"3. 当前 IP 是否正确（当前默认值: {ip}）。\n\n"
        f"如需修改 IP，可通过以下方式之一：\n"
        f"  • 设置环境变量：export HOMEBOT_IP=你的机器人IP\n"
        f"  • 在 OpenClaw / Claude Code MCP 配置的 env 中设置 HOMEBOT_IP\n"
        f"  • 修改 scripts/robot_config.py 中的 ROBOT_IP 默认值\n\n"
        f"也可先用 ping {ip} 测试网络连通性。"
    )
    return False, tips


# 缓存最近一次检测结果，避免同一轮对话中重复检测
_last_reachable: bool | None = None
_last_check_ip: str | None = None


async def ensure_robot_reachable(ports: list[int] | None = None) -> tuple[bool, str]:
    """带缓存的可达性检测，供工具入口统一调用。"""
    global _last_reachable, _last_check_ip
    if _last_reachable is True and _last_check_ip == ROBOT_IP:
        return True, ""
    reachable, msg = await check_robot_reachable(ports=ports)
    _last_reachable = reachable
    _last_check_ip = ROBOT_IP
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
            video_port=5560,
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


# 全局递送智能体实例（惰性初始化）
_delivery_agent = None


def _get_delivery_agent() -> "DeliveryAgent | None":
    """获取递送智能体实例。"""
    global _delivery_agent
    if _delivery_agent is None and _DELIVERY_AVAILABLE:
        try:
            _delivery_agent = DeliveryAgent(DeliveryAgentConfig.from_env())
        except Exception as e:
            print(f"[WARNING] 递送智能体初始化失败: {e}")
    return _delivery_agent


@mcp.tool()
async def delivery_plan(user_request: str = Field(description="用户的递送请求文本，例如'把矿泉水递给穿红衣服的人'")) -> str:
    """解析用户递送请求，提取抓取目标和递送目标，并生成确认话术。

    当用户说"把XX递给YY"、"帮我把XX拿到YY"等递送类请求时，先调用此工具。
    工具返回向用户确认的话术，需要用户明确同意后，再调用 delivery_confirm。
    """
    reachable, msg = await ensure_robot_reachable(ports=[CHASSIS_PORT, ARM_PORT, VIDEO_PORT])
    if not reachable:
        return msg
    agent = _get_delivery_agent()
    if agent is None:
        return "❌ 递送智能体不可用"
    try:
        result = agent.process_user_request(user_request)
        if result.get("action") == "parse":
            return (
                f"✅ 已解析递送请求\n"
                f"抓取目标: {result.get('grab_target')}\n"
                f"递送目标: {result.get('deliver_target')}\n"
                f"确认话术: {result.get('message')}"
            )
        return f"❌ 解析失败: {result.get('message', '未知错误')}"
    except Exception as e:
        return f"❌ 规划递送任务失败: {e}"


@mcp.tool()
async def delivery_confirm(
    confirmed: bool = Field(description="用户是否确认执行递送任务"),
    grab_target: str = Field(default="", description="抓取目标，例如'矿泉水'"),
    deliver_target: str = Field(default="", description="递送目标，例如'穿红衣服的人'"),
) -> str:
    """用户确认后执行预检查并启动递送任务。

    在 delivery_plan 之后，如果用户明确同意（如说"对"、"是的"、"好"），调用此工具。
    它会搜索抓取目标、判断可抓取性、确认递送目标是否在附近，通过后启动后台执行。
    """
    reachable, msg = await ensure_robot_reachable(ports=[CHASSIS_PORT, ARM_PORT, VIDEO_PORT])
    if not reachable:
        return msg
    agent = _get_delivery_agent()
    if agent is None:
        return "❌ 递送智能体不可用"
    try:
        # 如果显式传入了目标，同步到状态
        state = agent.state.get()
        if grab_target and state.grab_target != grab_target:
            agent.state.update(grab_target=grab_target)
        if deliver_target and state.deliver_target != deliver_target:
            agent.state.update(deliver_target=deliver_target)

        result = agent.process_confirmation(confirmed)
        action = result.get("action")

        if action == "execute":
            return (
                f"✅ 预检查通过，已开始执行递送任务\n"
                f"任务ID: {result.get('task_id')}\n"
                f"提示: {result.get('message')}"
            )
        elif action == "cancelled":
            return f"⛔ 任务已取消: {result.get('message')}"
        elif action == "precheck_need_search":
            return f"⚠️ 预检查: {result.get('message')}"
        elif action == "precheck_not_graspable":
            return f"⚠️ 预检查: {result.get('message')}"
        elif action == "precheck_destination_not_nearby":
            return f"⚠️ 预检查: {result.get('message')}"
        else:
            return f"❌ 确认处理失败: {result.get('message', '未知错误')}"
    except Exception as e:
        return f"❌ 确认递送任务失败: {e}"


@mcp.tool()
async def delivery_status() -> str:
    """获取当前递送任务的执行状态。"""
    reachable, msg = await ensure_robot_reachable(ports=[CHASSIS_PORT, ARM_PORT, VIDEO_PORT])
    if not reachable:
        return msg
    agent = _get_delivery_agent()
    if agent is None:
        return "❌ 递送智能体不可用"
    try:
        status = agent.get_status()
        return (
            f"✅ 递送任务状态\n"
            f"阶段: {status.get('phase')}\n"
            f"抓取目标: {status.get('grab_target')}\n"
            f"递送目标: {status.get('deliver_target')}\n"
            f"手持物体: {status.get('held_object')}\n"
            f"已确认: {status.get('confirmed')}\n"
            f"预检查通过: {status.get('precheck_passed')}\n"
            f"错误信息: {status.get('error_message')}"
        )
    except Exception as e:
        return f"❌ 获取递送状态失败: {e}"


if __name__ == "__main__":
    mcp.run()