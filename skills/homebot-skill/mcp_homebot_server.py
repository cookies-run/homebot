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

from scripts.chassis_control import HomeBotChassisController
from scripts.arm_control import HomeBotArmController
from scripts.what_does_robot_see_workflow import WhatDoesRobotSeeWorkflow
# 使用 grab_optimized 中的五阶段状态机抓取工作流
from scripts.grab_optimized import AutoGrabWorkflow
from scripts.robot_config import ROBOT_IP, CHASSIS_PORT, ARM_PORT

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
    result = chassis_controller.forward_cm(cm)
    return f"✅ 机器人前进 {cm} 厘米完成\n执行结果: {result}"


@mcp.tool()
async def chassis_backward(cm: float = Field(description="后退的距离，单位厘米，例如 10 表示后退10厘米")) -> str:
    """让机器人底盘后退指定距离"""
    result = chassis_controller.backward_cm(cm)
    return f"✅ 机器人后退 {cm} 厘米完成\n执行结果: {result}"


@mcp.tool()
async def chassis_left(degrees: float = Field(description="左转的角度，单位度数，例如 90 表示左转90度")) -> str:
    """让机器人底盘左转指定角度"""
    result = chassis_controller.left_deg(degrees)
    return f"✅ 机器人左转 {degrees} 度完成\n执行结果: {result}"


@mcp.tool()
async def chassis_right(degrees: float = Field(description="右转的角度，单位度数，例如 90 表示右转90度")) -> str:
    """让机器人底盘右转指定角度"""
    result = chassis_controller.right_deg(degrees)
    return f"✅ 机器人右转 {degrees} 度完成\n执行结果: {result}"


@mcp.tool()
async def chassis_stop() -> str:
    """紧急停止机器人底盘所有运动"""
    result = chassis_controller.stop()
    return f"✅ 机器人紧急停止完成\n执行结果: {result}"


@mcp.tool()
async def arm_move_joint(
    joint_name: str = Field(description="关节名称，可选值: base(基座), shoulder(肩关节), elbow(肘关节), wrist_flex(手腕俯仰), wrist_roll(手腕翻滚), gripper(夹爪)"),
    angle: float = Field(description="目标角度，单位度数")
) -> str:
    """移动机械臂指定关节到目标角度"""
    result = arm_controller.set_joint_angle(joint_name, angle)
    return f"✅ 机械臂 {joint_name} 移动到 {angle} 度完成\n执行结果: {result.success if result else False}"


@mcp.tool()
async def arm_get_positions() -> str:
    """获取机械臂所有关节当前角度位置"""
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
    result = arm_controller.send_command({}, source="emergency", priority=4)
    return f"✅ 机械臂停止完成\n执行结果: {result.success if result else False}"


@mcp.tool()
async def robot_what_does_robot_see() -> str:
    """捕获机器人摄像头最新画面，并用AI分析描述场景"""
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


if __name__ == "__main__":
    mcp.run()