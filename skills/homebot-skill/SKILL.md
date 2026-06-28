---
name: homebot
description: HomeBot 完整机器人控制器技能。当用户发出抓取/拿起/拿取类命令（如“抓取桌上的苹果”“帮我拿包纸巾”“把那个瓶子拿起来”）时，必须调用 auto_grab 工具执行自主视觉引导抓取；当用户发出递送类命令（如“把矿泉水递给穿红衣服的人”“帮我把纸巾拿到桌上”）时，必须调用 delivery_plan / delivery_confirm / delivery_status 工具执行完整递送闭环；同时集成底盘运动控制、机械臂关节控制、摄像头视觉画面捕获与AI分析，全部基于 ZeroMQ 局域网通信协议。
version: 1.0.3
metadata:
  openclaw:
    requires:
      bins:
      - python
      - pip
    install:
    - kind: pip
      requirements_file: requirements.txt
    emoji: "🤖"
---

# HomeBot Robot Controller

HomeBot 完整机器人控制器技能，集成五大功能模块：
- 🚗 **底盘控制**：前进/后退/转向，精确距离角度控制
- 🦾 **机械臂控制**：6自由度关节控制，夹爪控制，回原点
- 🤏 **自主抓取**：视觉引导自动识别、定位并抓取指定物品
- 📦 **递送任务**：理解自然语言 → 确认 → 预检查 → 接近/抓取/找目标/接近/放置完整闭环
- 👁️ **视觉查询**：一键捕获机器人摄像头画面，**自动调用火山引擎 LLM 分析图像内容**

全部基于 ZeroMQ REQ-REP / PUB 局域网通信协议，完美匹配 HomeBot 项目服务端架构。

## Agent 调用触发词

当收到以下任一类型的自然语言指令时，Agent 应调用本技能的 MCP 工具：

**抓取类（调用 `auto_grab`）：**

| 触发类型 | 示例命令 |
|---------|---------|
| 抓取 | “抓取桌上的一包纸巾”、“把那个红色苹果抓起来” |
| 拿起/拿取 | “帮我拿瓶矿泉水”、“把地上的玩具捡起来” |
| 取物 | “取一下桌角的遥控器”、“把笔递给我” |
| 收拾/移动 | “把这个空杯子拿到旁边” |

**递送类（调用 `delivery_plan` → 用户确认后调用 `delivery_confirm`）：**

| 触发类型 | 示例命令 |
|---------|---------|
| 递给人 | “把矿泉水递给穿红衣服的人”、“帮我把纸巾递给小明” |
| 递到地点 | “把杯子拿到餐桌上”、“帮我把书放到沙发上” |
| 拿取并递送 | “帮我把桌上的苹果拿给那边的人” |

**调用约定：**
- 抓取命令：首选调用 MCP 工具 `auto_grab(target="目标物品描述")`，不要先调用底盘或机械臂工具做预对准。
- 递送命令：必须先调用 `delivery_plan(user_request="用户原始请求")` 解析语义并获取确认话术；用户明确同意后，再调用 `delivery_confirm(confirmed=true, grab_target="...", deliver_target="...")` 执行预检查并启动递送。
- 如果 MCP 不可用、必须退回到 shell 执行时，请直接运行 `python scripts/grab_optimized.py --target "目标物品"`，**不要**运行任何名为 `auto_grab_workflow` 的旧脚本（该文件已废弃并重命名为 `_auto_grab_workflow_legacy.py`）。
- `target` 应包含物品名称及其位置/外观特征，例如 `"桌上的一包纸巾"`、`"红色的苹果"`。
- 当命令意图明确为抓取时，优先使用末端摄像头精对准：`auto_grab(target="...", use_end_camera=true)`。

## 快速开始

### 1. 配置机器人连接

**方式一：环境变量（推荐，适合 OpenClaw 等 Agent）**

```bash
# Windows
set HOMEBOT_IP=192.168.1.13
set ARK_API_KEY=your_volcengine_api_key

# Linux/Mac
export HOMEBOT_IP=192.168.1.13
export ARK_API_KEY=your_volcengine_api_key
```

**方式二：修改配置文件**

编辑 `scripts/robot_config.py` 修改机器人 IP：
```python
ROBOT_IP = "192.168.1.13"  # 修改为你的机器人IP
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 测试连接

```bash
# 测试底盘连接
python scripts/chassis_control.py forward 10

# 测试机械臂连接
python scripts/arm_control.py status

# 测试视觉捕获
python scripts/what_does_robot_see_workflow.py --no-analysis

# 测试自主抓取（默认抓取一包纸巾）
python scripts/grab_optimized.py
```

---

## 环境变量配置

所有配置都可通过环境变量设置，优先级：**环境变量 > 配置文件默认值**

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `HOMEBOT_IP` | 机器人 IP 地址 | `192.168.1.13` |
| `HOMEBOT_CHASSIS_PORT` | 底盘服务端口 | `5556` |
| `HOMEBOT_ARM_PORT` | 机械臂服务端口 | `5557` |
| `HOMEBOT_VIDEO_PORT` | 视频流端口 | `5560` |
| `HOMEBOT_END_VIDEO_PORT` | 机械臂末端摄像头端口 | `5561` |
| `HOMEBOT_CAPTURE_TIMEOUT` | 图像捕获超时(秒) | `10.0` |
| `HOMEBOT_OUTPUT_DIR` | 图像保存目录 | `.` |
| `ARK_API_KEY` | 火山引擎 API Key（视觉分析） | - |
| `ARK_MODEL_ID` | 火山引擎模型 ID | `doubao-seed-2-0-lite-260215` |

### 连接前检测

所有 MCP 工具在调用前都会先检测机器人服务是否可达（默认检测 `192.168.1.13` 的 `5556/5557/5560` 端口）。

- 如果检测通过，工具会正常执行；
- 如果检测失败，工具会立即返回提示信息，告知当前使用的 IP，并引导你确认机器人状态或修改 `HOMEBOT_IP`。

因此，当 MCP 提示“无法连接到机器人服务”时，请优先检查：
1. 机器人是否已开机并接入同一局域网；
2. 机器人后台服务是否已启动；
3. `HOMEBOT_IP` 是否与实际机器人 IP 一致。

### OpenClaw 配置示例

在 OpenClaw 配置中通过 `env` 设置环境变量：

```yaml
mcp:
  servers:
    homebot:
      command: "python"
      args: 
        - "{{skill_path}}/mcp_homebot_server.py"
      env:
        # 机器人连接配置
        HOMEBOT_IP: "192.168.1.13"
        HOMEBOT_CHASSIS_PORT: "5556"
        HOMEBOT_ARM_PORT: "5557"
        HOMEBOT_VIDEO_PORT: "5560"
        HOMEBOT_END_VIDEO_PORT: "5561"
        
        # 火山引擎视觉分析配置（可选）
        ARK_API_KEY: "your-api-key-here"
        ARK_MODEL_ID: "doubao-seed-2-0-lite-260215"
```

---

## 模块说明

| 模块 | 功能 | 默认端口 | 源文件 |
|------|------|----------|--------|
| `chassis` | 底盘运动控制 | 5556 | `chassis_control.py` |
| `arm` | 机械臂关节控制 | 5557 | `arm_control.py` |
| `grab` | 自主视觉引导抓取 | 5560/5561 | `grab_optimized.py` |
| `delivery` | 递送任务完整闭环 | 5556/5557/5560/5561 | `scripts/delivery/*.py` + `applications/delivery_agent/` |
| `vision` | 摄像头画面捕获+AI分析 | 5560 | `video_subscriber.py` / `what_does_robot_see_workflow.py` |
| `gestures` | 机械臂姿态动作（挥手/点头/摇头） | - | `arm_gestures.py` |

---

## 1. 底盘控制 (Chassis)

精确控制机器人底盘运动，支持指定距离前进后退，指定角度左转右转，实时速度控制。

### 命令行使用

```bash
# 前进指定距离（厘米）
python scripts/chassis_control.py forward 10

# 后退指定距离（厘米）
python scripts/chassis_control.py backward 20

# 右转指定角度（度）
python scripts/chassis_control.py right 90

# 左转指定角度（度）
python scripts/chassis_control.py left 45

# 设置速度（线速度 m/s, 角速度 rad/s）
python scripts/chassis_control.py velocity 0.2 0.0

# 紧急停止
python scripts/chassis_control.py stop

# 交互式控制（w/a/s/d 键盘控制）
python scripts/chassis_control.py interactive
```

### Python API

```python
from scripts.chassis_control import HomeBotChassisController
from scripts.robot_config import ROBOT_IP, CHASSIS_PORT

bot = HomeBotChassisController(ip=ROBOT_IP, port=CHASSIS_PORT)

# 前进 10cm
bot.forward_cm(10)

# 右转 90度
bot.right_deg(90)

# 停止
bot.stop()

bot.close()
```

---

## 2. 机械臂控制 (Arm)

6自由度机械臂控制，支持单个/多个关节角度设置，夹爪控制，回原点，优先级仲裁。

### 关节命名

| 关节名 | 说明 | 默认舵机ID |
|--------|------|-----------|
| `base` | 基座旋转 | 1 |
| `shoulder` | 肩关节 | 2 |
| `elbow` | 肘关节 | 3 |
| `wrist_flex` | 腕关节俯仰 | 4 |
| `wrist_roll` | 腕关节旋转 | 5 |
| `gripper` | 夹爪 | 6 |

### 优先级

| 优先级 | 控制源 | 说明 |
|--------|--------|------|
| 4 | emergency | 紧急停止（最高） |
| 3 | auto | 自动控制 |
| 2 | voice | 语音控制（默认） |
| 1 | web | 网页遥控 |

### 命令行使用

```bash
# 设置单个关节角度（度）
python scripts/arm_control.py joint base 0

# 同时设置多个关节角度
python scripts/arm_control.py joints "base:0,shoulder:10,elbow:45"

# 打开夹爪（90度）
python scripts/arm_control.py gripper open

# 关闭夹爪（0度）
python scripts/arm_control.py gripper close

# 设置夹爪角度（0-90度）
python scripts/arm_control.py gripper 45

# 回原点（休息位置，由服务端配置）
python scripts/arm_control.py home

# 获取当前所有关节角度
python scripts/arm_control.py status

# 紧急停止
python scripts/arm_control.py stop
```

### Python API

```python
from scripts.arm_control import HomeBotArmController
from scripts.robot_config import ROBOT_IP, ARM_PORT

arm = HomeBotArmController(robot_ip=ROBOT_IP, robot_port=ARM_PORT)

# 设置单个关节
resp = arm.set_joint_angle("shoulder", 30.0)

# 同时设置多个关节
resp = arm.set_joint_angles({
    "base": 0,
    "shoulder": 20,
    "elbow": 45,
    "wrist_flex": 0
})

# 打开夹爪
resp = arm.open_gripper()

# 回原点
resp = arm.move_home()

arm.close()
```

---

## 3. 自主抓取 (Grab)

基于 **五阶段状态机** 的自主视觉引导抓取：
- **Phase 0**: 机械臂回到观察姿态
- **Phase 1**: VLM 属性测姿 + 底盘粗定位
- **Phase 2**: 末端摄像头追踪 + 手腕姿态预部署
- **Phase 2.5**: 纯几何二次距离闭环，精准贴紧
- **Phase 3**: VLM 触达确认 / 对齐终审
- **Phase 4**: 夹紧、抬升、回缩

### 前置要求

1. 机器人底盘、机械臂服务已启动
2. 机身摄像头（端口 5560）已发布画面
3. 机械臂末端摄像头（端口 5561）已发布画面（当前实现 Phase 2 起依赖末端摄像头）
4. 已配置 VLM API Key（MiniMax 优先，可回退到火山引擎）
5. Python 环境已安装 `opencv-python` 和 `numpy`

### 命令行使用

```bash
# 默认抓取一包纸巾
python scripts/grab_optimized.py

# 抓取指定物品
python scripts/grab_optimized.py --target "一瓶矿泉水"

# 指定机器人 IP
python scripts/grab_optimized.py --ip 192.168.1.13

# 禁用末端摄像头（当前版本仍会检测，缺失会失败）
python scripts/grab_optimized.py --no-end-camera
```

### Python API

```python
from scripts.grab_optimized import AutoGrabWorkflow

workflow = AutoGrabWorkflow(
    robot_ip="192.168.1.13",
    video_port=5560,          # 机身摄像头
    end_video_port=5561,      # 末端摄像头
    arm_port=5557,
    max_attempts=8,
    use_end_camera=True,      # 建议启用
)

result = workflow.run(target_object="一包纸巾")
print(result)
```

---

## 4. 递送任务 (Delivery)

基于大模型语义解析与状态机的完整递送闭环：

1. **PARSE**：从用户请求中提取 `grab_target`（抓取目标）和 `deliver_target`（递送目标）。
2. **CONFIRM**：向用户复述计划，等待明确确认。
3. **PRE_CHECK**：搜索抓取目标、判断可抓取性、确认递送目标在附近。
4. **EXECUTE**：按阶段推进
   - `APPROACH_TARGET`：接近抓取目标
   - `GRASP`：抓取物品
   - `FIND_DESTINATION`：重新定位递送目标
   - `APPROACH_DESTINATION`：接近递送目标
   - `PLACE`：放置/释放物品

### 前置要求

1. 机器人底盘、机械臂服务已启动
2. 机身摄像头（端口 5560）已发布画面
3. 机械臂末端摄像头（端口 5561）已发布画面（抓取阶段使用）
4. 已配置 LLM API Key（用于语义解析）
5. Python 环境已安装 `opencv-python` 和 `numpy`

### 命令行使用

```bash
# 启动递送智能体独立入口
python ../../software/start_delivery_agent.py
```

### Python API

```python
import sys
import os
sys.path.append("../../software/src")

from applications.delivery_agent import DeliveryAgent, DeliveryAgentConfig

agent = DeliveryAgent(DeliveryAgentConfig.from_env())

# 1. 解析用户请求
plan = agent.process_user_request("把矿泉水递给穿红衣服的人")
print(plan["message"])  # 你要我拿起矿泉水，递给穿红衣服的人，对吗？

# 2. 用户确认后启动执行
result = agent.process_confirmation(True)
print(result)

# 3. 查询状态
print(agent.get_status())
```

### 可复用技能原语

递送任务底层由以下外部技能组合而成，均位于 `scripts/delivery/`：

| 技能 | 功能 | 源文件 |
|------|------|--------|
| `SearchSkill` | VLM 视觉搜索目标，返回 bbox/高度/姿态/可抓取性 | `scripts/delivery/search_skill.py` |
| `ApproachSkill` | 基于跟踪器的 PID 接近控制 | `scripts/delivery/approach_skill.py` |
| `GraspSkill` | 封装 `grab_optimized.py` 的自主抓取 | `scripts/delivery/grasp_skill.py` |
| `PlaceSkill` | 释放姿态 + 打开夹爪 + 机械臂复位 | `scripts/delivery/place_skill.py` |
| `TargetTracker` | 轻量 IoU 跟踪器，带目标切换迟滞 | `scripts/delivery/tracker_skill.py` |

---

## 5. 视觉查询 (Vision)

一键捕获机器人摄像头最新画面，**集成火山引擎 LLM 自动分析图像内容**。

### 前置要求

视觉分析功能需要配置火山引擎 API Key：

```bash
# 设置环境变量
export ARK_API_KEY="your-api-key-here"
export ARK_MODEL_ID="doubao-seed-2-0-lite-260215"  # 可选
```

### 一键完整工作流（捕获 + 分析）

```bash
# 捕获图像并自动分析
python scripts/what_does_robot_see_workflow.py

# 仅捕获图像，不进行分析
python scripts/what_does_robot_see_workflow.py --no-analysis

# 使用自定义提示词分析
python scripts/what_does_robot_see_workflow.py --prompt "图中有几个人？他们在做什么？"

# 指定不同模型
python scripts/what_does_robot_see_workflow.py --model doubao-vision-pro-250226
```

**输出示例：**
```
[INFO] 正在连接机器人 192.168.1.13:5560...
[OK] 图像捕获成功
[INFO] 保存位置: C:\...\homebot_capture_20260319_154530.jpg
[INFO] 文件大小: 45231 字节
[INFO] 正在使用火山引擎分析图片...
[INFO] 模型: doubao-vision-lite-250225
```

### 视频订阅工具

```bash
# 获取单张图像
python scripts/video_subscriber.py --ip 192.168.1.13 --port 5560

# 持续接收所有帧并保存到目录
python scripts/video_subscriber.py --ip 192.168.1.13 --port 5560 --keep-receiving --output-dir ./frames
```

### Python API

```python
from scripts.what_does_robot_see_workflow import WhatDoesRobotSeeWorkflow

# 完整工作流：捕获 + 分析
workflow = WhatDoesRobotSeeWorkflow(
    enable_analysis=True,
    prompt="描述图片中的主要物体",
    model="doubao-vision-lite-250225"
)

result = workflow.capture_and_analyze()
if result["success"]:
    print(f"图像路径: {result['image_path']}")
    print(f"分析结果: {result['analysis']}")

# 仅捕获图像
workflow = WhatDoesRobotSeeWorkflow(enable_analysis=False)
image_path = workflow.capture()

# 单独分析已有图片
analysis = workflow.analyze("path/to/image.jpg")
```

---

## MCP 服务器支持 🚀

本技能内置 **Model Context Protocol (MCP)** 服务器，可直接配置给 OpenClaw/LLM 调用，让 AI 自动操控机器人！

### 功能封装

MCP 服务器封装了以下工具：

| 工具名称 | 功能描述 |
|---------|---------|
| `chassis_forward` | 机器人前进指定距离（厘米） |
| `chassis_backward` | 机器人后退指定距离（厘米） |
| `chassis_left` | 机器人左转指定角度（度数） |
| `chassis_right` | 机器人右转指定角度（度数） |
| `chassis_stop` | 紧急停止机器人底盘 |
| `arm_move_joint` | 移动机械臂指定关节到目标角度 |
| `arm_get_positions` | 获取机械臂所有关节当前位置 |
| `arm_stop` | 停止机械臂所有运动 |
| `auto_grab` | **抓取/拿起/拿取指定物品（Agent 收到抓取命令时直接调用）**。机器人主动观察、识别、定位并执行抓取，无需手动预对准。 |
| `delivery_plan` | **递送任务第1步**：解析用户递送请求，提取抓取/递送目标，生成确认话术。 |
| `delivery_confirm` | **递送任务第2步**：用户确认后执行预检查并启动递送执行。 |
| `delivery_status` | **递送任务第3步**：查询当前递送任务的执行阶段与状态。 |
| `robot_what_does_robot_see` | 捕获机器人画面并 AI 分析场景 |

### MCP 配置方法

在 OpenClaw 配置文件 `config.yaml` 中添加：

```yaml
mcp:
  servers:
    homebot:
      command: "python"
      args: 
        - "{{skill_path}}/mcp_homebot_server.py"
      env:
        # === 机器人连接配置（必填）===
        HOMEBOT_IP: "192.168.1.13"
        HOMEBOT_END_VIDEO_PORT: "5561"
        
        # === 大模型配置（递送任务语义解析需要）===
        # DeliveryAgent 默认读取 software/src/configs 中的 LLM 配置
        # 请确保对应配置文件或环境变量中已设置 api_key / api_url / model
        
        # === 火山引擎视觉分析配置（可选，用于 robot_what_does_robot_see 功能）===
        ARK_API_KEY: "your-volcengine-api-key"
        ARK_MODEL_ID: "doubao-seed-2-0-lite-260215"
```

> **注意**: `{{skill_path}}` 是 OpenClaw 的变量，会自动替换为技能的实际路径。

### 依赖安装

```bash
pip install mcp
```

### 使用效果

配置完成后，LLM 即可**直接调用所有机器人控制工具**，自动完成：
- 根据自然语言指令控制机器人移动
- 调整机械臂位置
- 自主识别并抓取指定物品
- **理解递送请求，与用户确认后完成“抓取→递送→放置”完整闭环**
- 让机器人自动观察环境并报告场景

---

## 通信协议

全部基于 ZeroMQ 协议，完全匹配 HomeBot 项目服务端配置：

| 服务 | 模式 | 默认端口 |
|------|------|----------|
| 底盘控制 | REQ-REP | 5556 |
| 机械臂控制 | REQ-REP | 5557 |
| 视频发布 | PUB | 5560 |
| 末端摄像头发布 | PUB | 5561 |

HomeBot 服务端配置示例：
```python
# config.py
class ZMQConfig:
    chassis_service_addr: str = "tcp://*:5556"
    arm_service_addr: str = "tcp://*:5557"
    vision_pub_addr: str = "tcp://*:5560"
```

---

## 依赖

- Python 3.8+
- pyzmq >= 25.0.0
- Pillow >= 9.0.0
- numpy >= 1.24.0
- opencv-python >= 4.8.0（自主抓取功能需要）
- volcenginesdkarkruntime >= 1.0.0（视觉分析功能需要）
- mcp >= 1.0.0（MCP 服务器需要）

---

## 示例脚本

### 机械臂舞蹈

```bash
python scripts/dance.py
```

### 机械臂姿态动作（挥手、点头、摇头）

```bash
# 挥挥手
python scripts/arm_gestures.py wave
python scripts/arm_gestures.py wave --times 5  # 挥手5次

# 点点头
python scripts/arm_gestures.py nod

# 摇摇头
python scripts/arm_gestures.py shake

# 依次执行全部动作
python scripts/arm_gestures.py all
```

### 自主视觉抓取

```bash
python scripts/grab_optimized.py --target "桌上的红色苹果"
```

---

## 故障排除

### 无法连接到机器人服务 / 连接超时

MCP 工具调用前会检测机器人服务是否可达。如果提示“无法连接到机器人服务”，请按以下步骤排查：

1. 确认机器人已开机并连接到同一局域网；
2. 确认机器人后台服务已启动（运动、机械臂、视觉服务）：
   ```bash
   # 在机器人本机或启动服务的电脑上检查端口
   lsof -i :5556   # 底盘
   lsof -i :5557   # 机械臂
   lsof -i :5560   # 主摄像头
   ```
3. 确认 `HOMEBOT_IP` 与实际机器人 IP 一致（当前默认: `192.168.1.13`）：
   ```bash
   # 测试网络连通性
   ping 192.168.1.13
   ```
4. 如需修改 IP，可在启动 MCP 前设置环境变量：
   ```bash
   export HOMEBOT_IP=你的机器人IP
   ```
   或在 OpenClaw MCP 配置的 `env` 中修改 `HOMEBOT_IP`。

### 视觉分析失败

检查 ARK_API_KEY 是否已正确设置：
```bash
# Windows
echo %ARK_API_KEY%

# Linux/Mac
echo $ARK_API_KEY
```

### 自主抓取提示 "OpenCV Tracker 不可用"

安装 OpenCV：
```bash
pip install opencv-python numpy
```

### 自主抓取在 Phase 2 失败 "末端摄像头不可用"

确保末端摄像头服务已启动，并在 `scripts/robot_config.py` 中正确配置 `END_VIDEO_PORT`（默认 5561）。

### 端口冲突

检查端口是否被占用：
```bash
# Windows
netstat -ano | findstr 5556

# Linux/Mac
lsof -i :5556
```
