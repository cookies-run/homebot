---
name: homebot
description: HomeBot 机器人控制器技能。向 openclaw/LLM 暴露一组细粒度确定性 MCP 工具：底盘运动、机械臂关节控制、视觉查询，以及递送任务所需的原子技能——查找(search_target)、接近(approach_target)、精细抓取(auto_grab)、递送放置(deliver)。整体任务拆解与流程编排由 openclaw 负责，机器人端只执行单步动作并返回结构化反馈。当用户发出抓取/拿取命令时调用 auto_grab；当用户发出“把 X 递给 Y”这类递送命令时，由 openclaw 依次编排 search_target→approach_target→auto_grab→search_target→approach_target→deliver。全部基于 ZeroMQ 局域网通信协议。
version: 1.1.0
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

HomeBot 机器人控制器技能，向 openclaw 暴露**细粒度、无状态、单步返回反馈**的 MCP 工具。

**设计原则**：任务拆解与流程编排在 openclaw 侧完成；机器人端每个 MCP 工具只做一件事，并返回结构化反馈（成功/失败、位置、距离等），由 openclaw 据此决定下一步。机器人端**不做**多步状态机编排。

功能模块：
- 🚗 **底盘控制**：前进/后退/转向，精确距离角度控制
- 🦾 **机械臂控制**：6 自由度关节控制，夹爪控制，回原点
- 🔎 **查找 (search_target)**：旋转扫描定位单个目标，返回 bbox/高度/姿态/可抓性
- 🚶 **接近 (approach_target)**：底盘 PID 接近目标到指定距离（初步定位）
- 🤏 **精细抓取 (auto_grab)**：六阶段视觉引导状态机，完成末端精对准与抓取
- 📦 **递送 (deliver)**：机械臂伸出→松爪释放→复位
- 👁️ **视觉查询 (robot_what_does_robot_see)**：捕获画面并用 VLM 分析场景

全部基于 ZeroMQ REQ-REP / PUB 局域网通信协议。

---

## 递送任务的标准编排（由 openclaw 负责）

以指令“抓取纸巾，送到穿黑色上衣的人面前”为例，openclaw 应拆解并依次调用以下 MCP 工具：

| 步骤 | 动作 | MCP 工具 | 说明 |
|------|------|---------|------|
| 1 | 查找抓取目标 | `search_target(target="纸巾")` | 旋转扫描，found=true 时返回 bbox |
| 2 | 接近抓取目标 | `approach_target(target="纸巾", distance_cm=40)` | 底盘开到目标前约 40cm |
| 3 | 精细抓取 | `auto_grab(target="纸巾")` | 末端相机精对准并夹取 |
| 4 | 查找递送目标 | `search_target(target="穿黑色上衣的人")` | 旋转扫描定位人 |
| 5 | 接近递送目标 | `approach_target(target="穿黑色上衣的人", distance_cm=40)` | 底盘开到人前约 40cm |
| 6 | 递送放置 | `deliver(target="穿黑色上衣的人")` | 伸出机械臂、松爪释放、复位 |

**关键约定：**
- `search_target` 与 `approach_target` 是**初步定位**（把机器人开到目标正前方）；`auto_grab` 才是**精细抓取**（内含末端相机二次定位与对准）。三者职责不重叠、不冲突：先粗定位到位，再由 auto_grab 精抓。
- 每一步都会返回结构化 JSON。openclaw 应检查 `found`/`success` 字段：
  - `search_target` 返回 `found=false` → openclaw 可让机器人移动到别处再搜。
  - `approach_target` 返回 `success=false` → 可重试或先重新 `search_target`。
- **一次只处理一个目标**。抓取目标与递送目标需分两轮 `search_target`/`approach_target`。
- 纯抓取命令（无递送）可简化为：`search_target` →（可选 `approach_target`）→ `auto_grab`；若目标已在近前，也可直接调用 `auto_grab`（它内部会自行观察与粗定位）。

---

## Agent 调用触发词

**抓取类：**

| 触发类型 | 示例命令 | 编排 |
|---------|---------|------|
| 抓取/拿起/拿取 | “抓取桌上的纸巾”、“帮我拿瓶矿泉水” | (search_target →) approach_target → auto_grab |
| 取物/捡起 | “取一下遥控器”、“捡起地上的玩具” | 同上 |

**递送类：**

| 触发类型 | 示例命令 | 编排 |
|---------|---------|------|
| 递给人 | “把矿泉水递给穿红衣服的人” | 上表 6 步完整流程 |
| 递到地点 | “把杯子拿到餐桌上” | 6 步流程（第 4-5 步定位地点） |

---

## MCP 工具清单

| 工具名称 | 功能描述 |
|---------|---------|
| `chassis_forward` | 机器人前进指定距离（厘米） |
| `chassis_backward` | 机器人后退指定距离（厘米） |
| `chassis_left` | 机器人左转指定角度（度） |
| `chassis_right` | 机器人右转指定角度（度） |
| `chassis_stop` | 紧急停止机器人底盘 |
| `arm_move_joint` | 移动机械臂指定关节到目标角度 |
| `arm_get_positions` | 获取机械臂所有关节当前位置 |
| `arm_stop` | 停止机械臂所有运动 |
| `search_target` | **查找**：旋转扫描定位单个目标，返回 bbox/高度/姿态/可抓性 |
| `approach_target` | **接近**：底盘 PID 接近目标到指定距离（默认 40cm，初步定位） |
| `auto_grab` | **精细抓取**：六阶段视觉引导状态机，完成末端精对准与夹取 |
| `deliver` | **递送**：机械臂伸出 → 松爪释放 → 复位 |
| `robot_what_does_robot_see` | 捕获机器人画面并用 VLM 分析场景 |

> 说明：旧版的 `delivery_plan` / `delivery_confirm` / `delivery_status` 三个工具已移除。
> 递送流程的拆解与编排已上移到 openclaw，机器人端不再暴露整包递送状态机。

---

## 快速开始

### 1. 配置机器人连接（推荐用环境变量）

```bash
# Linux/Mac
export HOMEBOT_IP=192.168.1.13

# Windows
set HOMEBOT_IP=192.168.1.13
```

或修改 `scripts/robot_config.py` 中的 `ROBOT_IP`。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 测试连接

```bash
python scripts/chassis_control.py forward 10        # 底盘
python scripts/arm_control.py status                # 机械臂
python scripts/what_does_robot_see_workflow.py --no-analysis   # 视觉
python scripts/grab_optimized.py --target "一包纸巾"           # 精细抓取
```

---

## 环境变量配置

优先级：**环境变量 > 配置文件默认值**

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `HOMEBOT_IP` | 机器人 IP 地址 | `192.168.1.13` |
| `HOMEBOT_CHASSIS_PORT` | 底盘服务端口 | `5556` |
| `HOMEBOT_ARM_PORT` | 机械臂服务端口 | `5557` |
| `HOMEBOT_VIDEO_PORT` | 机身摄像头视频流端口 | `5560` |
| `HOMEBOT_END_VIDEO_PORT` | 机械臂末端摄像头端口 | `5561` |
| `HOMEBOT_CAPTURE_TIMEOUT` | 图像捕获超时(秒) | `10.0` |
| `HOMEBOT_OUTPUT_DIR` | 图像保存目录 | `.` |
| `HOMEBOT_SKIP_REACHABILITY_CHECK` | 跳过调用前的可达性检测 | `false` |
| `ARK_API_KEY` | 火山引擎 API Key（视觉分析回退用） | - |
| `ARK_MODEL_ID` | 火山引擎模型 ID | `doubao-seed-2-0-lite-260215` |

> VLM 提供方优先 MiniMax，缺失时回退火山引擎（详见 `scripts/minimax_vision_client.py` / `scripts/volcengine_vision_client.py`）。

### 连接前检测

所有 MCP 工具在执行前会先检测机器人服务端口是否可达（TCP 连接 `5556/5557/5560` 等，任一端口通即视为在线）。检测失败会立即返回提示（含当前 IP 与排查步骤），并打印 `[HomeBot]` 前缀诊断日志。若确定网络已通但被误拦，可设 `HOMEBOT_SKIP_REACHABILITY_CHECK=1` 跳过。

### openclaw MCP 配置示例

```yaml
mcp:
  servers:
    homebot:
      command: "python"
      args:
        - "{{skill_path}}/mcp_homebot_server.py"
      env:
        HOMEBOT_IP: "192.168.1.13"
        HOMEBOT_END_VIDEO_PORT: "5561"
        # 视觉分析回退用（可选）
        ARK_API_KEY: "your-volcengine-api-key"
        ARK_MODEL_ID: "doubao-seed-2-0-lite-260215"
```

---

## 工具详解

### 1. 底盘控制 (chassis_*)

精确控制底盘运动。命令行等价用法：

```bash
python scripts/chassis_control.py forward 10     # 前进 10cm
python scripts/chassis_control.py backward 20    # 后退 20cm
python scripts/chassis_control.py right 90       # 右转 90°
python scripts/chassis_control.py left 45        # 左转 45°
python scripts/chassis_control.py stop           # 紧急停止
```

### 2. 机械臂控制 (arm_*)

6 自由度关节控制。关节名：`base` / `shoulder` / `elbow` / `wrist_flex` / `wrist_roll` / `gripper`。
优先级：emergency(4) > auto(3) > voice(2) > web(1)。

```bash
python scripts/arm_control.py joint base 0
python scripts/arm_control.py gripper open       # 打开夹爪(90°)
python scripts/arm_control.py gripper close      # 关闭夹爪(0°)
python scripts/arm_control.py home               # 回原点
python scripts/arm_control.py status             # 查询各关节角度
```

### 3. 查找 (search_target)

`search_target(target)` — 旋转扫描定位**单个**目标。

- 先看当前画面；找不到则左转 120° 再看，最多 3 次（约转满一圈）。
- 返回 JSON：`found`、`bbox`(归一化 xyxy)、`height_cm`、`pose`、`graspable`、`scans_used`。
- `found=false` 时由 openclaw 决定是否移动到别处再搜。
- 底层：`scripts/delivery/search_skill.py`（VLM 视觉搜索）。

### 4. 接近 (approach_target)

`approach_target(target, distance_cm=40)` — 底盘 PID 接近目标到指定距离（**初步定位**）。

- 先用 VLM 定位一次拿到初始 bbox（要求目标大致在当前画面内，通常先 `search_target`），再基于底盘速度 PID 前进，过程中周期性重定位使目标框随接近放大，直至估计距离达到 `distance_cm` 或超时。
- 返回 JSON：`success`、`message`、`final_distance_cm`。
- 底层：`scripts/delivery/approach_skill.py`。
- **距离说明**：`distance_cm` 当前是基于目标框面积的**反推估计**（框面积≈0.08 时约 30cm，按平方反比缩放），**非真实深度**。误差受目标真实尺寸影响。后续会用**手眼外参标定**替换为更精确的距离感知，当前先用反推值。

### 5. 精细抓取 (auto_grab)

`auto_grab(target, use_end_camera=True)` — 六阶段视觉引导抓取状态机：

- Phase 0 观察复位 → Phase 1 VLM 属性测姿 + 底盘策略分流 → Phase 2 接近与手腕预部署（末端相机追踪）→ Phase 3 纯几何二次贴紧 → Phase 4 VLM 触达/对齐终审 → Phase 5 夹紧抬升回缩。
- **这是精细抓取环节**：即使前面已 `approach_target` 到位，auto_grab 仍会用末端相机做二次精定位与对准，与前面的粗定位不冲突。
- 前置：机身摄像头(5560) + 末端摄像头(5561) 已发布画面；已配置 VLM Key。
- 底层：`scripts/grab_optimized.py`（命令行：`python scripts/grab_optimized.py --target "一包纸巾"`）。

### 6. 递送 (deliver)

`deliver(target="")` — 机械臂伸到目标前 → 打开夹爪释放 → 复位。

- **假设机器人已用 `approach_target` 接近到位**，本工具不移动底盘。
- 返回 JSON：`success`、`message`。
- 底层：`scripts/delivery/place_skill.py`。

### 7. 视觉查询 (robot_what_does_robot_see)

一键捕获机身摄像头画面并用 VLM 分析。命令行：

```bash
python scripts/what_does_robot_see_workflow.py
python scripts/what_does_robot_see_workflow.py --no-analysis      # 仅捕获
python scripts/what_does_robot_see_workflow.py --prompt "图中有几个人？"
```

---

## 可复用技能原语

递送相关工具由以下技能原语组合，位于 `scripts/delivery/`，均为自包含实现（不依赖 `applications/` 包）：

| 技能 | 功能 | 源文件 |
|------|------|--------|
| `SearchSkill` | VLM 视觉搜索目标，返回 bbox/高度/姿态/可抓性 | `search_skill.py` |
| `ApproachSkill` | 基于跟踪器 + 周期性重定位的底盘 PID 接近 | `approach_skill.py` |
| `PlaceSkill` | 释放姿态 + 打开夹爪 + 机械臂复位 | `place_skill.py` |
| `GraspSkill` | 封装 `grab_optimized.py` 的自主抓取 | `grasp_skill.py` |
| `TargetTracker` | 轻量 IoU 跟踪器，带目标切换迟滞 | `tracker_skill.py` |
| `ChassisAdapter` / `ArmAdapter` / `VisionAdapter` | ZeroMQ 服务通信适配层 | `adapters.py` |

---

## 通信协议

| 服务 | 模式 | 默认端口 |
|------|------|----------|
| 底盘控制 | REQ-REP | 5556 |
| 机械臂控制 | REQ-REP | 5557 |
| 机身摄像头 | PUB | 5560 |
| 末端摄像头 | PUB | 5561 |

---

## 依赖

- Python 3.8+
- pyzmq >= 25.0.0
- Pillow >= 9.0.0
- numpy >= 1.24.0
- opencv-python >= 4.8.0（查找/接近/抓取需要）
- volcenginesdkarkruntime >= 1.0.0（视觉分析回退需要）
- mcp >= 1.0.0（MCP 服务器需要）

---

## 故障排除

### 无法连接到机器人服务 / 连接超时

MCP 工具调用前会检测端口可达性。若提示“无法连接”：
1. 确认机器人已开机并接入同一局域网；
2. 确认后台服务已启动（`lsof -i :5556` / `:5557` / `:5560`）；
3. 确认 `HOMEBOT_IP` 与实际一致（`ping 192.168.1.13`）；
4. 网络已通但被误拦时可设 `HOMEBOT_SKIP_REACHABILITY_CHECK=1`。

### 搜索/接近/递送技能不可用

工具返回“技能不可用/初始化失败”时，通常是 `scripts/delivery/` 依赖（`software/src` 下的 `common`/`configs`/`services`）未就绪或视觉/底盘/机械臂适配器无法连接。检查上述服务端口与 Python 依赖。

### 接近一直超时 / 到不了位

`approach_target` 依赖周期性 VLM 重定位来收敛。若一直超时：确认机身摄像头有画面、VLM Key 已配置、目标在画面内（先 `search_target` 转向目标）。

### 精细抓取在 Phase 2 失败 “末端摄像头不可用”

确保末端摄像头服务已启动，`scripts/robot_config.py` 中 `END_VIDEO_PORT`（默认 5561）配置正确。

### 视觉分析失败

检查 VLM Key（MiniMax 或 `ARK_API_KEY`）是否正确设置。
