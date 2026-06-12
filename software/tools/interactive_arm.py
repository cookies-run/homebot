#!/usr/bin/env python3
"""
Interactive Arm Debugger - 交互式机械臂关节调试工具

结合机械臂 CAD 图使用：
  ① base       - 基座旋转 (J1)
  ② shoulder   - 肩关节   (J2) - 控制大臂抬/落
  ③ elbow      - 肘关节   (J3) - 控制小臂伸/屈
  ④ wrist_flex - 腕屈伸   (J4) - 控制末端俯仰
  ⑤ wrist_roll - 腕旋转   (J5) - 控制末端自旋
  ⑥ gripper    - 夹爪     (J6) - 0=闭合, 90=张开

连杆长度 (参考图):
  大臂 (②→③): ~116mm
  小臂 (③→④): ~135mm

用法:
    cd ~/homebot/software/src
    python ../tools/interactive_arm.py

命令:
    [关节号][+/-][角度]  如: 1+5, 2-10, 3=45
    [关节号]d            关节失能 (可手动掰动)
    g[角度]              夹爪角度, 如 g0, g45, g90
    h / home             一键复位到休息位置
    r                    读取当前状态
    step [1/5/15]        切换默认步进
    s [name]             保存当前姿态到文件
    l [name]             加载姿态
    list                 列出已保存的姿态
    move                 多关节同时移动 (输入 JSON)
    q / quit / exit      退出
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from configs import get_config
from hal.ftservo_driver import FTServoBus


# 从 configs 读取实际限位，避免硬编码不一致
cfg = get_config()
_lim = cfg.arm.joint_limits
ARM_JOINTS = {
    1: ("base", "基座旋转", _lim["base"][0], _lim["base"][1]),
    2: ("shoulder", "肩关节", _lim["shoulder"][0], _lim["shoulder"][1]),
    3: ("elbow", "肘关节", _lim["elbow"][0], _lim["elbow"][1]),
    4: ("wrist_flex", "腕屈伸", _lim["wrist_flex"][0], _lim["wrist_flex"][1]),
    5: ("wrist_roll", "腕旋转", _lim["wrist_roll"][0], _lim["wrist_roll"][1]),
    6: ("gripper", "夹爪", _lim["gripper"][0], _lim["gripper"][1]),
}

SAVE_DIR = os.path.expanduser("~/.homebot/arm_poses")
os.makedirs(SAVE_DIR, exist_ok=True)


def angle_to_pos(angle: float) -> int:
    return int(2048 + angle * 11.377)


def pos_to_angle(pos: int) -> float:
    return (pos - 2048) / 11.377


def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def print_status(bus: FTServoBus, step: int):
    """打印当前机械臂状态表格"""
    print("\n" + "=" * 65)
    print("  交互式机械臂调试器")
    print("  当前步进: ±{}°".format(step))
    print("=" * 65)
    print(f"  {'#':<3} {'关节':<12} {'描述':<10} {'当前角度':<10} {'当前位置':<10} {'范围':<15}")
    print("  " + "-" * 61)

    for jid, (name, desc, lo, hi) in ARM_JOINTS.items():
        pos = bus.read_position(jid)
        if pos is not None:
            angle = pos_to_angle(pos)
            marker = ""
            if jid == 6:
                state = "张开" if angle > 45 else "闭合"
                marker = f" [{state}]"
            print(f"  {jid:<3} {name:<12} {desc:<10} {angle:>7.1f}°   {pos:>7}    [{lo},{hi}]{marker}")
        else:
            print(f"  {jid:<3} {name:<12} {desc:<10} {'N/A':>7}     {'N/A':>7}    [{lo},{hi}]")

    print("=" * 65)


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def move_joint(bus: FTServoBus, jid: int, delta: float) -> bool:
    """相对移动单个关节"""
    name, desc, lo, hi = ARM_JOINTS[jid]
    pos = bus.read_position(jid)
    if pos is None:
        print(f"[ERR] 无法读取 {name} 当前位置")
        return False
    cur_angle = pos_to_angle(pos)
    target = clamp(cur_angle + delta, lo, hi)
    target_pos = angle_to_pos(target)

    if jid == 6:
        bus.write_position(jid, target_pos, speed=500, acc=50)
    else:
        bus.write_position(jid, target_pos, speed=800, acc=50)
    print(f"[OK] {name}({desc}): {cur_angle:.1f}° → {target:.1f}°")
    time.sleep(0.3)
    return True


def set_joint(bus: FTServoBus, jid: int, angle: float) -> bool:
    """绝对设置单个关节角度"""
    name, desc, lo, hi = ARM_JOINTS[jid]
    target = clamp(angle, lo, hi)
    target_pos = angle_to_pos(target)

    bus.write_position(jid, target_pos, speed=800, acc=50)
    print(f"[OK] {name}({desc}) → {target:.1f}°")
    time.sleep(0.3)
    return True


def disable_joint(bus: FTServoBus, jid: int):
    """失能单个关节"""
    name, desc, _, _ = ARM_JOINTS[jid]
    bus.torque_disable(jid)
    print(f"[OK] {name}({desc}) 扭矩已失能，可手动调整")


def _write_joint(bus: FTServoBus, jid: int, angle: float, speed: int = 800, acc: int = 50):
    """安全地写入单个关节角度，并验证位置变化"""
    pos = angle_to_pos(angle)
    before = bus.read_position(jid)
    ok = bus.write_position(jid, pos, speed=speed, acc=acc)
    time.sleep(0.2)
    after = bus.read_position(jid)
    print(f"  [DEBUG] ID={jid} target={angle:.1f}°({pos}) before={before} after={after} -> {'OK' if ok else 'FAIL'}")


def home_all(bus: FTServoBus):
    """一键复位到休息位置"""
    cfg = get_config()
    rest = cfg.arm.rest_position
    print("\n[HOME] 正在复位到休息位置...")
    for jid, (name, _, _, _) in ARM_JOINTS.items():
        angle = rest.get(name, 0)
        _write_joint(bus, jid, angle)
    time.sleep(1.5)
    print("[HOME] 复位完成")


def high_pose(bus: FTServoBus):
    """一键到达高举避碰姿态（末端在基座前方116mm、上方135mm）"""
    print("\n[HIGH] 正在移动到高举避碰姿态...")
    print("       shoulder=90°  elbow=90°  wrist_flex=0°  gripper=90°")
    _write_joint(bus, 2, 90)   # shoulder
    _write_joint(bus, 3, 90)   # elbow
    _write_joint(bus, 4, 0)    # wrist_flex
    _write_joint(bus, 6, 90, speed=500)  # gripper
    time.sleep(1.5)
    print("[HIGH] 高举到位，末端在安全高度")


def save_pose(bus: FTServoBus, name: str):
    """保存当前姿态"""
    pose = {}
    for jid, (jname, _, _, _) in ARM_JOINTS.items():
        pos = bus.read_position(jid)
        pose[jname] = pos_to_angle(pos) if pos is not None else 0
    path = os.path.join(SAVE_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(pose, f, indent=2)
    print(f"[SAVE] 姿态 '{name}' 已保存到 {path}")


def load_pose(bus: FTServoBus, name: str):
    """加载姿态"""
    path = os.path.join(SAVE_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"[ERR] 姿态 '{name}' 不存在")
        return
    with open(path, 'r') as f:
        pose = json.load(f)
    print(f"\n[LOAD] 正在加载姿态 '{name}'...")
    positions = {}
    for jid, (jname, _, _, _) in ARM_JOINTS.items():
        angle = pose.get(jname, 0)
        positions[jid] = (angle_to_pos(angle), 800, 50)
    bus.sync_write_positions(positions)
    time.sleep(1.5)
    print(f"[LOAD] 姿态 '{name}' 加载完成")


def list_poses():
    """列出已保存的姿态"""
    files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith('.json')])
    if not files:
        print("[INFO] 没有已保存的姿态")
        return
    print("\n[SAVED POSES]")
    for f in files:
        name = f[:-5]
        path = os.path.join(SAVE_DIR, f)
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        with open(path) as fh:
            pose = json.load(fh)
        print(f"  - {name:<15} ({mtime})  base={pose.get('base',0):.0f} shoulder={pose.get('shoulder',0):.0f} elbow={pose.get('elbow',0):.0f}")


def multi_move(bus: FTServoBus):
    """多关节同时移动"""
    print("\n输入 JSON 格式目标角度，例如:")
    print('  {"base": 0, "shoulder": -30, "elbow": 90, "wrist_flex": 0, "wrist_roll": 0, "gripper": 45}')
    print('  或简写: {"2": -30, "3": 90}')
    raw = input("> ").strip()
    if not raw:
        return
    try:
        targets = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[ERR] JSON 格式错误: {e}")
        return

    positions = {}
    for jid, (name, _, lo, hi) in ARM_JOINTS.items():
        key = name
        if str(jid) in targets:
            key = str(jid)
        if key in targets:
            angle = clamp(float(targets[key]), lo, hi)
            positions[jid] = (angle_to_pos(angle), 800, 50)

    if positions:
        bus.sync_write_positions(positions)
        time.sleep(1.5)
        print(f"[OK] 多关节移动完成: {list(positions.keys())}")
    else:
        print("[ERR] 没有有效的关节目标")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='交互式机械臂调试')
    parser.add_argument('--port', default=None, help='串口设备')
    args = parser.parse_args()

    cfg = get_config()
    port = args.port or cfg.arm.serial_port
    baudrate = cfg.arm.baudrate

    print("=" * 65)
    print("交互式机械臂关节调试工具")
    print("=" * 65)
    print(f"串口: {port}")
    print(f"波特率: {baudrate}")

    bus = FTServoBus(port, baudrate)
    if not bus.connect():
        print("[ERR] 串口连接失败")
        sys.exit(1)
    print("[OK] 串口已连接\n")

    # 先逐个使能扭矩（广播在某些固件上不稳定）
    print("[DEBUG] 逐个使能舵机扭矩...")
    for jid in range(1, 7):
        ok = bus.torque_enable(jid)
        print(f"  ID={jid} torque_enable -> {'OK' if ok else 'FAIL'}")
    time.sleep(0.3)
    print("[DEBUG] 扭矩使能完成")

    step = 5  # 默认步进

    try:
        while True:
            clear_screen()
            print_status(bus, step)
            print("\n命令: [关节][+/-][角度]  g[角度]  h(ome)  high  r(ead)  step[N]  s(ave)  l(oad)  list  move  q(uit)")
            print("示例: 1+5  2-10  3=45  g0  g90  high  step1  step15  s rest  l rest  move")
            raw = input("> ").strip().lower()

            if not raw:
                continue
            if raw in ('q', 'quit', 'exit'):
                break
            if raw in ('h', 'home'):
                home_all(bus)
                continue
            if raw == 'high':
                high_pose(bus)
                continue
            if raw == 'r':
                continue  # 刷新状态
            if raw == 'list':
                list_poses()
                input("\n按回车继续...")
                continue
            if raw == 'move':
                multi_move(bus)
                input("\n按回车继续...")
                continue
            if raw.startswith('step'):
                try:
                    step = int(raw[4:])
                    print(f"[OK] 步进已设为 ±{step}°")
                    time.sleep(0.5)
                except ValueError:
                    print("[ERR] step 后需要数字，如 step1, step5, step15")
                    time.sleep(1)
                continue
            if raw.startswith('s '):
                save_pose(bus, raw[2:].strip())
                time.sleep(0.5)
                continue
            if raw.startswith('l '):
                load_pose(bus, raw[2:].strip())
                time.sleep(0.5)
                continue

            # 解析 [关节][+/-][角度]  如 1+5, 2-10, 3=45
            # 也支持 [关节]d 失能
            import re
            m = re.match(r'^(\d+)([\+\-=])([\d\.]+)$', raw)
            if m:
                jid = int(m.group(1))
                op = m.group(2)
                val = float(m.group(3))
                if jid not in ARM_JOINTS:
                    print(f"[ERR] 无效关节号: {jid}")
                    time.sleep(1)
                    continue
                if op == '=':
                    set_joint(bus, jid, val)
                else:
                    delta = val if op == '+' else -val
                    move_joint(bus, jid, delta)
                continue

            # 失能命令: 1d, 2d, ...
            m = re.match(r'^(\d+)d$', raw)
            if m:
                jid = int(m.group(1))
                if jid in ARM_JOINTS:
                    disable_joint(bus, jid)
                else:
                    print(f"[ERR] 无效关节号: {jid}")
                time.sleep(0.5)
                continue

            # 夹爪命令: g, g0, g45, g90
            m = re.match(r'^g([\d\.]+)?$', raw)
            if m:
                angle = float(m.group(1)) if m.group(1) else None
                if angle is not None:
                    set_joint(bus, 6, angle)
                else:
                    # 切换
                    pos = bus.read_position(6)
                    cur = pos_to_angle(pos) if pos else 45
                    target = 0 if cur > 45 else 90
                    set_joint(bus, 6, target)
                continue

            print(f"[ERR] 未知命令: '{raw}'，输入 h 查看帮助")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        print("\n失能扭矩并关闭串口...")
        bus.torque_disable(-1)
        bus.disconnect()
        print("[完成]")


if __name__ == '__main__':
    main()
