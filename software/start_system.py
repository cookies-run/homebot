#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HomeBot System Launcher
跨平台启动器，支持 Windows/Linux/macOS
"""

import os
import sys
import socket
import subprocess
import signal
import time
import platform
from pathlib import Path


# 服务配置
SERVICES = [
    {
        "name": "Motion Service",
        "module": "services.motion_service",
        "port": 5556,  # 底盘服务端口
        "port2": 5557,  # 机械臂服务端口
        "desc": "Chassis + Arm Service",
        "args": ["--service", "both"]
    },
    {
        "name": "Vision Service",
        "module": "services.vision_service",
        "port": 5560,
        "desc": "Body Camera Vision Service",
        "args": ["--addr", "tcp://*:5560", "--device", "1"]
    },
    {
        "name": "End Vision Service",
        "module": "services.vision_service",
        "port": 5561,
        "desc": "End Camera Vision Service",
        "args": ["--addr", "tcp://*:5561", "--device-name", "USB摄像头"]
    },
    {
        "name": "WakeupASR Service",
        "module": "services.speech_service",
        "port": 5571,
        "desc": "Voice Wakeup + ASR (PUB)",
        "args": ["wakeup"]
    },
    {
        "name": "Speech Interaction",
        "module": "applications.speech_interaction",
        "port": None,  # SUB模式，不绑定端口
        "desc": "Voice Dialogue + TTS (SUB)"
    },
    {
        "name": "Web Control",
        "module": "applications.remote_control",
        "port": 5002,
        "desc": "Web Server",
        "args": [
            "--host", "0.0.0.0",
            "--port", "5002",
            "--vision", "tcp://127.0.0.1:5560",
            "--end-vision", "tcp://127.0.0.1:5561"
        ]
    }
]


def print_header():
    """打印启动标题"""
    print("=" * 50)
    print("   HomeBot System Launcher")
    print("=" * 50)
    print()


def check_port(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False  # 端口可用
        except socket.error:
            return True  # 端口被占用


def get_process_on_port(port):
    """获取占用端口的进程 PID（跨平台）"""
    try:
        if platform.system() == "Windows":
            # Windows: 使用 netstat 和 findstr
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        return int(parts[-1])
        else:
            # Linux/macOS: 使用 lsof
            result = subprocess.run(
                ["lsof", "-i", f"TCP:{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return None


def kill_process(pid):
    """终止进程"""
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], 
                         capture_output=True, check=False)
        else:
            os.kill(pid, signal.SIGKILL)
        return True
    except Exception:
        return False


def check_ports():
    """检查所有端口状态"""
    print("[Check] Checking required ports...")
    print()
    
    occupied = []
    for svc in SERVICES:
        # 跳过无端口的服务（如SUB模式应用）
        if svc.get("port") is None:
            print(f"[OK] {svc['name']} (no port required)")
            continue
            
        # 检查主端口
        if check_port(svc["port"]):
            print(f"[WARN] Port {svc['port']} is occupied ({svc['desc']})")
            occupied.append(svc)
        else:
            print(f"[OK] Port {svc['port']} is available")
        
        # 检查第二端口（如果有）
        if "port2" in svc:
            if check_port(svc["port2"]):
                print(f"[WARN] Port {svc['port2']} is occupied ({svc['desc']} - Arm)")
                if svc not in occupied:
                    occupied.append(svc)
            else:
                print(f"[OK] Port {svc['port2']} is available")
    
    print()
    return occupied


def prompt_user(occupied):
    """提示用户处理被占用的端口"""
    print("=" * 50)
    print("[WARNING] Some ports are already in use!")
    print()
    print("This may cause services to fail starting.")
    print()
    print("Options:")
    print("   1. Kill occupying processes and continue")
    print("   2. Continue anyway (may cause errors)")
    print("   3. Exit")
    print("=" * 50)
    print()
    
    try:
        choice = input("Select option [1-3]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n[Exit] User cancelled.")
        return False
    
    if choice == "3":
        print("[Exit] User cancelled.")
        return False
    elif choice == "2":
        print("[Continue] Starting with warnings...")
        print()
        return True
    elif choice == "1":
        print("[Action] Killing processes on occupied ports...")
        for svc in occupied:
            pid = get_process_on_port(svc["port"])
            if pid:
                print(f"   Killing PID {pid} on port {svc['port']}...")
                kill_process(pid)
        print("[OK] Cleanup complete")
        time.sleep(2)
        print()
        return True
    else:
        print("[Exit] Invalid option.")
        return False


# 全局存储启动的进程，方便停止
_started_processes = []


def get_venv_python(script_dir):
    """获取虚拟环境中的 Python 解释器路径"""
    # 检查当前是否已在 venv 中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.prefix != sys.base_prefix):
        return sys.executable

    # 尝试找到项目 venv
    venv_python = Path(script_dir).parent / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)

    # Windows 路径
    venv_python_win = Path(script_dir).parent / "venv" / "Scripts" / "python.exe"
    if venv_python_win.exists():
        return str(venv_python_win)

    return sys.executable


def start_service(svc, src_dir, python_exec, log_dir):
    """启动单个服务（在当前终端后台运行）"""
    print(f"[Start] Starting {svc['name']}...")

    cmd = [python_exec, "-m", svc["module"]]

    # 添加额外参数（如果有）
    if "args" in svc:
        cmd.extend(svc["args"])

    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir)

    # 日志文件
    log_file = log_dir / f"{svc['module'].replace('.', '_')}.log"

    # 在当前终端后台运行，输出写入日志文件
    log_fp = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=src_dir,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        env=env,
    )

    _started_processes.append({
        "name": svc["name"],
        "proc": proc,
        "log": log_file,
    })

    print(f"   [PID {proc.pid}] {svc['name']} -> {log_file}")


def stop_all_services():
    """停止所有已启动的服务"""
    print()
    print("=" * 50)
    print("[Stop] Stopping all services...")
    for item in _started_processes:
        proc = item["proc"]
        if proc.poll() is None:  # 还在运行
            print(f"   Stopping {item['name']} (PID {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    print("[OK] All services stopped.")
    print("=" * 50)


def main():
    """主函数"""
    print_header()

    # 检查端口
    occupied = check_ports()

    # 如果有端口被占用，询问用户
    if occupied:
        if not prompt_user(occupied):
            input("\nPress Enter to exit...")
            sys.exit(1)
    else:
        print("[OK] All ports are available")
        print()

    # 切换到 src 目录
    script_dir = Path(__file__).parent
    src_dir = script_dir / "src"

    if not src_dir.exists():
        print(f"[Error] Directory not found: {src_dir}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # 日志目录
    log_dir = script_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # 获取 Python 解释器
    python_exec = get_venv_python(script_dir)
    print(f"[Info] Python: {python_exec}")
    print(f"[Info] Log dir: {log_dir}")
    print()

    # 注册退出清理
    def signal_handler(signum, _frame):
        print(f"\n[Signal] Received signal {signum}")
        stop_all_services()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动所有服务
    for svc in SERVICES:
        start_service(svc, str(src_dir), python_exec, log_dir)
        if svc != SERVICES[-1]:
            time.sleep(1)

    # 等待服务启动
    print()
    print("Waiting for services to start...")
    time.sleep(3)

    # 检查启动状态
    running = sum(1 for item in _started_processes if item["proc"].poll() is None)
    print(f"[Status] {running}/{len(_started_processes)} services running")
    print()

    # 打印完成信息
    print("=" * 50)
    print("[OK] All services started!")
    print()
    print("Services:")
    for item in _started_processes:
        proc = item["proc"]
        status = "Running" if proc.poll() is None else f"Exit {proc.poll()}"
        print(f"   - {item['name']} [{status}] -> {item['log']}")
    print()

    # 从 SERVICES 中获取 Web 控制端的实际端口
    web_port = 5002
    for svc in SERVICES:
        if svc["name"] == "Web Control":
            web_port = svc.get("port", 5002)
            break

    # 获取局域网IP
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "127.0.0.1"

    print("访问地址:")
    print(f"   本机:     http://localhost:{web_port}")
    if lan_ip != "127.0.0.1":
        print(f"   局域网:   http://{lan_ip}:{web_port}")
    print(f"   机身视频: http://localhost:{web_port}/video_feed")
    print(f"   末端视频: http://localhost:{web_port}/end_video_feed")
    print("=" * 50)
    print()
    print("按 Ctrl+C 停止所有服务")
    print()

    # 保持运行，等待用户中断
    try:
        while True:
            time.sleep(1)
            # 检查是否有服务异常退出
            for item in _started_processes:
                if item["proc"].poll() is not None and item["proc"].poll() != 0:
                    print(f"[WARN] {item['name']} exited with code {item['proc'].poll()}")
                    print(f"       Check log: {item['log']}")
    except KeyboardInterrupt:
        pass
    finally:
        stop_all_services()


if __name__ == "__main__":
    main()
