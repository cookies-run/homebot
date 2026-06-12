"""
motion_service 模块入口
可以同时启动底盘和机械臂服务，支持共享串口总线
"""
import sys
import os
import argparse
import time
import threading

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from configs import get_config
from services.motion_service.servo_bus_manager import ServoBusManager


def main():
    parser = argparse.ArgumentParser(description='HomeBot 运动控制服务')
    parser.add_argument('--service', choices=['chassis', 'arm', 'both'], 
                       default='both', help='启动哪个服务 (默认: both)')
    parser.add_argument('--port', default=None, help='串口设备（覆盖配置）')
    parser.add_argument('--chassis-addr', default=None, help='底盘服务ZeroMQ地址')
    parser.add_argument('--arm-addr', default=None, help='机械臂服务ZeroMQ地址')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HomeBot 运动控制服务启动器")
    print("=" * 60)
    
    # 获取配置
    config = get_config()
    
    # 覆盖串口配置
    if args.port:
        config.chassis.serial_port = args.port
        config.arm.serial_port = args.port  # 确保机械臂使用同一串口
        print(f"[配置] 使用指定串口: {args.port}")
    else:
        # 检查 auto 是否解析成功
        if config.chassis.serial_port == "auto":
            print("[错误] 未检测到可用串口设备。")
            print("       请连接硬件后重试，或手动指定串口:")
            print("         python -m services.motion_service --port /dev/ttyUSB0")
            print("       查看可用设备:")
            print("         python tools/list_devices.py")
            sys.exit(1)
        print(f"[配置] 底盘串口: {config.chassis.serial_port}")
        print(f"[配置] 机械臂串口: {config.arm.serial_port}")
    
    # 初始化共享串口总线
    bus_manager = ServoBusManager()
    if not bus_manager.initialize(config.chassis.serial_port, config.chassis.baudrate):
        print("[错误] 串口总线初始化失败，退出")
        sys.exit(1)
    
    print("[系统] 共享串口总线已初始化")
    
    # 存储服务和线程
    services = {}
    threads = []
    
    # 启动底盘服务
    if args.service in ('chassis', 'both'):
        from services.motion_service.chassis_service import ChassisService
        
        chassis_addr = args.chassis_addr or config.chassis.service_addr
        chassis_service = ChassisService(rep_addr=chassis_addr, use_shared_bus=True)
        services['chassis'] = chassis_service
        
        # 使用非守护线程，避免解释器关闭时的问题
        chassis_thread = threading.Thread(target=chassis_service.start, daemon=False)
        threads.append(('chassis', chassis_thread))
        chassis_thread.start()
        
        print(f"[服务] 底盘服务已启动 ({chassis_addr})")
    
    # 启动机械臂服务
    if args.service in ('arm', 'both'):
        from services.motion_service.arm_service import ArmService
        
        arm_addr = args.arm_addr or config.zmq.arm_service_addr
        arm_service = ArmService(rep_addr=arm_addr)
        services['arm'] = arm_service
        
        # 使用非守护线程，避免解释器关闭时的问题
        arm_thread = threading.Thread(target=arm_service.start, daemon=False)
        threads.append(('arm', arm_thread))
        arm_thread.start()
        
        print(f"[服务] 机械臂服务已启动 ({arm_addr})")
    
    print("=" * 60)
    print("按 Ctrl+C 停止所有服务")
    print("=" * 60)
    
    # 等待所有线程
    restart_failures = {}
    next_restart_time = {}

    try:
        while True:
            time.sleep(0.1)
            now = time.time()

            # 检查是否有线程异常退出
            dead_threads = []
            for name, thread in threads:
                if not thread.is_alive():
                    dead_threads.append(name)

            for name in dead_threads:
                # 指数退避：连续失败时增加等待时间
                if name in next_restart_time and now < next_restart_time[name]:
                    continue

                print(f"[警告] {name} 服务线程已退出，尝试重启...")

                # 停止旧实例
                if name in services:
                    try:
                        services[name].stop()
                    except Exception as e:
                        print(f"[警告] 停止旧 {name} 服务时出错: {e}")
                    del services[name]

                # 确保共享总线仍然连接（可能被异常关闭，或驱动 close 时断开）
                if not bus_manager.is_initialized():
                    print("[系统] 共享串口总线已断开，尝试重新初始化...")
                    if not bus_manager.initialize(config.chassis.serial_port, config.chassis.baudrate):
                        print("[错误] 共享串口总线重新初始化失败，跳过本次重启")
                        failures = restart_failures.get(name, 0) + 1
                        restart_failures[name] = failures
                        delay = min(30, 2 ** failures)
                        next_restart_time[name] = now + delay
                        continue

                # 重新创建并启动服务
                try:
                    if name == "chassis":
                        from services.motion_service.chassis_service import ChassisService
                        new_service = ChassisService(rep_addr=chassis_addr, use_shared_bus=True)
                        services["chassis"] = new_service
                        new_thread = threading.Thread(target=new_service.start, daemon=False)
                        new_thread.start()
                        print(f"[服务] 底盘服务已重启 ({chassis_addr})")
                        restart_failures[name] = 0
                        threads = [(n, t) for n, t in threads if n != name]
                        threads.append(("chassis", new_thread))
                    elif name == "arm":
                        from services.motion_service.arm_service import ArmService
                        new_service = ArmService(rep_addr=arm_addr)
                        services["arm"] = new_service
                        new_thread = threading.Thread(target=new_service.start, daemon=False)
                        new_thread.start()
                        print(f"[服务] 机械臂服务已重启 ({arm_addr})")
                        restart_failures[name] = 0
                        threads = [(n, t) for n, t in threads if n != name]
                        threads.append(("arm", new_thread))
                except Exception as e:
                    print(f"[错误] 重启 {name} 服务失败: {e}")
                    failures = restart_failures.get(name, 0) + 1
                    restart_failures[name] = failures
                    delay = min(30, 2 ** failures)
                    next_restart_time[name] = now + delay
                    print(f"[系统] {name} 服务将在 {delay} 秒后再次重试")
                    # 如果只启动了一个服务，退出主程序
                    if len(threads) == 1:
                        print("[系统] 服务已停止")
                        return

            if not threads:
                print("[系统] 所有服务已停止")
                return
    except KeyboardInterrupt:
        print("\n[系统] 收到停止信号，正在关闭服务...")
    finally:
        # 停止所有服务
        for name, service in services.items():
            print(f"[服务] 正在停止 {name}...")
            try:
                service.stop()
            except Exception as e:
                print(f"[警告] 停止 {name} 时出错: {e}")

        # 等待所有线程结束（给线程一些时间来清理）
        print("[系统] 等待服务线程结束...")
        for name, thread in threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    print(f"[警告] {name} 线程未能在 2 秒内结束")

        # 关闭共享总线
        print("[系统] 正在关闭串口总线...")
        bus_manager.close()

        # 短暂延迟确保所有输出完成
        time.sleep(0.1)
        print("[系统] 所有服务已停止")


if __name__ == '__main__':
    main()
