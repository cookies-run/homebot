#!/usr/bin/env python3
"""递送智能体独立入口。

用法：
    python start_delivery_agent.py --request "把矿泉水递给穿红衣服的人"
    python start_delivery_agent.py --interactive
"""
import argparse
import os
import sys
import time

# 添加 src 到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_current_dir, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from applications.delivery_agent import DeliveryAgent, DeliveryAgentConfig
from common.logging import get_logger

logger = get_logger(__name__)


def run_once(agent: DeliveryAgent, request: str):
    """单次运行：解析 → 确认（自动模拟） → 预检查 → 执行。"""
    print(f"\n[用户] {request}")
    result = agent.process_user_request(request)
    print(f"[Agent] {result.get('message', result)}")

    if result.get("action") == "parse":
        # 自动确认（非交互模式）
        print("[系统] 自动确认...")
        result = agent.process_confirmation(True)
        print(f"[Agent] {result.get('message', result)}")

    if result.get("action") in ("precheck_need_search", "precheck_not_graspable", "precheck_destination_not_nearby"):
        print(f"[Agent] {result.get('message')}")
        print("[系统] 由于是非交互模式，停止等待用户决策。")
        return

    # 等待执行完成或失败
    print("[系统] 等待任务执行...")
    for _ in range(300):  # 最多等 300 秒
        status = agent.get_status()
        phase = status.get("phase")
        print(f"[状态] phase={phase}, held={status.get('held_object')}, error={status.get('error_message')}")
        if phase in ("completed", "failed"):
            break
        time.sleep(1)

    print(f"\n[最终结果] {agent.get_status()}")


def run_interactive(agent: DeliveryAgent):
    """交互模式。"""
    print("\n=== HomeBot 递送智能体 ===")
    print("输入递送请求，例如：把矿泉水递给穿红衣服的人")
    print("输入 'status' 查看状态，输入 'quit' 退出。\n")

    pending_confirmation = False

    while True:
        try:
            user_input = input("[用户] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "status":
            print(agent.get_status())
            continue

        if pending_confirmation:
            confirmed = user_input.lower() in ("对", "是的", "好", "确认", "yes", "y", "ok")
            result = agent.process_confirmation(confirmed)
            pending_confirmation = False
        else:
            result = agent.process_user_request(user_input)
            if result.get("action") == "parse":
                pending_confirmation = True

        print(f"[Agent] {result.get('message', result)}")

        # 处理需要用户决策的预检查分支
        if result.get("action") in ("precheck_need_search", "precheck_not_graspable", "precheck_destination_not_nearby"):
            print("[Agent] 请回复：尝试 / 取消 / 搜索")

    print("\n[系统] 退出递送智能体。")


def main():
    parser = argparse.ArgumentParser(description="HomeBot 递送智能体")
    parser.add_argument("--request", "-r", type=str, help="单次运行请求")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--handbook", type=str, default=None, help="手册路径")
    args = parser.parse_args()

    config = DeliveryAgentConfig.from_env()
    if args.handbook:
        config.handbook_path = args.handbook

    print("[系统] 初始化递送智能体...")
    agent = DeliveryAgent(config)
    print(f"[系统] 当前状态: {agent.get_status()}")

    try:
        if args.request:
            run_once(agent, args.request)
        else:
            run_interactive(agent)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
