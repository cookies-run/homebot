#!/usr/bin/env python3
"""
直接调用 SearchSkill 测试寻物能力（机器人端执行）

用法：
    source /Users/yi/works/智绘屿/yu-freetime/robots/HomeBot/venv/bin/activate
    cd /Users/yi/works/智绘屿/yu-freetime/robots/HomeBot/skills/homebot-skill/scripts
    python test_search_skill.py --target "一包纸巾"

参数：
    --target      要搜索的目标描述（默认：一包纸巾）
    --scan        是否执行旋转扫描（需要底盘可用）
    --align       找到目标后是否精对准机身正面（需要底盘可用，可与 --scan 同时用）
    --provider    VLM provider，默认 minimax
    --ip          机器人 IP，默认 localhost
    --port        机身摄像头视频端口，默认 5560

日志输出到 stderr；模型原始返回、画面描述、最终 bbox 都会打印。
"""
import argparse
import os
import sys
import time
from pathlib import Path

# 把 skills 目录加入路径，确保能导入 delivery 下的模块
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# 把 software/src 加入路径，确保能导入 common/configs
_SRC_DIR = _SCRIPT_DIR / "../../../software/src"
if _SRC_DIR.exists():
    sys.path.insert(0, str(_SRC_DIR.resolve()))

from delivery.adapters import VisionAdapter
from delivery.search_skill import SearchSkill


def main():
    parser = argparse.ArgumentParser(description="测试 SearchSkill 寻物能力")
    parser.add_argument("--target", default="一包纸巾", help="要搜索的目标描述")
    parser.add_argument("--scan", action="store_true", help="是否执行旋转扫描（需要底盘）")
    parser.add_argument("--align", action="store_true", help="扫描找到目标后是否精对准（需要底盘）")
    parser.add_argument("--provider", default="minimax", help="VLM provider")
    parser.add_argument("--ip", default="localhost", help="机器人 IP")
    parser.add_argument("--port", type=int, default=5560, help="机身摄像头视频端口")
    args = parser.parse_args()

    video_addr = f"tcp://{args.ip}:{args.port}"
    print(f"[TEST] 连接机身摄像头: {video_addr}", file=sys.stderr)
    vision_adapter = VisionAdapter(video_addr)
    vision_adapter.start()

    skill = SearchSkill(vision_adapter=vision_adapter, provider=args.provider)

    kwargs = {}
    if args.scan or args.align:
        try:
            from chassis_control import HomeBotChassisController
            from robot_config import CHASSIS_PORT
            chassis = HomeBotChassisController(f"tcp://{args.ip}:{CHASSIS_PORT}")
            if args.scan:
                kwargs["rotate_fn"] = lambda: chassis.left_deg(120)
                kwargs["rotate_deg"] = 120
                kwargs["max_rotations"] = 3
            if args.align:
                kwargs["align_fn"] = lambda deg: chassis.left_deg(deg) if deg > 0 else chassis.right_deg(-deg)
                kwargs["camera_hfov_deg"] = 60.0
                kwargs["align_threshold"] = 0.1
                kwargs["max_align_attempts"] = 3
        except Exception as e:
            print(f"[TEST] 底盘初始化失败: {e}", file=sys.stderr)

    mode = []
    if "rotate_fn" in kwargs:
        mode.append("旋转扫描")
    if "align_fn" in kwargs:
        mode.append("精对准")
    if not mode:
        mode.append("单帧")
    print(f"[TEST] 执行{'+'.join(mode)}寻物", file=sys.stderr)

    if "rotate_fn" in kwargs:
        result = skill.search_with_scan(args.target, **kwargs)
    else:
        result = skill.search(args.target, max_retries=0)

    print("\n========== 最终结果 ==========", file=sys.stderr)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)

    # 保存本次分析的图片，方便复盘
    print(f"\n[TEST] 分析图片保存目录: {skill._temp_dir}", file=sys.stderr)

    vision_adapter.stop()


if __name__ == "__main__":
    main()
