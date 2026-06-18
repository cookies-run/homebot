#!/usr/bin/env python3
"""
纯视觉 Tracker 测试（不驱动底盘/机械臂）

流程：
    1. 从机身摄像头取一帧，调用 VLM 获取目标 bbox；
    2. 用该 bbox 初始化 OpenCV Tracker（CSRT/KCF/MOSSE fallback）；
    3. 连续取 N 帧并运行 tracker.update()，输出 center_x_ratio / area_ratio / 耗时；
    4. 统计平均耗时与丢失次数。

用法：
    source /Users/yi/works/智绘屿/yu-freetime/robots/HomeBot/venv/bin/activate
    python test_tracker_only.py --target "一包纸巾" --frames 30
"""

import sys
import os
import time
import re
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import cv2

from video_subscriber import VideoSubscriber
from grab_optimized import analyze_images_with_fallback

ROBOT_IP = "localhost"
BODY_CAM_PORT = 5560


def detect_bbox_with_vlm(image_path: str, target_object: str, img_w: int, img_h: int):
    """从图片中调用 VLM 定位目标，返回归一化 bbox [x1,y1,x2,y2]（0~1）"""
    prompt = f"""你是机器人的高精度视觉定位助手。请在图片中找到目标物体"{target_object}"的精确包围框。

请返回一个 JSON 数组：[x_min, y_min, x_max, y_max]，其中每个值是 0 到 1000 之间的整数或浮点数，表示相对于图片宽度和高度的归一化坐标（0 表示最左/最上，1000 表示最右/最下）。

只输出这个数组，不要任何解释、Markdown 标记或额外文字。如果图片中确实找不到该目标，请只回复：未找到"""

    t0 = time.time()
    result, used_provider = analyze_images_with_fallback(
        image_paths=[image_path],
        prompt=prompt,
        max_tokens=128,
    )
    dt = time.time() - t0
    result = result.strip()
    print(f"[TEST] VLM({used_provider}) 定位耗时: {dt*1000:.1f}ms, 结果: {result}")

    if "未找到" in result or "not found" in result.lower():
        return None
    m = re.search(r"\[([\d\s.,]+)\]", result)
    if not m:
        print("[TEST] 无法解析 bbox")
        return None
    nums = [float(v.strip()) for v in m.group(1).split(",") if v.strip()]
    if len(nums) != 4:
        print(f"[TEST] bbox 坐标数量不对: {nums}")
        return None
    x1, y1, x2, y2 = nums

    # 坐标格式兼容：0~1000 归一化 / 0~1 归一化 / 真实像素
    max_val = max(nums)
    if max_val > 1.0:
        if max_val <= 1000.0:
            x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
            print(f"[TEST] VLM 返回 0-1000 归一化坐标，已转换: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")
        else:
            x1, y1, x2, y2 = x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h
            print(f"[TEST] VLM 返回像素坐标，已归一化: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")

    x1, x2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
    y1, y2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        print("[TEST] VLM 返回框过小")
        return None
    return x1, y1, x2, y2


def create_tracker():
    for name in ["TrackerCSRT_create", "TrackerKCF_create"]:
        try:
            ctor = getattr(cv2, name)
            tracker = ctor()
            print(f"[TEST] 使用追踪器: {name}")
            return tracker
        except Exception as e:
            print(f"[TEST] {name} 不可用: {e}")
    raise RuntimeError("当前 OpenCV 没有可用的 Tracker")


def main():
    parser = argparse.ArgumentParser(description="纯视觉 Tracker 测试")
    parser.add_argument("--target", default="一包纸巾", help="目标物体")
    parser.add_argument("--frames", type=int, default=30, help="追踪更新帧数")
    parser.add_argument("--out-dir", default="tracker_test_captures", help="测试图片保存目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TEST] 连接机身摄像头 {ROBOT_IP}:{BODY_CAM_PORT}")
    sub = VideoSubscriber(ROBOT_IP, BODY_CAM_PORT)

    # ---- 第 1 帧：初始化 ----
    print("[TEST] 等待第 1 帧用于 VLM 初始化...")
    frame_bytes = sub.wait_for_frame(timeout_seconds=5.0)
    if frame_bytes is None:
        print("[TEST] 取图失败，退出")
        sub.close()
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_path = out_dir / f"init_{ts}.jpg"
    init_path.write_bytes(frame_bytes)
    print(f"[TEST] 初始化图片已保存: {init_path}")

    img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("[TEST] 图片解码失败")
        sub.close()
        return 1
    h, w = img.shape[:2]
    print(f"[TEST] 图像尺寸: {w}x{h}")

    bbox_norm = detect_bbox_with_vlm(str(init_path), args.target, w, h)
    if bbox_norm is None:
        print("[TEST] VLM 未定位到目标，无法初始化 Tracker")
        sub.close()
        return 1

    x1, y1, x2, y2 = bbox_norm
    bbox = (int(x1 * w), int(y1 * h), int((x2 - x1) * w), int((y2 - y1) * h))
    print(f"[TEST] 初始框（像素）: {bbox}")

    tracker = create_tracker()
    try:
        tracker.init(img, bbox)
        print("[TEST] Tracker 初始化成功，开始连续追踪...\n")
    except Exception as e:
        print(f"[TEST] Tracker 初始化失败: {e}")
        sub.close()
        return 1

    # ---- 连续追踪 ----
    times = []
    lost = 0
    for i in range(1, args.frames + 1):
        t0 = time.time()
        frame_bytes = sub.wait_for_frame(timeout_seconds=2.0)
        if frame_bytes is None:
            print(f"[TEST] 第 {i} 帧取图失败，中断")
            break
        img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        ok, bbox = tracker.update(img)
        dt = time.time() - t0
        times.append(dt)

        if ok:
            x, y, bw, bh = bbox
            cx_ratio = (x + bw / 2.0) / w
            area_ratio = (bw * bh) / (w * h)
            print(
                f"[TEST] frame {i:02d}: ok | "
                f"center_x={cx_ratio:.3f} | area={area_ratio:.3f} | "
                f"cost={dt*1000:.1f}ms"
            )
        else:
            lost += 1
            print(f"[TEST] frame {i:02d}: LOST | cost={dt*1000:.1f}ms")

    sub.close()

    if times:
        avg = sum(times) / len(times) * 1000
        print(f"\n[TEST] 平均取图+追踪耗时: {avg:.1f}ms")
    print(f"[TEST] 总帧数: {len(times)}, 丢失次数: {lost}")
    return 0 if lost == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
