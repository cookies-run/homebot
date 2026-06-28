"""跟踪器迟滞测试。"""
import os
import sys

import pytest

# 外部可复用技能路径
_delivery_skills_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../skills/homebot-skill/scripts/delivery")
)
if _delivery_skills_path not in sys.path:
    sys.path.insert(0, _delivery_skills_path)

from tracker_skill import TargetTracker, Detection


def make_detections(cx, cy, w, h, count=1):
    """辅助函数：生成 detection 列表。"""
    detections = []
    for _ in range(count):
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=0.9))
    return detections


def test_primary_target_stays_stable_with_small_jitter():
    """目标小幅抖动时不应切换主要目标。"""
    tracker = TargetTracker(selection_strategy="center", switch_margin=0.20, switch_min_frames=3)

    # 初始目标 A 在中心
    tracker.update(make_detections(0.5, 0.5, 0.1, 0.1))
    primary_a = tracker.primary_target
    assert primary_a is not None

    # 目标 B 稍微偏离中心但差距不大，连续 5 帧出现
    for _ in range(5):
        tracker.update(make_detections(0.5, 0.5, 0.1, 0.1) + make_detections(0.52, 0.5, 0.1, 0.1))

    assert tracker.primary_target.id == primary_a.id


def test_primary_target_switches_after_persistent_better_candidate():
    """新目标显著更优并持续多帧后才切换。"""
    tracker = TargetTracker(selection_strategy="center", switch_margin=0.20, switch_min_frames=3)

    # 目标 A 在中心偏右
    tracker.update(make_detections(0.6, 0.5, 0.1, 0.1))
    primary_a = tracker.primary_target

    # 目标 B 明显更靠近中心，连续 3 帧
    for _ in range(3):
        tracker.update(make_detections(0.6, 0.5, 0.1, 0.1) + make_detections(0.5, 0.5, 0.1, 0.1))

    assert tracker.primary_target.id != primary_a.id
    # 新目标应更靠近中心
    cx, cy = tracker.primary_target.center
    assert abs(cx - 0.5) < 0.05


def test_switch_requires_min_frames():
    """只出现 1 帧的更优目标不切换。"""
    tracker = TargetTracker(selection_strategy="center", switch_margin=0.20, switch_min_frames=3)

    tracker.update(make_detections(0.6, 0.5, 0.1, 0.1))
    primary_a = tracker.primary_target

    # 只出现 1 帧更优目标
    tracker.update(make_detections(0.6, 0.5, 0.1, 0.1) + make_detections(0.5, 0.5, 0.1, 0.1))
    # 接下来 2 帧回到只有 A
    tracker.update(make_detections(0.6, 0.5, 0.1, 0.1))
    tracker.update(make_detections(0.6, 0.5, 0.1, 0.1))

    assert tracker.primary_target.id == primary_a.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
