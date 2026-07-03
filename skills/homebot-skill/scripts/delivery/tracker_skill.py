"""轻量 IoU 跟踪器（递送智能体专用）。

借鉴 applications/human_follow/tracker.py 的思路，但独立实现，避免修改原文件。
新增主要目标切换迟滞，减少多人/多目标场景下的帧间抖动。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class TargetStatus(Enum):
    TRACKING = "tracking"
    LOST = "lost"


@dataclass
class Detection:
    """检测结果。"""

    bbox: Tuple[float, float, float, float]  # 归一化 xyxy
    confidence: float
    class_name: str = "object"

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Target:
    """跟踪目标。"""

    id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    status: TargetStatus = TargetStatus.TRACKING
    age: int = 0
    time_since_update: int = 0
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def update(self, detection: Detection):
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.age += 1
        self.time_since_update = 0
        self.status = TargetStatus.TRACKING
        self.history.append(self.bbox)
        if len(self.history) > 10:
            self.history.pop(0)

    def mark_missed(self):
        self.time_since_update += 1
        if self.time_since_update > 30:
            self.status = TargetStatus.LOST

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def predict(self) -> Tuple[float, float, float, float]:
        if len(self.history) < 2:
            return self.bbox
        vx = sum((self.history[i][0] - self.history[i - 1][0]) +
                 (self.history[i][2] - self.history[i - 1][2])
                 for i in range(1, len(self.history))) / ((len(self.history) - 1) * 2)
        vy = sum((self.history[i][1] - self.history[i - 1][1]) +
                 (self.history[i][3] - self.history[i - 1][3])
                 for i in range(1, len(self.history))) / ((len(self.history) - 1) * 2)
        x1, y1, x2, y2 = self.bbox
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)


def compute_iou(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float]) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


class TargetTracker:
    """目标跟踪器，带主要目标切换迟滞。"""

    _id_counter = 0

    def __init__(self,
                 max_age: int = 30,
                 min_iou: float = 0.3,
                 selection_strategy: str = "center",
                 switch_margin: float = 0.20,
                 switch_min_frames: int = 3):
        self.max_age = max_age
        self.min_iou = min_iou
        self.selection_strategy = selection_strategy
        self.switch_margin = switch_margin
        self.switch_min_frames = switch_min_frames

        self.targets: List[Target] = []
        self.primary_target: Optional[Target] = None
        self._candidate_counter = 0
        self._frame_center = (0.5, 0.5)

    def reset(self):
        self.targets = []
        self.primary_target = None
        TargetTracker._id_counter = 0
        self._candidate_counter = 0

    def get_primary_target(self) -> Optional[Target]:
        return self.primary_target

    def _create_target(self, detection: Detection) -> Target:
        TargetTracker._id_counter += 1
        target = Target(
            id=TargetTracker._id_counter,
            bbox=detection.bbox,
            confidence=detection.confidence,
        )
        target.history.append(detection.bbox)
        return target

    def update(self, detections: List[Detection]) -> Optional[Target]:
        if not self.targets or not detections:
            matched, unmatched_targets, unmatched_detections = [], self.targets.copy(), detections.copy()
        else:
            matched, unmatched_targets, unmatched_detections = self._match_detections(detections)

        for target, detection in matched:
            target.update(detection)

        for target in unmatched_targets:
            target.mark_missed()

        for detection in unmatched_detections:
            new_target = self._create_target(detection)
            self.targets.append(new_target)

        self.targets = [t for t in self.targets if t.time_since_update < self.max_age]

        self._select_primary_target()
        return self.primary_target

    def _match_detections(self, detections: List[Detection]):
        iou_matrix = np.zeros((len(self.targets), len(detections)))
        for i, target in enumerate(self.targets):
            predicted_bbox = target.predict() if target.time_since_update > 0 and len(target.history) >= 2 else target.bbox
            for j, det in enumerate(detections):
                iou_matrix[i, j] = compute_iou(predicted_bbox, det.bbox)

        matched = []
        used_detections = set()
        while True:
            max_iou = self.min_iou
            max_i = -1
            max_j = -1
            for i in range(len(self.targets)):
                if self.targets[i].status == TargetStatus.LOST:
                    continue
                for j in range(len(detections)):
                    if j in used_detections:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_i = i
                        max_j = j
            if max_i == -1:
                break
            matched.append((self.targets[max_i], detections[max_j]))
            used_detections.add(max_j)

        unmatched_targets = [t for t in self.targets if not any(m[0] == t for m in matched)]
        unmatched_detections = [d for j, d in enumerate(detections) if j not in used_detections]
        return matched, unmatched_targets, unmatched_detections

    def _select_primary_target(self):
        if not self.targets:
            self.primary_target = None
            return

        valid_targets = [t for t in self.targets if t.time_since_update == 0]
        if not valid_targets:
            self.primary_target = None
            return

        if self.selection_strategy == "center":
            def score(t):
                cx, cy = t.center
                fx, fy = self._frame_center
                return -((cx - fx) ** 2 + (cy - fy) ** 2)
            best = max(valid_targets, key=score)
        elif self.selection_strategy == "largest":
            best = max(valid_targets, key=lambda t: t.area)
        else:
            best = max(valid_targets, key=lambda t: t.confidence)

        if self.primary_target is None or self.primary_target not in self.targets:
            self.primary_target = best
            self._candidate_counter = 0
            return

        if best.id == self.primary_target.id:
            self._candidate_counter = 0
            return

        current_score = self._score(self.primary_target)
        best_score = self._score(best)

        # 新目标必须显著优于当前目标
        if self.selection_strategy == "center":
            improved = best_score > current_score * (1 + self.switch_margin)
        else:
            improved = best_score > current_score * (1 + self.switch_margin)

        if improved:
            self._candidate_counter += 1
            if self._candidate_counter >= self.switch_min_frames:
                self.primary_target = best
                self._candidate_counter = 0
        else:
            self._candidate_counter = 0

    def _score(self, target: Target) -> float:
        if self.selection_strategy == "center":
            cx, cy = target.center
            fx, fy = self._frame_center
            # 距离越近分数越高（取负的距离平方，再取反转为正）
            return -((cx - fx) ** 2 + (cy - fy) ** 2)
        elif self.selection_strategy == "largest":
            return target.area
        else:
            return target.confidence
