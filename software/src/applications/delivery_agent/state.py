"""场景状态管理。"""
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class DeliveryPhase(str, Enum):
    """递送任务阶段。"""

    IDLE = "idle"
    PLANNING = "planning"
    CONFIRMING = "confirming"
    PRE_CHECKING = "pre_checking"
    APPROACH_TARGET = "approach_target"
    GRASP = "grasp"
    FIND_DESTINATION = "find_destination"
    APPROACH_DESTINATION = "approach_destination"
    PLACE = "place"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SceneState:
    """递送任务场景状态。"""

    phase: DeliveryPhase = DeliveryPhase.IDLE
    user_request: Optional[str] = None
    grab_target: Optional[str] = None
    deliver_target: Optional[str] = None
    confirmed: bool = False
    precheck_passed: bool = False

    # 预检查信息
    grab_target_info: dict = field(default_factory=dict)
    deliver_target_info: dict = field(default_factory=dict)

    # 执行期状态
    held_object: Optional[str] = None
    target_last_bbox: Optional[Tuple[float, float, float, float]] = None
    destination_last_bbox: Optional[Tuple[float, float, float, float]] = None

    retry_count: int = 0
    error_message: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """序列化为字典。"""
        data = asdict(self)
        data["phase"] = self.phase.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SceneState":
        """从字典恢复。"""
        data = dict(data)
        phase_value = data.pop("phase", DeliveryPhase.IDLE.value)
        data["phase"] = DeliveryPhase(phase_value)
        # 过滤掉 dataclass 中没有的字段，避免旧持久化数据导致错误
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class SceneStateManager:
    """场景状态管理器（线程安全单例）。"""

    _instance: Optional["SceneStateManager"] = None
    _lock = threading.Lock()

    def __new__(cls, persist_path: str = "/tmp/homebot_scene_state.json"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init(persist_path)
        return cls._instance

    def _init(self, persist_path: str):
        self._state = SceneState()
        self._persist_path = persist_path
        self._mutex = threading.Lock()
        self._load()

    def reset(self):
        """重置状态。"""
        with self._mutex:
            self._state = SceneState()
            self._persist()

    def get(self) -> SceneState:
        """获取当前状态副本。"""
        with self._mutex:
            return SceneState.from_dict(self._state.to_dict())

    def update(self, **kwargs):
        """更新状态字段并持久化。"""
        with self._mutex:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            self._state.updated_at = datetime.now().isoformat()
            self._persist()

    def transition(self, phase: DeliveryPhase, **kwargs):
        """切换阶段并更新字段。"""
        kwargs["phase"] = phase
        self.update(**kwargs)

    def _persist(self):
        """持久化到 JSON 文件。"""
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._state.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 持久化失败不应影响主流程
            print(f"[SceneStateManager] 持久化失败: {e}")

    def _load(self):
        """从 JSON 文件恢复。"""
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._state = SceneState.from_dict(data)
        except Exception as e:
            print(f"[SceneStateManager] 加载失败: {e}")
            self._state = SceneState()
