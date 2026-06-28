"""递送任务阶段编排器。"""
import threading
import time
from typing import Optional

from common.logging import get_logger
from .state import SceneStateManager, DeliveryPhase
from .config import DeliveryAgentConfig

logger = get_logger(__name__)


class DeliveryOrchestrator:
    """递送任务编排器。

    负责在预检查通过后，按阶段推进递送任务。
    """

    def __init__(self,
                 state_manager: SceneStateManager,
                 search_skill,
                 approach_skill,
                 grasp_skill,
                 place_skill,
                 config: DeliveryAgentConfig):
        self.state = state_manager
        self.search_skill = search_skill
        self.approach_skill = approach_skill
        self.grasp_skill = grasp_skill
        self.place_skill = place_skill
        self.config = config

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> dict:
        """启动编排器（后台线程）。"""
        if self._thread is not None and self._thread.is_alive():
            return {"status": "running", "message": "编排器已在运行"}

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return {"status": "started", "message": "递送任务已启动"}

    def stop(self):
        """停止编排器。"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _run(self):
        """主执行循环。"""
        try:
            self._execute_phase_approach_target()
            if self._should_stop():
                return

            self._execute_phase_grasp()
            if self._should_stop():
                return

            self._execute_phase_find_destination()
            if self._should_stop():
                return

            self._execute_phase_approach_destination()
            if self._should_stop():
                return

            self._execute_phase_place()
        except Exception as e:
            logger.exception("编排器异常")
            self.state.transition(DeliveryPhase.FAILED, error_message=str(e))

    def _should_stop(self) -> bool:
        return self._stop_event.is_set() or self.state.get().phase == DeliveryPhase.FAILED

    def _execute_phase_approach_target(self):
        """接近抓取目标。"""
        self.state.transition(DeliveryPhase.APPROACH_TARGET)
        info = self.state.get().grab_target_info
        bbox = info.get("bbox")
        if not bbox:
            self.state.transition(DeliveryPhase.FAILED, error_message="缺少抓取目标 bbox")
            return

        result = self.approach_skill.approach(tuple(bbox))
        if result.get("success"):
            self.state.update(target_last_bbox=bbox)
        else:
            # 重试一次：旋转搜索
            logger.info("接近目标失败，尝试旋转搜索")
            self.approach_skill.rotate_search(30)
            time.sleep(0.5)
            result = self.approach_skill.approach(tuple(bbox))
            if not result.get("success"):
                self.state.transition(DeliveryPhase.FAILED, error_message=result.get("message", "接近目标失败"))

    def _execute_phase_grasp(self):
        """抓取目标。"""
        self.state.transition(DeliveryPhase.GRASP)
        target = self.state.get().grab_target
        result = self.grasp_skill.execute(target)
        if result.get("success"):
            self.state.update(held_object=target)
        else:
            self.state.transition(DeliveryPhase.FAILED, error_message=result.get("message", "抓取失败"))

    def _execute_phase_find_destination(self):
        """重新定位递送目标。"""
        self.state.transition(DeliveryPhase.FIND_DESTINATION)
        target = self.state.get().deliver_target
        info = self.state.get().deliver_target_info
        # 如果预检查时已找到且在视野中，直接复用
        if info.get("nearby") and info.get("bbox"):
            self.state.update(destination_last_bbox=info.get("bbox"))
            return

        # 否则旋转搜索
        for attempt in range(3):
            result = self.search_skill.search(target)
            if result.get("found"):
                self.state.update(
                    destination_last_bbox=result.get("bbox"),
                    deliver_target_info={**info, **result, "nearby": True},
                )
                return
            self.approach_skill.rotate_search(45)
            time.sleep(0.5)

        self.state.transition(DeliveryPhase.FAILED, error_message="无法重新定位递送目标")

    def _execute_phase_approach_destination(self):
        """接近递送目标。"""
        self.state.transition(DeliveryPhase.APPROACH_DESTINATION)
        info = self.state.get().deliver_target_info
        bbox = info.get("bbox") or self.state.get().destination_last_bbox
        if not bbox:
            self.state.transition(DeliveryPhase.FAILED, error_message="缺少递送目标 bbox")
            return

        result = self.approach_skill.approach(tuple(bbox))
        if result.get("success"):
            self.state.update(destination_last_bbox=bbox)
        else:
            self.state.transition(DeliveryPhase.FAILED, error_message=result.get("message", "接近递送目标失败"))

    def _execute_phase_place(self):
        """放置物体。"""
        self.state.transition(DeliveryPhase.PLACE)
        bbox = self.state.get().destination_last_bbox
        result = self.place_skill.execute(destination_bbox=bbox, confirm_hand=False)
        if result.get("status") == "success" or result.get("success"):
            self.place_skill.reset_arm()
            self.state.transition(DeliveryPhase.COMPLETED, error_message=None)
        else:
            self.state.transition(DeliveryPhase.FAILED, error_message=result.get("message", "放置失败"))
