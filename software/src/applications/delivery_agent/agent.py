"""递送智能体运行时。

加载 agents/delivery_manager.md 作为大模型工作手册，维护多轮对话上下文，
根据手册输出调用Planner/Orchestrator/Skill完成递送任务。
"""
import json
import os
import sys
import threading
from typing import Optional

from common.logging import get_logger
from configs.config import get_config
from configs.ai_config import get_ai_credentials

from .config import DeliveryAgentConfig
from .state import SceneStateManager, DeliveryPhase
from .adapter import ChassisAdapter, ArmAdapter, VisionAdapter

# 导入外部可复用技能（skills/homebot-skill/scripts/delivery/）
_delivery_skills_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../skills/homebot-skill/scripts/delivery")
)
if _delivery_skills_path not in sys.path:
    sys.path.insert(0, _delivery_skills_path)

from search_skill import SearchSkill
from approach_skill import ApproachSkill
from grasp_skill import GraspSkill
from place_skill import PlaceSkill

from .orchestrator import DeliveryOrchestrator

logger = get_logger(__name__)


class DeliveryAgent:
    """递送管理智能体。"""

    def __init__(self, config: Optional[DeliveryAgentConfig] = None):
        self.config = config or DeliveryAgentConfig.from_env()
        self.state = SceneStateManager(self.config.scene_state_persist_path)
        self._handbook = self._load_handbook()
        self._llm_client = None
        self._llm_model = None
        self._init_llm()

        # 适配器与技能
        self.vision_adapter = VisionAdapter(self.config.vision_sub_addr)
        self.chassis_adapter = ChassisAdapter(self.config.chassis_service_addr)
        self.arm_adapter = ArmAdapter(self.config.arm_service_addr)

        self.search_skill = SearchSkill(self.vision_adapter)
        self.approach_skill = ApproachSkill(
            self.chassis_adapter,
            self.vision_adapter,
            approach_distance_cm=self.config.approach_distance_cm,
        )
        self.grasp_skill = GraspSkill()
        self.place_skill = PlaceSkill(self.arm_adapter, self.vision_adapter)

        self._orchestrator: Optional[DeliveryOrchestrator] = None

    def _load_handbook(self) -> str:
        """加载智能体手册。"""
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../..")
        )
        path = self.config.resolve_handbook_path(project_root)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"加载手册失败: {e}")
            return ""

    def _init_llm(self):
        """初始化 LLM 客户端。"""
        try:
            cfg = get_config().llm
            secrets = get_ai_credentials()
            api_key = cfg.api_key or secrets.llm.api_key
            model = cfg.model or secrets.llm.model

            if not api_key or not model:
                logger.warning("LLM 配置不完整，语义解析将不可用")
                return

            from services.llm_service.llm_client import get_llm_client
            self._llm_client = get_llm_client()
            self._llm_model = model
        except Exception as e:
            logger.error(f"初始化 LLM 客户端失败: {e}")

    def _call_llm(self, messages: list, max_tokens: int = 256) -> Optional[str]:
        """调用 LLM。"""
        if self._llm_client is None or self._llm_model is None:
            logger.error("LLM 客户端未初始化")
            return None
        try:
            response = self._llm_client.chat_completion(
                model=self._llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None

    def _build_messages(self, user_text: str, include_history: bool = True) -> list:
        messages = [{"role": "system", "content": self._handbook}]
        if include_history:
            # 仅保留最近 10 轮，避免过长
            state_dict = self.state.get().to_dict()
            history = state_dict.get("history", [])
            if not isinstance(history, list):
                history = []
            for msg in history[-10:]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _append_history(self, role: str, content: str):
        """追加对话历史到场景状态。"""
        state_dict = self.state.get().to_dict()
        history = state_dict.get("history", [])
        if not isinstance(history, list):
            history = []
        history.append({"role": role, "content": content})
        # 截断到最近 20 轮
        self.state.update(history=history[-20:])

    def process_user_request(self, request: str) -> dict:
        """处理用户请求（解析 + 确认）。"""
        self.state.reset()
        self.state.transition(
            DeliveryPhase.PLANNING,
            user_request=request,
        )
        self._append_history("user", request)

        if not self._llm_client:
            return {
                "action": "error",
                "message": "LLM 未配置，无法解析语义。",
            }

        messages = self._build_messages(request)
        response = self._call_llm(messages)
        if response is None:
            return {"action": "error", "message": "LLM 调用失败"}

        self._append_history("assistant", response)
        parsed = self._parse_llm_response(response)

        if parsed.get("action") == "parse":
            self.state.update(
                grab_target=parsed.get("grab_target"),
                deliver_target=parsed.get("deliver_target"),
                phase=DeliveryPhase.CONFIRMING,
            )
        return parsed

    def process_confirmation(self, confirmed: bool) -> dict:
        """处理用户确认。"""
        state = self.state.get()
        if state.phase != DeliveryPhase.CONFIRMING:
            return {"action": "error", "message": "当前不在确认阶段"}

        if not confirmed:
            self.state.transition(DeliveryPhase.IDLE)
            return {"action": "cancelled", "message": "任务已取消"}

        self.state.update(confirmed=True)
        return self._run_precheck()

    def _run_precheck(self) -> dict:
        """执行预检查。"""
        self.state.transition(DeliveryPhase.PRE_CHECKING)
        state = self.state.get()
        grab_target = state.grab_target
        deliver_target = state.deliver_target

        if not grab_target or not deliver_target:
            self.state.transition(DeliveryPhase.FAILED, error_message="缺少抓取或递送目标")
            return {"action": "error", "message": "缺少目标信息"}

        # 检查抓取目标
        grab_info = self.search_skill.search(grab_target)
        graspable = self.search_skill.check_graspable(grab_info)

        # 检查递送目标
        deliver_info = self.search_skill.search(deliver_target)
        deliver_info["nearby"] = deliver_info.get("found", False)

        self.state.update(
            grab_target_info={**grab_info, **graspable},
            deliver_target_info=deliver_info,
        )

        if not grab_info.get("found"):
            return {
                "action": "precheck_need_search",
                "message": f"附近没看到{grab_target}，需要我转一圈再找吗？",
            }

        if not graspable.get("graspable"):
            return {
                "action": "precheck_not_graspable",
                "message": f"{grab_target}{graspable.get('reason', '无法抓取')}，要换别的物品还是让我尝试一下？",
            }

        if not deliver_info.get("nearby"):
            return {
                "action": "precheck_destination_not_nearby",
                "message": f"附近没看到{deliver_target}，需要我转一圈再找吗？",
            }

        # 预检查通过
        self.state.update(precheck_passed=True)
        return self._start_execution()

    def _start_execution(self) -> dict:
        """启动执行编排器。"""
        state = self.state.get()
        if self._orchestrator is not None:
            self._orchestrator.stop()

        self._orchestrator = DeliveryOrchestrator(
            self.state,
            self.search_skill,
            self.approach_skill,
            self.grasp_skill,
            self.place_skill,
            self.config,
        )
        result = self._orchestrator.start()
        return {
            "action": "execute",
            "message": "预检查通过，开始执行递送任务。",
            "task_id": id(self._orchestrator),
        }

    def get_status(self) -> dict:
        """获取当前状态。"""
        state = self.state.get()
        return {
            "phase": state.phase.value,
            "grab_target": state.grab_target,
            "deliver_target": state.deliver_target,
            "held_object": state.held_object,
            "confirmed": state.confirmed,
            "precheck_passed": state.precheck_passed,
            "error_message": state.error_message,
        }

    def retry_search(self, target_type: str = "grab") -> dict:
        """旋转搜索后重试预检查。"""
        self.approach_skill.rotate_search(45)
        import time
        time.sleep(1.0)
        return self._run_precheck()

    def force_execute(self) -> dict:
        """用户选择强制尝试执行。"""
        self.state.update(precheck_passed=True)
        return self._start_execution()

    @staticmethod
    def _parse_llm_response(text: str) -> dict:
        """解析 LLM 返回的 JSON。"""
        if not text:
            return {"action": "error", "message": "LLM 返回为空"}
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    start = i + 1
                    break
            end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            return {"action": "error", "message": "LLM 返回不是 JSON 对象"}
        except Exception as e:
            logger.error(f"解析 LLM 响应失败: {e}, raw={text[:200]}")
            return {"action": "error", "message": f"解析失败: {e}"}

    def close(self):
        """释放资源。"""
        if self._orchestrator:
            self._orchestrator.stop()
        self.chassis_adapter.close()
        self.arm_adapter.close()
        self.vision_adapter.stop()
