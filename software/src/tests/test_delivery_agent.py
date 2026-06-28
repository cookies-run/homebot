"""递送智能体状态机测试。"""
from unittest.mock import MagicMock

import pytest

from applications.delivery_agent.agent import DeliveryAgent
from applications.delivery_agent.config import DeliveryAgentConfig
from applications.delivery_agent.state import SceneStateManager, DeliveryPhase


@pytest.fixture
def agent():
    """创建一个所有技能都被 mock 的 agent。"""
    # 清理单例，避免测试间状态污染
    SceneStateManager._instance = None

    config = DeliveryAgentConfig(
        handbook_path="agents/delivery_manager.md",
        scene_state_persist_path="/tmp/test_homebot_scene_state.json",
    )
    agent = DeliveryAgent(config)

    # Mock LLM 客户端
    agent._llm_client = MagicMock()
    agent._llm_model = "test-model"

    # Mock 技能
    agent.search_skill = MagicMock()
    agent.approach_skill = MagicMock()
    agent.grasp_skill = MagicMock()
    agent.place_skill = MagicMock()

    return agent


def test_process_user_request_parses_and_asks_confirmation(agent):
    """用户请求后应进入确认阶段。"""
    agent._llm_client.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"action": "parse", "grab_target": "矿泉水", "deliver_target": "穿红衣服的人", "message": "你要我拿起矿泉水，递给穿红衣服的人，对吗？"}',
                    "tool_calls": [],
                }
            }
        ]
    }

    result = agent.process_user_request("把矿泉水递给穿红衣服的人")

    assert result["action"] == "parse"
    assert agent.state.get().phase == DeliveryPhase.CONFIRMING
    assert agent.state.get().grab_target == "矿泉水"
    assert agent.state.get().deliver_target == "穿红衣服的人"


def test_confirmation_triggers_precheck(agent):
    """用户确认后应执行预检查。"""
    agent.state.update(
        phase=DeliveryPhase.CONFIRMING,
        grab_target="矿泉水",
        deliver_target="穿红衣服的人",
    )
    agent.search_skill.search.side_effect = [
        {"found": True, "bbox": [0.4, 0.4, 0.6, 0.6], "height_cm": 20, "pose": "upright"},
        {"found": True, "bbox": [0.3, 0.3, 0.5, 0.7], "nearby": True},
    ]
    agent.search_skill.check_graspable.return_value = {"graspable": True, "reason": "可抓取"}
    agent.approach_skill.approach.return_value = {"success": True, "message": "已接近"}
    agent.grasp_skill.execute.return_value = {"success": True, "message": "已抓取"}
    agent.place_skill.execute.return_value = {"success": True, "message": "已放置"}
    agent.place_skill.reset_arm.return_value = {"status": "success"}

    result = agent.process_confirmation(True)

    assert result["action"] == "execute"
    assert agent.state.get().precheck_passed is True


def test_precheck_reports_ungraspable(agent):
    """目标不可抓取时应返回异常确认。"""
    agent.state.update(
        phase=DeliveryPhase.CONFIRMING,
        grab_target="冰箱",
        deliver_target="人",
    )
    agent.search_skill.search.return_value = {"found": True, "bbox": [0.1, 0.1, 0.9, 0.9], "height_cm": 150}
    agent.search_skill.check_graspable.return_value = {"graspable": False, "reason": "太高"}

    result = agent.process_confirmation(True)

    assert result["action"] == "precheck_not_graspable"
    assert agent.state.get().phase == DeliveryPhase.PRE_CHECKING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
