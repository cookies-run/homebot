"""递送任务虚拟端到端测试（多场景）。

不连接真实机器人，全部用 mock 适配器/技能跑通：
plan → confirm → precheck → orchestrator(APPROACH→GRASP→FIND→APPROACH→PLACE)

包含场景：
1. 正常完整流程
2. 抓取目标不存在
3. 递送目标（人/地点）未找到
4. 目标无法抓取

同时打印每个技能被调用时的内部动作说明。
"""
import os
import sys
import time
import json
import itertools
from unittest.mock import MagicMock

# 将 src 加入路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC)

from applications.delivery_agent import DeliveryAgent, DeliveryAgentConfig
from applications.delivery_agent.state import SceneStateManager, DeliveryPhase


def print_stage(label: str, data=None):
    print(f"\n{'='*70}")
    print(f"【{label}】")
    if data is not None:
        if isinstance(data, dict):
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(data)


def reset_agent(persist_path: str):
    SceneStateManager._instance = None
    if os.path.exists(persist_path):
        os.remove(persist_path)


class MockSearchSkill:
    """带日志的搜索技能 mock。"""

    def __init__(self, scenario: str):
        self.scenario = scenario
        if scenario == "target_not_found":
            self._results = itertools.repeat({"found": False, "reason": "未在画面中找到目标"})
        elif scenario == "destination_not_found":
            self._results = iter([
                {"found": True, "bbox": [0.45, 0.45, 0.55, 0.65], "height_cm": 20, "pose": "upright"},
                {"found": False, "reason": "未在画面中找到目标"},
                {"found": False, "reason": "未在画面中找到目标"},
                {"found": False, "reason": "未在画面中找到目标"},
            ])
        elif scenario == "not_graspable":
            self._results = itertools.repeat({"found": True, "bbox": [0.2, 0.2, 0.8, 0.8], "height_cm": 60, "pose": "upright"})
        else:  # normal
            self._results = iter([
                {"found": True, "bbox": [0.45, 0.45, 0.55, 0.65], "height_cm": 20, "pose": "upright"},
                {"found": True, "bbox": [0.40, 0.40, 0.50, 0.70], "nearby": True},
            ])

    def search(self, target: str, max_retries: int = 2) -> dict:
        print(f"    [SearchSkill.search] 调用视觉/VLM 搜索目标: '{target}'")
        result = next(self._results)
        print(f"    [SearchSkill.search] 返回: {json.dumps(result, ensure_ascii=False)}")
        return result

    def check_graspable(self, grab_info: dict) -> dict:
        height = grab_info.get("height_cm", 0)
        if height > 25:
            print(f"    [SearchSkill.check_graspable] 判断：目标高度约{height}cm，超出机械臂安全抓取范围 → 不可抓取")
            return {"graspable": False, "reason": f"目标高度约{height}cm，超出机械臂安全抓取范围"}
        print(f"    [SearchSkill.check_graspable] 判断：目标高度{height}cm，姿态{grab_info.get('pose')} → 可抓取")
        return {"graspable": True, "reason": "可抓取"}


class MockApproachSkill:
    """带日志的接近技能 mock。"""

    def approach(self, target_bbox, timeout_s=30.0, update_callback=None) -> dict:
        print(f"    [ApproachSkill.approach] 锁定目标 bbox={target_bbox}，计算相对位置")
        print(f"    [ApproachSkill.approach] 发送底盘指令: 前进+转向，直到距离约 30cm")
        return {"success": True, "message": "已接近目标到约 30cm", "final_distance_cm": 30.0}

    def rotate_search(self, step_deg: float = 30.0) -> dict:
        print(f"    [ApproachSkill.rotate_search] 底盘原地旋转 {step_deg}° 搜索目标")
        return {"status": "success"}


class MockGraspSkill:
    """带日志的抓取技能 mock。"""

    def execute(self, target: str) -> dict:
        print(f"    [GraspSkill.execute] 启动 AutoGrabWorkflow，目标='{target}'")
        print(f"    [GraspSkill.execute] 动作序列: 机械臂定位 → 夹爪闭合 → 提起物体")
        return {"success": True, "message": f"已抓取{target}"}


class MockPlaceSkill:
    """带日志的放置技能 mock。"""

    def execute(self, destination_bbox=None, confirm_hand=False) -> dict:
        print(f"    [PlaceSkill.execute] 移动到释放姿态: base=-90, shoulder=30, elbow=120")
        if confirm_hand:
            print(f"    [PlaceSkill.execute] 末端摄像头确认手部位置（当前跳过）")
        print(f"    [PlaceSkill.execute] 打开夹爪释放物体")
        return {"status": "success", "message": "物体已释放"}

    def reset_arm(self) -> dict:
        print(f"    [PlaceSkill.reset_arm] 机械臂复位到休息位置")
        return {"status": "success"}


def make_mocked_skills(scenario: str):
    search_skill = MockSearchSkill(scenario)
    approach_skill = MockApproachSkill()
    grasp_skill = MockGraspSkill()
    place_skill = MockPlaceSkill()
    return search_skill, approach_skill, grasp_skill, place_skill


def run_scenario(name: str, scenario: str, user_request: str):
    print_stage(f"场景: {name}")

    persist_path = f"/tmp/sim_homebot_scene_state_{scenario}.json"
    reset_agent(persist_path)

    config = DeliveryAgentConfig(
        handbook_path=os.path.join(PROJECT_ROOT, "..", "agents", "delivery_manager.md"),
        scene_state_persist_path=persist_path,
    )
    agent = DeliveryAgent(config)

    # Mock LLM：直接返回结构化 parse 结果
    agent._llm_client = MagicMock()
    agent._llm_model = "virtual-model"
    agent._llm_client.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "action": "parse",
                        "grab_target": "矿泉水",
                        "deliver_target": "穿红衣服的人",
                        "constraints": [],
                        "message": "你要我拿起矿泉水，递给穿红衣服的人，对吗？",
                    }, ensure_ascii=False)
                }
            }
        ]
    }

    # Mock 适配器
    agent.vision_adapter = MagicMock()
    agent.chassis_adapter = MagicMock()
    agent.arm_adapter = MagicMock()

    # Mock 技能
    search_skill, approach_skill, grasp_skill, place_skill = make_mocked_skills(scenario)
    agent.search_skill = search_skill
    agent.approach_skill = approach_skill
    agent.grasp_skill = grasp_skill
    agent.place_skill = place_skill

    # 1. plan
    print("\n>> 1) plan_delivery_task")
    plan_result = agent.process_user_request(user_request)
    print_stage("plan 结果", plan_result)

    # 2. confirm
    print("\n>> 2) confirm_delivery_task (confirmed=True)")
    confirm_result = agent.process_confirmation(True)
    print_stage("confirm 结果", confirm_result)

    # 3. 等待编排器
    if confirm_result.get("action") == "execute":
        print("\n>> 3) 编排器后台执行阶段")
        print("    阶段推进：")
        last_phase = None
        for _ in range(40):
            status = agent.get_status()
            phase = status.get("phase")
            if phase != last_phase:
                print(f"      → {phase}")
                last_phase = phase
            if phase in (DeliveryPhase.COMPLETED.value, DeliveryPhase.FAILED.value):
                break
            time.sleep(0.05)

    print_stage("最终状态", agent.get_status())
    agent.close()


def main():
    scenarios = [
        ("正常完整流程", "normal", "把矿泉水递给穿红衣服的人"),
        ("抓取目标不存在", "target_not_found", "把矿泉水递给穿红衣服的人"),
        ("递送目标未找到", "destination_not_found", "把矿泉水递给穿红衣服的人"),
        ("目标无法抓取", "not_graspable", "把矿泉水递给穿红衣服的人"),
    ]

    for name, key, request in scenarios:
        run_scenario(name, key, request)

    print("\n" + "="*70)
    print("所有虚拟场景测试完成")


if __name__ == "__main__":
    main()
