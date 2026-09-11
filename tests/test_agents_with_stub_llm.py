"""用桩（stub）LLM 测试 Agent 的业务逻辑，不依赖真实 API Key。

真实项目中 LLM 输出不确定，因此单测只针对"给定 LLM 决策后，
Agent 节点是否正确更新共享 State"这一部分做验证，
是否针对不同请求给出正确判断可留给 LLM 层面的评测/集成测试。
"""
from biz_travel_agents.agents import requirement_agent, supervisor
from biz_travel_agents.agents.requirement_agent import TripRequirement, requirement_node
from biz_travel_agents.agents.supervisor import RouteDecision, supervisor_node


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, *_args, **_kwargs):
        return self._result


class _FakeChatModel:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, _schema):
        return _FakeStructuredLLM(self._result)


def test_requirement_node_updates_state_from_stub_llm(monkeypatch):
    fake_result = TripRequirement(
        origin="北京",
        destination="上海",
        depart_date="2026-01-10",
        return_date="2026-01-12",
        travelers=1,
        budget=4000,
        cabin_preference="economy",
        purpose="拜访客户",
    )
    monkeypatch.setattr(
        requirement_agent, "build_chat_model", lambda: _FakeChatModel(fake_result)
    )

    state = {"user_request": "帮我订下周一到上海出差的机票和酒店", "iteration": 0}
    update = requirement_node(state)

    assert update["requirements"]["origin"] == "北京"
    assert update["requirements"]["destination"] == "上海"
    assert update["iteration"] == 1
    assert update["messages"][0][0] == "ai"


def test_supervisor_routes_per_stub_decision(monkeypatch):
    fake_result = RouteDecision(next="flight", reason="需求已解析，尚缺航班信息")
    monkeypatch.setattr(supervisor, "build_chat_model", lambda: _FakeChatModel(fake_result))

    update = supervisor_node({"user_request": "...", "iteration": 1})
    assert update["next_agent"] == "flight"


def test_supervisor_force_finish_at_max_iterations():
    update = supervisor_node({"iteration": 999})
    assert update["next_agent"] == "FINISH"


def test_supervisor_routes_to_confirm_after_budget_approved(monkeypatch):
    fake_result = RouteDecision(next="confirm", reason="预算已通过，尚未确认预订")
    monkeypatch.setattr(supervisor, "build_chat_model", lambda: _FakeChatModel(fake_result))

    update = supervisor_node(
        {
            "iteration": 5,
            "budget_check": {"approved": True},
            "booking_confirmation": None,
        }
    )
    assert update["next_agent"] == "confirm"
