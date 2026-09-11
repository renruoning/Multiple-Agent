"""机票/酒店预订Agent 在 LLM 不可用时的降级路径测试。"""
from biz_travel_agents.agents import flight_agent, hotel_agent
from biz_travel_agents.agents.flight_agent import flight_node
from biz_travel_agents.agents.hotel_agent import hotel_node


class _AlwaysFailingStructuredLLM:
    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("LLM API 不可用")


class _AlwaysFailingChatModel:
    def with_structured_output(self, _schema):
        return _AlwaysFailingStructuredLLM()


def test_flight_node_falls_back_to_cheapest_option_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(flight_agent, "build_chat_model", lambda: _AlwaysFailingChatModel())

    state = {
        "requirements": {
            "origin": "北京",
            "destination": "上海",
            "depart_date": "2026-01-10",
            "cabin_preference": "economy",
        },
        "iteration": 1,
    }
    update = flight_node(state)

    cheapest_price = min(o["price"] for o in update["flight_options"])
    assert update["selected_flight"]["price"] == cheapest_price
    assert "[降级]" in update["messages"][0][1]


def test_hotel_node_falls_back_to_cheapest_option_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(hotel_agent, "build_chat_model", lambda: _AlwaysFailingChatModel())

    state = {
        "requirements": {
            "origin": "北京",
            "destination": "上海",
            "depart_date": "2026-01-10",
            "return_date": "2026-01-12",
        },
        "iteration": 1,
    }
    update = hotel_node(state)

    cheapest_price = min(o["price_per_night"] for o in update["hotel_options"])
    assert update["selected_hotel"]["price_per_night"] == cheapest_price
    assert "[降级]" in update["messages"][0][1]
