"""预订确认Agent节点的测试：不涉及 LLM，纯校验 State 更新与幂等复用。"""
from biz_travel_agents.agents.confirmation_agent import confirm_node
from biz_travel_agents.tools.booking_tools import reset_booking_store_for_tests


def _base_state():
    return {
        "requirements": {"travelers": 2},
        "selected_flight": {"flight_no": "CA1234", "date": "2026-01-10"},
        "selected_hotel": {"hotel_name": "上海希尔顿酒店", "checkin": "2026-01-10"},
        "iteration": 5,
    }


def test_confirm_node_updates_state_and_is_idempotent_across_calls():
    reset_booking_store_for_tests()
    state = _base_state()

    first = confirm_node(state)
    assert first["booking_confirmation"]["idempotent_hit"] is False
    assert first["iteration"] == 6

    # 模拟 Supervisor 因为某种原因把 confirm 又调度了一次
    second = confirm_node(state)
    assert second["booking_confirmation"]["confirmation_id"] == (
        first["booking_confirmation"]["confirmation_id"]
    )
    assert second["booking_confirmation"]["idempotent_hit"] is True
