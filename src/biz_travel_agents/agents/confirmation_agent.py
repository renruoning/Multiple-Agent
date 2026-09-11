"""预订确认Agent：预算审批通过后，真正调用（模拟的）下单接口完成确认。

这是整个流程里唯一有真实副作用的一步，因此必须保证幂等：无论 Supervisor
因为什么原因把这个节点又调度了一次，同一笔行程只会真正下单一次
（见 `tools/booking_tools.py` 与 `core/idempotency.py`）。
"""
from __future__ import annotations

from ..state import TravelState
from ..tools.booking_tools import confirm_booking


def confirm_node(state: TravelState) -> dict:
    flight = state["selected_flight"]
    hotel = state["selected_hotel"]
    req = state["requirements"]

    result = confirm_booking(flight=flight, hotel=hotel, travelers=req.get("travelers", 1))

    verb = "复用了已有的" if result["idempotent_hit"] else "首次生成"
    summary = f"[预订确认Agent] {verb}确认单，确认号 {result['confirmation_id']}。"
    return {
        "booking_confirmation": result,
        "messages": [("ai", summary)],
        "iteration": state.get("iteration", 0) + 1,
    }
