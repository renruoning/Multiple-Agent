"""预算/审批/报销Agent：校验方案是否符合企业差旅政策，并生成报销单据。"""
from __future__ import annotations

from ..state import TravelState
from ..tools.policy_tools import check_policy


def budget_node(state: TravelState) -> dict:
    req = state["requirements"]
    flight = state["selected_flight"]
    hotel = state["selected_hotel"]

    travelers = req.get("travelers", 1)
    total_cost = flight["price"] * travelers + hotel["total_price"] * travelers

    result = check_policy(
        total_cost=total_cost,
        cabin=flight["cabin"],
        hotel_price_per_night=hotel["price_per_night"],
        budget_limit=req.get("budget"),
    )

    if result["approved"]:
        summary = (
            f"[预算审批Agent] 审批通过 ✅，总花费 {result['total_cost']:.0f} 元，"
            f"预算上限 {result['budget_limit']:.0f} 元。"
        )
    else:
        summary = (
            f"[预算审批Agent] 审批未通过 ❌：{'；'.join(result['violations'])}。"
            "已退回给机票/酒店Agent 重新选择更经济的方案。"
        )

    return {
        "budget_check": result,
        "messages": [("ai", summary)],
        "iteration": state.get("iteration", 0) + 1,
    }
