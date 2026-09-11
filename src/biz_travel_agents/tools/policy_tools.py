"""模拟企业差旅政策引擎，用于预算/审批/报销Agent 做合规校验。"""
from __future__ import annotations

from typing import Any

DEFAULT_POLICY: dict[str, Any] = {
    "max_total_budget": 5000,
    "max_hotel_price_per_night": 800,
    "allowed_cabin": ["economy"],
}


def check_policy(
    total_cost: float,
    cabin: str,
    hotel_price_per_night: float,
    budget_limit: float | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """校验本次出差方案是否符合企业差旅政策与预算上限。"""
    policy = policy or DEFAULT_POLICY
    limit = budget_limit if budget_limit is not None else policy["max_total_budget"]

    violations: list[str] = []
    if total_cost > limit:
        violations.append(f"总花费 {total_cost:.0f} 元超过预算上限 {limit:.0f} 元")
    if cabin not in policy["allowed_cabin"]:
        violations.append(f"舱位「{cabin}」不在企业允许范围 {policy['allowed_cabin']} 内")
    if hotel_price_per_night > policy["max_hotel_price_per_night"]:
        violations.append(
            f"酒店单价 {hotel_price_per_night:.0f} 元/晚超过限额 "
            f"{policy['max_hotel_price_per_night']:.0f} 元/晚"
        )

    return {
        "approved": len(violations) == 0,
        "violations": violations,
        "total_cost": total_cost,
        "budget_limit": limit,
    }
