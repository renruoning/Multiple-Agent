"""通用输入/输出护栏（Guardrails）：把数据交给下一个 Agent 之前做基本校验。

护栏本身与具体业务无关，只提供"规则 -> 违规信息列表"的通用形状；
具体规则由调用方组装（例如 requirement_agent 用它校验 LLM 解析出的
出差需求是否合理）。这里做的是"软校验"（只记录违规，不中断流程），
需要硬性拦截的场景可以在调用方根据返回的违规列表自行决定是否抛异常。
"""
from __future__ import annotations

from typing import Any, Callable

Rule = Callable[[dict[str, Any]], "str | None"]  # 返回 None 表示通过，否则返回违规描述


def check(data: dict[str, Any], rules: list[Rule]) -> list[str]:
    """依次执行规则，收集所有违规信息（不中断执行）。"""
    violations = []
    for rule in rules:
        message = rule(data)
        if message:
            violations.append(message)
    return violations


def field_positive(field_name: str) -> Rule:
    def rule(data: dict[str, Any]) -> str | None:
        value = data.get(field_name)
        if value is None or value <= 0:
            return f"字段 {field_name!r} 必须为正数，实际为 {value!r}"
        return None

    return rule


def date_order(start_field: str, end_field: str) -> Rule:
    def rule(data: dict[str, Any]) -> str | None:
        start, end = data.get(start_field), data.get(end_field)
        if start and end and start > end:
            return f"{start_field}({start}) 不应晚于 {end_field}({end})"
        return None

    return rule
