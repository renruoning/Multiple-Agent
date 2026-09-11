"""报告生成节点：流程结束后，将各 Agent 的结果汇总为面向用户的出差方案。"""
from __future__ import annotations

from typing import Any

from ..config import build_chat_model
from ..core.context import ContextManager
from ..state import TravelState

SYSTEM_PROMPT = """你是企业智能商旅系统的汇总助手。
请将本次出差预订的各项结果（需求、航班、酒店、行程、预算审批、预订确认）整理成一份
简洁清晰、适合直接发给出差员工的中文方案说明，使用 Markdown 格式，
包含：出差概要、机票信息、酒店信息、逐日行程、预算审批结论、预订确认号。
"""

# 执行轨迹可能会随着 Supervisor 反复打回重选而变长，这里用通用的
# ContextManager 把喂给 LLM 的轨迹压缩到固定规模，避免无限增长。
_context_manager = ContextManager(keep_recent=8)


def _message_to_tuple(message: Any) -> tuple[str, str]:
    if isinstance(message, tuple):
        return message
    role = getattr(message, "type", "ai")
    content = getattr(message, "content", str(message))
    return (role, content)


def report_node(state: TravelState) -> dict:
    llm = build_chat_model()

    trace = [_message_to_tuple(m) for m in state.get("messages", [])]
    compacted_trace = _context_manager.compact(trace)
    trace_text = "\n".join(f"- [{role}] {content}" for role, content in compacted_trace)

    human_prompt = (
        f"需求: {state.get('requirements')}\n"
        f"航班: {state.get('selected_flight')}\n"
        f"酒店: {state.get('selected_hotel')}\n"
        f"行程: {state.get('itinerary')}\n"
        f"预算审批: {state.get('budget_check')}\n"
        f"预订确认: {state.get('booking_confirmation')}\n"
        f"执行轨迹（可能已压缩）:\n{trace_text}\n"
    )
    response = llm.invoke([("system", SYSTEM_PROMPT), ("human", human_prompt)])
    final_report = response.content

    return {
        "final_report": final_report,
        "messages": [("ai", "[报告Agent] 出差方案已生成完毕。")],
    }
