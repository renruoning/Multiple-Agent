"""协调/主控Agent（Supervisor）：动态决定下一步该调用哪个专业Agent。

这是整个多智能体框架的调度核心：其余 Agent 只负责各自领域的任务，
彼此互不直接调用，所有的流转都通过 Supervisor 读取共享 State 后决策。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..config import build_chat_model
from ..core.retry import retry_with_backoff
from ..state import TravelState

MAX_ITERATIONS = 12

SYSTEM_PROMPT = """你是企业智能商旅多智能体系统的「协调/主控Agent」（Supervisor）。
你负责根据当前任务状态，决定下一步应该调用哪个专业Agent，或判断整个出差预订流程已经完成。

可选的 Agent：
- requirement：需求理解Agent，从用户原始请求中提取结构化出差需求
- flight：机票预订Agent，查询并选择合适航班
- hotel：酒店预订Agent，查询并选择合适酒店
- itinerary：行程规划Agent，将机票和酒店信息整合为逐日行程
- budget：预算/审批/报销Agent，校验总花费是否符合企业差旅政策
- confirm：预订确认Agent，在预算审批通过后调用（模拟的）下单接口正式确认
  机票和酒店预订；该步骤具备幂等性，重复调度不会重复下单
- FINISH：所有必要步骤都已完成，且已生成预订确认，可以结束流程

调度规则：
1. 典型顺序是 requirement → flight → hotel → itinerary → budget → confirm → FINISH。
2. 尚未解析需求（requirements 为空）时，必须先路由到 requirement。
3. 已有需求但缺少航班/酒店时，路由到 flight / hotel。
4. 航班和酒店都已选定但还没有 itinerary 时，路由到 itinerary。
5. itinerary 已生成但还没做预算审批（budget_check 为空）时，路由到 budget。
6. 如果 budget_check.approved 为 False，说明超预算或违反政策，必须路由回 flight
   （或 hotel，取决于是机票还是酒店导致超支）让其重新选择更经济的方案，
   不能直接 FINISH。
7. 如果 budget_check.approved 为 True 但还没有 booking_confirmation，路由到 confirm。
8. 如果 booking_confirmation 已存在，直接 FINISH。
9. 如果 iteration 已经很大（接近上限），为避免死循环，直接 FINISH 并说明原因。
"""


class RouteDecision(BaseModel):
    next: Literal[
        "requirement", "flight", "hotel", "itinerary", "budget", "confirm", "FINISH"
    ] = Field(description="下一步应该执行的 Agent，或 FINISH 表示流程结束")
    reason: str = Field(description="做出该路由决策的简要理由，一句话")


def _summarize_state(state: TravelState) -> str:
    return (
        f"user_request: {state.get('user_request')}\n"
        f"requirements: {state.get('requirements')}\n"
        f"selected_flight: {state.get('selected_flight')}\n"
        f"selected_hotel: {state.get('selected_hotel')}\n"
        f"itinerary_present: {bool(state.get('itinerary'))}\n"
        f"budget_check: {state.get('budget_check')}\n"
        f"booking_confirmation: {state.get('booking_confirmation')}\n"
        f"iteration: {state.get('iteration', 0)}\n"
    )


@retry_with_backoff(max_attempts=3, base_delay=0.3, retriable_exceptions=(Exception,))
def _invoke_structured(structured_llm, messages) -> RouteDecision:
    # Supervisor 每一轮都要调 LLM，是整个图里被调用最频繁的一环，
    # 也最值得包一层重试：LLM API 的限流/超时等瞬时错误不应该直接
    # 打断整个多智能体流程。
    return structured_llm.invoke(messages)


def supervisor_node(state: TravelState) -> dict:
    iteration = state.get("iteration", 0)
    if iteration >= MAX_ITERATIONS:
        return {
            "next_agent": "FINISH",
            "messages": [("ai", "[协调Agent] 已达到最大迭代次数，强制结束流程。")],
        }

    llm = build_chat_model()
    structured_llm = llm.with_structured_output(RouteDecision)
    decision = _invoke_structured(
        structured_llm, [("system", SYSTEM_PROMPT), ("human", _summarize_state(state))]
    )

    summary = f"[协调Agent] 下一步 → {decision.next}。理由：{decision.reason}"
    return {
        "next_agent": decision.next,
        "messages": [("ai", summary)],
    }
