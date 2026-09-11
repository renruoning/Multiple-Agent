"""多智能体共享的全局状态（Shared State）定义。

LangGraph 中所有 Agent 节点都读写同一个 State，这是多智能体之间
协作与信息传递的核心机制（而不是靠 Agent 之间直接调用）。
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

AgentName = Literal["requirement", "flight", "hotel", "itinerary", "budget", "confirm", "FINISH"]


class TravelState(TypedDict, total=False):
    # 对话 / 执行轨迹，每个 Agent 执行后追加一条消息，便于审计与调试
    messages: Annotated[list, add_messages]

    # 用户原始出差请求
    user_request: str

    # 需求理解Agent 解析出的结构化需求
    requirements: dict[str, Any]

    # 机票预订Agent 产出
    flight_options: list[dict[str, Any]]
    selected_flight: Optional[dict[str, Any]]

    # 酒店预订Agent 产出
    hotel_options: list[dict[str, Any]]
    selected_hotel: Optional[dict[str, Any]]

    # 行程规划Agent 产出
    itinerary: Optional[dict[str, Any]]

    # 预算/审批/报销Agent 产出
    budget_check: Optional[dict[str, Any]]

    # 预订确认Agent 产出（幂等执行，见 tools/booking_tools.py）
    booking_confirmation: Optional[dict[str, Any]]

    # 协调/主控Agent（Supervisor）的路由决策
    next_agent: AgentName

    # 循环轮次计数，避免 Supervisor 无限往返调度
    iteration: int

    # 最终汇总给用户的出差方案报告
    final_report: Optional[str]
