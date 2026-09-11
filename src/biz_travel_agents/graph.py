r"""构建智能商旅多智能体框架的 LangGraph 执行图。

拓扑结构（Supervisor 模式）：

    START -> supervisor -> {requirement, flight, hotel, itinerary, budget} -> supervisor
                        \-> report -> END

Supervisor 每次决策后都会回到自身重新评估状态，直到判定 FINISH，
再由 report 节点汇总出最终方案。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import (
    budget_node,
    flight_node,
    hotel_node,
    itinerary_node,
    report_node,
    requirement_node,
    supervisor_node,
)
from .state import TravelState

DOMAIN_AGENTS = ("requirement", "flight", "hotel", "itinerary", "budget")


def _route_from_supervisor(state: TravelState) -> str:
    next_agent = state.get("next_agent", "requirement")
    if next_agent == "FINISH":
        return "report"
    return next_agent


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("requirement", requirement_node)
    graph.add_node("flight", flight_node)
    graph.add_node("hotel", hotel_node)
    graph.add_node("itinerary", itinerary_node)
    graph.add_node("budget", budget_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {**{name: name for name in DOMAIN_AGENTS}, "report": "report"},
    )

    for name in DOMAIN_AGENTS:
        graph.add_edge(name, "supervisor")

    graph.add_edge("report", END)

    return graph.compile()
