r"""构建智能商旅多智能体框架的 LangGraph 执行图。

拓扑结构（Supervisor 模式）：

    START -> supervisor -> {requirement, flight, hotel, itinerary, budget, confirm} -> supervisor
                        \-> report -> END

Supervisor 每次决策后都会回到自身重新评估状态，直到判定 FINISH，
再由 report 节点汇总出最终方案。每个节点都用 `traced_node` 包裹，
运行结束后可以通过传入的 `tracer` 查看每一步的耗时与成败（见 core/tracing.py）。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import (
    budget_node,
    confirm_node,
    flight_node,
    hotel_node,
    itinerary_node,
    report_node,
    requirement_node,
    supervisor_node,
)
from .core.tracing import Tracer, traced_node
from .state import TravelState

DOMAIN_AGENTS = ("requirement", "flight", "hotel", "itinerary", "budget", "confirm")

_NODE_FUNCS = {
    "requirement": requirement_node,
    "flight": flight_node,
    "hotel": hotel_node,
    "itinerary": itinerary_node,
    "budget": budget_node,
    "confirm": confirm_node,
    "report": report_node,
}


def _route_from_supervisor(state: TravelState) -> str:
    next_agent = state.get("next_agent", "requirement")
    if next_agent == "FINISH":
        return "report"
    return next_agent


def build_graph(tracer: Tracer | None = None):
    """构建并编译执行图。

    传入外部 `tracer` 可以在运行结束后查看每个节点的执行耗时/成败
    （见 `main.py` 的用法）；不传则每次调用内部新建一个，互不影响，
    方便在测试里重复调用而不产生共享状态。
    """
    tracer = tracer if tracer is not None else Tracer()

    graph = StateGraph(TravelState)

    graph.add_node("supervisor", traced_node(tracer, "supervisor")(supervisor_node))
    for name, fn in _NODE_FUNCS.items():
        graph.add_node(name, traced_node(tracer, name)(fn))

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
