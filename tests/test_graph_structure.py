"""验证 LangGraph 执行图的拓扑结构（不调用 LLM，无需 API Key）。"""
from biz_travel_agents.graph import DOMAIN_AGENTS, build_graph


def test_graph_compiles_with_expected_nodes():
    app = build_graph()
    node_names = set(app.get_graph().nodes.keys())

    expected = {"supervisor", "report", *DOMAIN_AGENTS}
    assert expected.issubset(node_names)


def test_route_from_supervisor_maps_finish_to_report():
    from biz_travel_agents.graph import _route_from_supervisor

    assert _route_from_supervisor({"next_agent": "FINISH"}) == "report"
    assert _route_from_supervisor({"next_agent": "flight"}) == "flight"
