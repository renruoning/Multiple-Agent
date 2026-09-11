"""Tracer / traced_node 的单元测试。"""
import pytest

from biz_travel_agents.core.tracing import Tracer, traced_node


def test_traced_node_records_success():
    tracer = Tracer()

    @traced_node(tracer, "demo")
    def node(state):
        return {"ok": True}

    result = node({})
    assert result == {"ok": True}
    assert len(tracer.events) == 1
    assert tracer.events[0].node == "demo"
    assert tracer.events[0].status == "success"
    assert tracer.events[0].duration_ms >= 0


def test_traced_node_records_error_and_reraises():
    tracer = Tracer()

    @traced_node(tracer, "demo")
    def node(state):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        node({})

    assert tracer.events[0].status == "error"
    assert "boom" in tracer.events[0].error


def test_tracer_clear_and_as_dicts():
    tracer = Tracer()

    @traced_node(tracer, "demo")
    def node(state):
        return {}

    node({})
    assert len(tracer.as_dicts()) == 1
    tracer.clear()
    assert tracer.events == []
