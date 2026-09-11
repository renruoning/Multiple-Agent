"""IdempotencyStore 的单元测试（纯 Python，无需 LLM/API Key）。"""
import pytest

from biz_travel_agents.core.idempotency import IdempotencyStore, idempotent


def test_run_executes_once_for_same_key():
    store = IdempotencyStore()
    calls = []

    def side_effecting():
        calls.append(1)
        return "result"

    assert store.run("k1", side_effecting) == "result"
    assert store.run("k1", side_effecting) == "result"
    assert len(calls) == 1  # 第二次直接命中缓存，没有真正执行


def test_run_allows_retry_after_failure():
    store = IdempotencyStore()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return "ok"

    with pytest.raises(ValueError):
        store.run("k2", flaky)

    assert store.run("k2", flaky) == "ok"
    assert calls["n"] == 2


def test_idempotent_decorator_dedupes_by_key_fn():
    store = IdempotencyStore()
    calls = []

    @idempotent(store, key_fn=lambda order_id: order_id)
    def place_order(order_id: str):
        calls.append(order_id)
        return f"confirmed-{order_id}"

    assert place_order("A1") == "confirmed-A1"
    assert place_order("A1") == "confirmed-A1"
    assert place_order("A2") == "confirmed-A2"
    assert calls == ["A1", "A2"]
