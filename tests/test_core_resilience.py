"""call_with_resilience：重试 + 熔断 + 降级 组合行为的单元测试。"""
from biz_travel_agents.core.circuit_breaker import CircuitBreaker, CircuitState
from biz_travel_agents.core.resilience import call_with_resilience


def test_returns_primary_result_when_call_succeeds():
    breaker = CircuitBreaker(failure_threshold=3)
    result = call_with_resilience(
        lambda: "ok", breaker=breaker, fallback_fn=lambda: "fallback"
    )
    assert result == "ok"
    assert breaker.state == CircuitState.CLOSED


def test_falls_back_after_retries_exhausted_without_opening_breaker_prematurely():
    breaker = CircuitBreaker(failure_threshold=5)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise ValueError("boom")

    result = call_with_resilience(
        flaky,
        breaker=breaker,
        fallback_fn=lambda: "fallback",
        max_attempts=2,
        base_delay=0.01,
    )

    assert result == "fallback"
    assert calls["n"] == 2  # 内部重试了 2 次才放弃
    # 熔断器只把"这一次调用"记一次失败，不是记 2 次
    assert breaker.stats().consecutive_failures == 1


def test_falls_back_immediately_when_breaker_already_open():
    breaker = CircuitBreaker(failure_threshold=1)
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise ValueError("boom")

    # 先把熔断器打开
    call_with_resilience(
        always_fails, breaker=breaker, fallback_fn=lambda: "fallback", max_attempts=1
    )
    assert breaker.state == CircuitState.OPEN
    calls["n"] = 0

    result = call_with_resilience(
        always_fails, breaker=breaker, fallback_fn=lambda: "fallback-2", max_attempts=3
    )
    assert result == "fallback-2"
    assert calls["n"] == 0  # 熔断已经 OPEN，根本没有真正调用 always_fails
