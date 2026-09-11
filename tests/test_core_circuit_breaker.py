"""CircuitBreaker 的单元测试，用可控的 fake clock 模拟时间流逝。"""
import pytest

from biz_travel_agents.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_stays_closed_below_failure_threshold():
    breaker = CircuitBreaker(failure_threshold=3)

    def failing():
        raise ValueError("boom")

    for _ in range(2):
        with pytest.raises(ValueError):
            breaker.call(failing)

    assert breaker.state == CircuitState.CLOSED


def test_opens_after_reaching_failure_threshold_and_fails_fast():
    breaker = CircuitBreaker(failure_threshold=2)
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        raise ValueError("boom")

    for _ in range(2):
        with pytest.raises(ValueError):
            breaker.call(failing)

    assert breaker.state == CircuitState.OPEN

    # 熔断后再调用不应该真正执行 fn，而是直接快速失败
    with pytest.raises(CircuitBreakerOpenError):
        breaker.call(failing)
    assert calls["n"] == 2


def test_transitions_to_half_open_after_recovery_timeout_and_closes_on_success():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, clock=clock)

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert breaker.state == CircuitState.OPEN

    clock.advance(5.0)
    assert breaker.state == CircuitState.OPEN  # 还没到冷却时间

    clock.advance(6.0)
    assert breaker.state == CircuitState.HALF_OPEN

    result = breaker.call(lambda: "ok")
    assert result == "ok"
    assert breaker.state == CircuitState.CLOSED


def test_half_open_probe_failure_reopens_circuit():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, clock=clock)

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    clock.advance(11.0)
    assert breaker.state == CircuitState.HALF_OPEN

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("boom again")))
    assert breaker.state == CircuitState.OPEN


def test_reset_returns_to_closed():
    breaker = CircuitBreaker(failure_threshold=1)
    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert breaker.state == CircuitState.OPEN

    breaker.reset()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.stats().consecutive_failures == 0
