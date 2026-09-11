"""retry_with_backoff 的单元测试（用可控的 sleep_fn，测试不会真的等待）。"""
import pytest

from biz_travel_agents.core.retry import retry_with_backoff


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}
    sleeps = []

    @retry_with_backoff(max_attempts=3, base_delay=0.01, sleep_fn=sleeps.append)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2  # 前两次失败后各 sleep 一次


def test_retry_raises_last_error_after_exhausting_attempts():
    @retry_with_backoff(max_attempts=2, base_delay=0.01, sleep_fn=lambda _delay: None)
    def always_fails():
        raise ValueError("still broken")

    with pytest.raises(ValueError, match="still broken"):
        always_fails()


def test_retry_does_not_catch_non_retriable_exceptions():
    @retry_with_backoff(
        max_attempts=3, base_delay=0.01, sleep_fn=lambda _delay: None,
        retriable_exceptions=(ConnectionError,),
    )
    def raises_type_error():
        raise TypeError("not retriable")

    with pytest.raises(TypeError):
        raises_type_error()
