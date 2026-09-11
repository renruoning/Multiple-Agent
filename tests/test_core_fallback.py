"""with_fallback / static_fallback 的单元测试。"""
import pytest

from biz_travel_agents.core.fallback import static_fallback, with_fallback


def test_with_fallback_returns_primary_result_when_no_error():
    @with_fallback(static_fallback("fallback"))
    def primary():
        return "primary"

    assert primary() == "primary"


def test_with_fallback_invokes_fallback_on_error():
    @with_fallback(static_fallback("fallback-value"))
    def primary():
        raise RuntimeError("boom")

    assert primary() == "fallback-value"


def test_with_fallback_passes_through_args_to_fallback():
    def fallback(x, y):
        return f"fallback-{x}-{y}"

    @with_fallback(fallback)
    def primary(x, y):
        raise RuntimeError("boom")

    assert primary(1, 2) == "fallback-1-2"


def test_with_fallback_only_catches_declared_exceptions():
    @with_fallback(static_fallback("fallback"), exceptions=(ValueError,))
    def primary():
        raise TypeError("not covered")

    with pytest.raises(TypeError):
        primary()


def test_with_fallback_calls_on_fallback_hook():
    seen = []

    @with_fallback(static_fallback("x"), on_fallback=lambda exc: seen.append(str(exc)))
    def primary():
        raise RuntimeError("boom")

    primary()
    assert seen == ["boom"]
