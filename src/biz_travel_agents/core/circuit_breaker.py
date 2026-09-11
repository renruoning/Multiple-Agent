"""通用熔断器（Circuit Breaker）。

当下游服务（LLM API、机票/酒店查询接口等）持续失败时，如果还在不断
重试/调用，只会让故障雪上加霜——拖慢自己、加重对方负担。熔断器在连续
失败次数超过阈值后主动"跳闸"（OPEN），在冷却时间内直接快速失败，不再
真正发起调用；冷却结束后放行一次"试探请求"（HALF_OPEN），成功则恢复
（CLOSED），失败则继续保持熔断。

状态机：CLOSED --(连续失败达到阈值)--> OPEN --(冷却超时)--> HALF_OPEN
        HALF_OPEN --(试探成功)--> CLOSED
        HALF_OPEN --(试探失败)--> OPEN（重新计时）
"""
from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(RuntimeError):
    """熔断器处于 OPEN（或 HALF_OPEN 试探名额已用完）时，调用被直接拒绝。"""


@dataclass
class CircuitBreakerStats:
    state: CircuitState
    consecutive_failures: int
    opened_at: float | None


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._clock = clock

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_calls_in_flight = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def _maybe_transition_to_half_open(self) -> None:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if self._clock() - self._opened_at >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls_in_flight = 0

    def _before_call(self) -> None:
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.OPEN:
                remaining = self._recovery_timeout - (self._clock() - (self._opened_at or 0))
                raise CircuitBreakerOpenError(
                    f"熔断器处于 OPEN 状态，距离下次尝试还有 {max(remaining, 0):.1f}s"
                )
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls_in_flight >= self._half_open_max_calls:
                    raise CircuitBreakerOpenError("熔断器处于 HALF_OPEN 状态，试探名额已用完")
                self._half_open_calls_in_flight += 1

    def _on_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._half_open_calls_in_flight = 0

    def _on_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                # 试探失败：立刻回到 OPEN 并重新计时冷却
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._half_open_calls_in_flight = 0
            elif self._consecutive_failures >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()

    def call(self, fn: Callable[[], T]) -> T:
        """执行 fn；若熔断器处于 OPEN，直接抛出 `CircuitBreakerOpenError`
        而不真正调用 fn（快速失败）。"""
        self._before_call()
        try:
            result = fn()
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def stats(self) -> CircuitBreakerStats:
        with self._lock:
            return CircuitBreakerStats(
                state=self._state,
                consecutive_failures=self._consecutive_failures,
                opened_at=self._opened_at,
            )

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_calls_in_flight = 0


def circuit_protected(breaker: CircuitBreaker):
    """装饰器写法：`@circuit_protected(breaker)`。"""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(lambda: fn(*args, **kwargs))

        return wrapper

    return decorator
