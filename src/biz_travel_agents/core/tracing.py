"""轻量级结构化执行追踪，用于观测多智能体系统每一步的执行情况。

不依赖任何外部可观测性后端，只是把事件收集到内存里；真实项目里可以在
`Tracer.emit` 内对接 OpenTelemetry / 结构化日志系统，Agent 代码不需要改动。
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class TraceEvent:
    node: str
    status: str  # "success" | "error"
    duration_ms: float
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self):
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()

    def as_dicts(self) -> list[dict[str, Any]]:
        return [event.__dict__ for event in self.events]


def traced_node(tracer: Tracer, name: str):
    """包裹一个 Agent/节点函数，记录其执行耗时与成败，异常照常向上抛出。"""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                duration_ms = (time.perf_counter() - start) * 1000
                tracer.emit(
                    TraceEvent(node=name, status="error", duration_ms=duration_ms, error=str(exc))
                )
                raise
            else:
                duration_ms = (time.perf_counter() - start) * 1000
                tracer.emit(TraceEvent(node=name, status="success", duration_ms=duration_ms))
                return result

        return wrapper

    return decorator
