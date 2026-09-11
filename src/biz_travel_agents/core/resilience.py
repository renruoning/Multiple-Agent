"""把 重试 + 熔断 + 降级 三层保护组合成一次调用的标准姿势。

这是本框架里所有"调用外部依赖"（尤其是 LLM）的推荐用法：
1. 单次调用内的瞬时抖动，交给 `retry_with_backoff` 处理；
2. 如果连续多次调用（重试耗尽后仍失败）都失败，交给熔断器直接快速
   失败，不再浪费时间重试，也不再拖累下游；
3. 无论是哪种失败，最终都应该有一个降级方案兜底，保证核心流程不中断
   （见各 Agent 里的 `_fallback_*` 函数）。
"""
from __future__ import annotations

from typing import Callable, TypeVar

from .circuit_breaker import CircuitBreaker
from .fallback import with_fallback
from .retry import retry_with_backoff

T = TypeVar("T")


def call_with_resilience(
    fn: Callable[[], T],
    *,
    breaker: CircuitBreaker,
    fallback_fn: Callable[[], T],
    max_attempts: int = 2,
    base_delay: float = 0.2,
) -> T:
    """执行 fn：先在熔断器保护下做有限次重试，全部失败（或熔断器已 OPEN）
    则调用 `fallback_fn` 得到一个降级结果，而不是让异常向上传播。
    """
    retried = retry_with_backoff(max_attempts=max_attempts, base_delay=base_delay)(fn)

    @with_fallback(fallback_fn)
    def protected() -> T:
        return breaker.call(retried)

    return protected()
