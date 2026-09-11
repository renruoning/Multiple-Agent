"""通用降级（Fallback / Graceful Degradation）组件。

当主路径（真实 LLM 调用、真实机票/酒店查询接口）失败、或被熔断器判定为
不可用时，比起让整个多智能体流程直接崩掉，更好的做法通常是降级：
返回一份规则兜底/缓存/默认数据，牺牲一部分"智能程度"换取核心链路可用。
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_fallback(
    fallback_fn: Callable[..., T],
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_fallback: Callable[[BaseException], None] | None = None,
):
    """主逻辑抛出 `exceptions` 中的异常时，改为调用 `fallback_fn`（同样的参数）。"""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return fn(*args, **kwargs)
            except exceptions as exc:  # noqa: BLE001
                if on_fallback is not None:
                    on_fallback(exc)
                else:
                    logger.warning("%s 失败，已降级：%s", fn.__name__, exc)
                return fallback_fn(*args, **kwargs)

        return wrapper

    return decorator


def static_fallback(value: T) -> Callable[..., T]:
    """返回一个忽略所有参数、直接给出固定降级值的 fallback 函数。"""

    def _fallback(*_args: Any, **_kwargs: Any) -> T:
        return value

    return _fallback
