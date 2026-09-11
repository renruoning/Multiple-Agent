"""通用重试组件：对瞬时性失败（网络抖动、限流、超时）做指数退避重试。

典型用法是包裹外部调用：LLM API 调用、真实的机票/酒店预订 API 等。
"""
from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.1,
    retriable_exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep_fn: Callable[[float], None] = time.sleep,
):
    """指数退避重试装饰器。

    第 i 次重试（从 0 开始）等待 ``min(base_delay * 2**i, max_delay)``，
    再加上 ``[0, jitter*delay]`` 的随机抖动，避免大量失败请求同时重试
    造成"重试风暴"。耗尽 `max_attempts` 后重新抛出最后一次的异常。
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except retriable_exceptions as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt == max_attempts - 1:
                        break
                    delay = min(base_delay * (2**attempt), max_delay)
                    delay += random.uniform(0, jitter * delay)
                    sleep_fn(delay)
            assert last_error is not None
            raise last_error

        return wrapper

    return decorator
