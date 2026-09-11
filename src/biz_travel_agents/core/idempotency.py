"""通用幂等性执行组件。

在多智能体系统里，任何"有副作用"的操作（真正下单、真正扣费、真正发消息）
如果因为网络重试、Agent 重新规划、Supervisor 重复调度等原因被执行了多次，
后果往往是不可逆的（比如重复订票、重复扣款）。`IdempotencyStore` 提供一个
通用的"同一个 key 只真正执行一次"的保护层，可以包裹任意有副作用的函数使用。
"""
from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ExecutionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class _Record:
    status: ExecutionStatus
    result: Any = None
    error: BaseException | None = None
    created_at: float = field(default_factory=time.monotonic)


class IdempotencyStore:
    """基于内存字典的幂等执行存储。

    生产环境可以把内部存储换成 Redis / 数据库，对外接口保持不变即可。
    """

    def __init__(self, ttl_seconds: float | None = 3600):
        self._ttl_seconds = ttl_seconds
        self._records: dict[str, _Record] = {}
        self._lock = threading.Lock()

    def _is_expired(self, record: _Record) -> bool:
        if self._ttl_seconds is None:
            return False
        return (time.monotonic() - record.created_at) > self._ttl_seconds

    def run(self, key: str, fn: Callable[[], T]) -> T:
        """以 key 为幂等键执行 fn；同一个 key 只会真正执行一次副作用。

        - key 从未出现过或已过期：真正执行 fn，并缓存结果。
        - key 已经成功执行过：直接返回缓存结果，不再重复执行 fn。
        - key 上一次执行失败：本次会重新尝试执行 fn。
        - key 正在执行中（并发重入场景）：抛出异常，避免并发重复执行。
        """
        with self._lock:
            record = self._records.get(key)
            if record is not None and not self._is_expired(record):
                if record.status == ExecutionStatus.SUCCEEDED:
                    return record.result
                if record.status == ExecutionStatus.IN_PROGRESS:
                    raise RuntimeError(f"幂等键 {key!r} 正在执行中，拒绝并发重复执行")
                # FAILED 或已过期：允许重试，继续往下真正执行
            self._records[key] = _Record(status=ExecutionStatus.IN_PROGRESS)

        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 - 记录任意异常后重新抛出
            with self._lock:
                self._records[key] = _Record(status=ExecutionStatus.FAILED, error=exc)
            raise
        else:
            with self._lock:
                self._records[key] = _Record(status=ExecutionStatus.SUCCEEDED, result=result)
            return result

    def status_of(self, key: str) -> ExecutionStatus | None:
        record = self._records.get(key)
        return record.status if record else None

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


def idempotent(store: IdempotencyStore, key_fn: Callable[..., str]):
    """装饰器写法：`@idempotent(store, key_fn=lambda *a, **kw: ...)`。"""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = key_fn(*args, **kwargs)
            return store.run(key, lambda: fn(*args, **kwargs))

        return wrapper

    return decorator
