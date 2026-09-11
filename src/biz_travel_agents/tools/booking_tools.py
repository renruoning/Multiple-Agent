"""模拟真实的机票/酒店下单确认接口（有副作用，因此需要幂等保护）。

真实场景里这里会调用航司/OTA 的下单 API；如果因为 Supervisor 重复调度、
网络重试等原因被多次调用，意味着重复购票、重复扣款，所以用
`IdempotencyStore` 包裹，保证同一笔行程只会真正下单一次。
"""
from __future__ import annotations

import hashlib
from typing import Any

from ..core.idempotency import ExecutionStatus, IdempotencyStore

_store = IdempotencyStore()
# 模拟"下游下单接口真正被调用了几次"，仅用于测试验证幂等确实生效
_booking_counter = {"count": 0}


def _make_key(flight: dict[str, Any], hotel: dict[str, Any], travelers: int) -> str:
    raw = f"{flight['flight_no']}|{flight['date']}|{hotel['hotel_name']}|{hotel['checkin']}|{travelers}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _do_confirm(flight: dict[str, Any], hotel: dict[str, Any], travelers: int, key: str) -> dict:
    _booking_counter["count"] += 1
    return {
        "confirmation_id": f"CONF-{key}",
        "flight_no": flight["flight_no"],
        "hotel_name": hotel["hotel_name"],
        "travelers": travelers,
    }


def confirm_booking(flight: dict[str, Any], hotel: dict[str, Any], travelers: int = 1) -> dict:
    """确认预订。多次调用同一笔行程只会真正下单一次（幂等）。"""
    key = _make_key(flight, hotel, travelers)
    was_already_done = _store.status_of(key) == ExecutionStatus.SUCCEEDED
    result = _store.run(key, lambda: _do_confirm(flight, hotel, travelers, key))
    return {**result, "idempotent_hit": was_already_done}


def reset_booking_store_for_tests() -> None:
    _store.clear()
    _booking_counter["count"] = 0


def booking_call_count_for_tests() -> int:
    return _booking_counter["count"]
