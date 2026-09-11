"""预订确认工具的幂等性测试：验证"下游下单接口"只会被真正调用一次。"""
from biz_travel_agents.tools.booking_tools import (
    booking_call_count_for_tests,
    confirm_booking,
    reset_booking_store_for_tests,
)


def test_confirm_booking_is_idempotent():
    reset_booking_store_for_tests()
    flight = {"flight_no": "CA1234", "date": "2026-01-10"}
    hotel = {"hotel_name": "上海希尔顿酒店", "checkin": "2026-01-10"}

    first = confirm_booking(flight, hotel, travelers=1)
    second = confirm_booking(flight, hotel, travelers=1)

    assert first["confirmation_id"] == second["confirmation_id"]
    assert first["idempotent_hit"] is False
    assert second["idempotent_hit"] is True
    assert booking_call_count_for_tests() == 1  # 下游"真正下单接口"只被调用了一次


def test_confirm_booking_different_trip_gets_new_confirmation():
    reset_booking_store_for_tests()
    flight = {"flight_no": "CA1234", "date": "2026-01-10"}
    hotel1 = {"hotel_name": "上海希尔顿酒店", "checkin": "2026-01-10"}
    hotel2 = {"hotel_name": "上海如家酒店", "checkin": "2026-01-10"}

    r1 = confirm_booking(flight, hotel1, travelers=1)
    r2 = confirm_booking(flight, hotel2, travelers=1)

    assert r1["confirmation_id"] != r2["confirmation_id"]
    assert booking_call_count_for_tests() == 2
