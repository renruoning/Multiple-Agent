"""模拟外部系统工具的单元测试（不涉及 LLM，无需 API Key）。"""
from biz_travel_agents.tools.flight_tools import search_flights
from biz_travel_agents.tools.hotel_tools import search_hotels
from biz_travel_agents.tools.policy_tools import check_policy


def test_search_flights_sorted_by_price_and_deterministic():
    a = search_flights("北京", "上海", "2026-01-10", cabin="economy")
    b = search_flights("北京", "上海", "2026-01-10", cabin="economy")

    assert a == b  # 相同输入应产生相同结果（伪随机可复现）
    prices = [o["price"] for o in a]
    assert prices == sorted(prices)
    assert all(o["cabin"] == "economy" for o in a)


def test_search_flights_business_is_more_expensive():
    economy = search_flights("北京", "上海", "2026-01-10", cabin="economy")
    business = search_flights("北京", "上海", "2026-01-10", cabin="business")
    assert business[0]["price"] > economy[0]["price"]


def test_search_hotels_computes_total_price_from_nights():
    options = search_hotels("上海", "2026-01-10", "2026-01-12")
    for o in options:
        assert o["nights"] == 2
        assert o["total_price"] == o["price_per_night"] * o["nights"]


def test_check_policy_approves_within_limits():
    result = check_policy(
        total_cost=3000, cabin="economy", hotel_price_per_night=500, budget_limit=5000
    )
    assert result["approved"] is True
    assert result["violations"] == []


def test_check_policy_rejects_over_budget():
    result = check_policy(
        total_cost=8000, cabin="economy", hotel_price_per_night=500, budget_limit=5000
    )
    assert result["approved"] is False
    assert any("预算" in v for v in result["violations"])


def test_check_policy_rejects_disallowed_cabin():
    result = check_policy(
        total_cost=1000, cabin="business", hotel_price_per_night=500, budget_limit=5000
    )
    assert result["approved"] is False
    assert any("舱位" in v for v in result["violations"])
