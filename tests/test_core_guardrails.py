"""通用护栏（Guardrails）的单元测试。"""
from biz_travel_agents.core.guardrails import check, date_order, field_positive


def test_check_returns_no_violations_for_valid_data():
    data = {"budget": 1000, "depart_date": "2026-01-01", "return_date": "2026-01-03"}
    violations = check(data, [field_positive("budget"), date_order("depart_date", "return_date")])
    assert violations == []


def test_check_flags_non_positive_budget():
    violations = check({"budget": -1}, [field_positive("budget")])
    assert len(violations) == 1
    assert "budget" in violations[0]


def test_check_flags_reversed_dates():
    data = {"depart_date": "2026-01-05", "return_date": "2026-01-01"}
    violations = check(data, [date_order("depart_date", "return_date")])
    assert len(violations) == 1


def test_check_ignores_missing_fields_gracefully():
    violations = check({}, [field_positive("budget"), date_order("a", "b")])
    # budget 缺失也算不通过；日期缺失时无法比较，视为通过
    assert len(violations) == 1
