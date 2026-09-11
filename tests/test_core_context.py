"""ContextManager 的单元测试（纯 Python，无需 LLM）。"""
from biz_travel_agents.core.context import ContextManager, estimate_tokens


def test_compact_keeps_all_when_within_keep_recent():
    cm = ContextManager(keep_recent=5)
    messages = [("ai", f"msg{i}") for i in range(3)]
    assert cm.compact(messages) == messages


def test_compact_summarizes_older_messages():
    cm = ContextManager(keep_recent=2)
    messages = [("ai", f"msg{i}") for i in range(5)]
    result = cm.compact(messages)

    assert len(result) == 3  # 1 条摘要 + 最近 2 条
    assert result[0][0] == "system"
    assert "3" in result[0][1]  # 省略了 3 条
    assert result[1:] == messages[-2:]


def test_compact_uses_custom_summarizer():
    cm = ContextManager(keep_recent=1, summarizer=lambda older: f"summary-of-{len(older)}")
    messages = [("ai", "a"), ("ai", "b"), ("ai", "c")]
    result = cm.compact(messages)
    assert result[0] == ("system", "summary-of-2")


def test_estimate_tokens_is_positive_and_monotonic():
    assert estimate_tokens("hi") > 0
    assert estimate_tokens("a" * 300) > estimate_tokens("a" * 3)
