"""测试全局夹具。"""
import pytest

from biz_travel_agents.config import LLM_CIRCUIT_BREAKER


@pytest.fixture(autouse=True)
def _reset_llm_circuit_breaker():
    """LLM_CIRCUIT_BREAKER 是跨 Agent 共享的单例，测试之间必须互不影响。"""
    LLM_CIRCUIT_BREAKER.reset()
    yield
    LLM_CIRCUIT_BREAKER.reset()
