"""LLM 提供方配置：通过环境变量在 Anthropic / OpenAI 之间切换。"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .core.circuit_breaker import CircuitBreaker

load_dotenv()

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
}

# 所有 Agent 共用同一个 LLM 供应方，因此共用同一个熔断器实例：
# 一旦发现该依赖持续失败，任何 Agent 的下一次调用都会直接快速失败并
# 走降级路径，而不必每个调用点各自重新摸索一遍"服务是不是挂了"。
LLM_CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float


def get_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["anthropic"])
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    return LLMConfig(provider=provider, model=model, temperature=temperature)


def build_chat_model():
    """根据配置构造一个 LangChain ChatModel 实例。

    所有 Agent 节点都通过这个函数获取模型，因此更换 LLM 提供方
    只需要改 .env，不需要改任何 Agent 代码。
    """
    cfg = get_llm_config()

    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature)

    if cfg.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg.model, temperature=cfg.temperature)

    raise ValueError(
        f"不支持的 LLM_PROVIDER: {cfg.provider!r}，请在 .env 中设置为 'anthropic' 或 'openai'"
    )
