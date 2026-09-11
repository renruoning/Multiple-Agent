"""通用上下文窗口管理：控制喂给 LLM 的历史消息规模，避免超出上下文长度。

多智能体系统运行轮次越多，执行轨迹（messages）越长；如果每次都把全部历史
原样丢给 LLM，会导致成本上升、延迟增加，甚至超出模型的上下文窗口。
这里提供一个与具体 LLM 提供方无关的裁剪/压缩策略。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

Message = tuple[str, str]  # (role, content)


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（不引入 tiktoken 等额外依赖）。

    中英混合文本经验上 1 token 约等于 2~4 个字符，这里取 3 作为折中，
    仅用于预算控制，不要求精确。
    """
    return max(1, len(text) // 3)


@dataclass
class ContextManager:
    max_tokens: int = 4000
    keep_recent: int = 6
    summarizer: Callable[[Sequence[Message]], str] | None = None

    def compact(self, messages: Sequence[Message]) -> list[Message]:
        """把消息历史压缩到 token 预算内。

        策略：始终保留最近 `keep_recent` 条消息；更早的消息统一替换成一条
        摘要消息（由 `summarizer` 生成，未提供时退化为占位摘要），
        避免历史无限增长后被原样塞进每一次 LLM 调用。
        """
        messages = list(messages)
        if len(messages) <= self.keep_recent:
            return messages

        recent = messages[-self.keep_recent :] if self.keep_recent > 0 else []
        older = messages[: len(messages) - len(recent)]

        recent_tokens = sum(estimate_tokens(content) for _, content in recent)
        if recent_tokens >= self.max_tokens:
            # 即使只算最近消息就已经超预算了，这里也只保留最近的，
            # 交给上层的真实 LLM 调用自行做进一步截断。
            return recent

        if self.summarizer is not None:
            summary_text = self.summarizer(older)
        else:
            summary_text = f"（已省略 {len(older)} 条更早的历史记录）"

        return [("system", summary_text), *recent]
