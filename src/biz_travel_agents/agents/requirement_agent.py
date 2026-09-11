"""需求理解Agent：将用户自然语言出差请求解析为结构化需求。"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from ..config import build_chat_model
from ..state import TravelState

SYSTEM_PROMPT = """你是企业智能商旅系统中的「需求理解Agent」。
你的任务是从用户的出差请求中提取结构化的商旅需求信息。
如果用户没有明确给出某些信息，请结合常识给出合理的默认值（例如：未提预算时按经济舱、
中档酒店的合理市场价估算；未提人数时默认1人）。
今天的日期是 {today}，日期字段请输出 YYYY-MM-DD 格式的绝对日期。
"""


class TripRequirement(BaseModel):
    origin: str = Field(description="出发城市")
    destination: str = Field(description="目的地城市")
    depart_date: str = Field(description="出发日期，YYYY-MM-DD")
    return_date: str = Field(description="返程日期，YYYY-MM-DD")
    travelers: int = Field(default=1, description="出行人数")
    budget: float = Field(description="预算上限（人民币元），未提及时给出合理估计")
    cabin_preference: str = Field(description="舱位偏好，取值 economy 或 business")
    purpose: str = Field(description="出差事由，简要概括")


def requirement_node(state: TravelState) -> dict:
    llm = build_chat_model()
    structured_llm = llm.with_structured_output(TripRequirement)

    result: TripRequirement = structured_llm.invoke(
        [
            ("system", SYSTEM_PROMPT.format(today=date.today().isoformat())),
            ("human", state["user_request"]),
        ]
    )
    requirements = result.model_dump()

    summary = (
        f"[需求理解Agent] 已解析出差需求：{requirements['origin']} → "
        f"{requirements['destination']}，{requirements['depart_date']} 至 "
        f"{requirements['return_date']}，{requirements['travelers']} 人，"
        f"预算 {requirements['budget']:.0f} 元，舱位偏好 {requirements['cabin_preference']}。"
    )
    return {
        "requirements": requirements,
        "messages": [("ai", summary)],
        "iteration": state.get("iteration", 0) + 1,
    }
