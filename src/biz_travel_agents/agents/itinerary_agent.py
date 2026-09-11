"""行程规划Agent：整合机票与酒店信息，生成完整的日程安排。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import build_chat_model
from ..state import TravelState

SYSTEM_PROMPT = """你是企业智能商旅系统中的「行程规划Agent」。
请根据已确定的航班和酒店信息，生成一份逐日行程安排（包含交通、入住/退房、
预留的工作/会议时间段），语言简洁、可直接发给出差员工使用。
"""


class DayPlan(BaseModel):
    date: str = Field(description="日期 YYYY-MM-DD")
    schedule: str = Field(description="当天的行程安排，一到两句话")


class ItineraryPlan(BaseModel):
    days: list[DayPlan] = Field(description="按日期顺序排列的逐日行程")
    notes: str = Field(description="行程备注，例如提醒事项")


def itinerary_node(state: TravelState) -> dict:
    flight = state.get("selected_flight")
    hotel = state.get("selected_hotel")
    req = state["requirements"]

    llm = build_chat_model()
    structured_llm = llm.with_structured_output(ItineraryPlan)
    human_prompt = (
        f"出差需求: {req}\n已选航班: {flight}\n已选酒店: {hotel}"
    )
    plan: ItineraryPlan = structured_llm.invoke(
        [("system", SYSTEM_PROMPT), ("human", human_prompt)]
    )
    itinerary = plan.model_dump()

    summary = f"[行程规划Agent] 已生成 {len(itinerary['days'])} 天的行程安排。"
    return {
        "itinerary": itinerary,
        "messages": [("ai", summary)],
        "iteration": state.get("iteration", 0) + 1,
    }
