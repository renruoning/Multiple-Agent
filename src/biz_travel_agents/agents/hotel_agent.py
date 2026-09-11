"""酒店预订Agent：查询候选酒店并结合需求/预算选出最合适的一个。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import build_chat_model
from ..state import TravelState
from ..tools.hotel_tools import search_hotels

SYSTEM_PROMPT = """你是企业智能商旅系统中的「酒店预订Agent」。
请根据出差需求和候选酒店列表，选择最合适的一个酒店：
- 优先满足预算约束
- 其次考虑星级、位置合理性
- 若收到「预算审批未通过，需要重新选择更经济方案」的提示，必须选择候选中单价更低的选项。
"""


class HotelChoice(BaseModel):
    hotel_name: str = Field(description="选择的酒店名称，必须来自候选列表")
    reason: str = Field(description="选择理由，一句话")


def hotel_node(state: TravelState) -> dict:
    req = state["requirements"]
    options = search_hotels(
        city=req["destination"],
        checkin=req["depart_date"],
        checkout=req["return_date"],
    )

    budget_check = state.get("budget_check")
    retry_hint = ""
    if budget_check and not budget_check.get("approved", True):
        retry_hint = (
            "\n\n注意：上一轮预算审批未通过，违规原因："
            f"{budget_check.get('violations')}。请重新选择更经济的方案。"
        )

    llm = build_chat_model()
    structured_llm = llm.with_structured_output(HotelChoice)
    human_prompt = f"出差需求: {req}\n候选酒店: {options}{retry_hint}"
    choice: HotelChoice = structured_llm.invoke(
        [("system", SYSTEM_PROMPT), ("human", human_prompt)]
    )

    selected = next(
        (o for o in options if o["hotel_name"] == choice.hotel_name), options[0]
    )
    summary = (
        f"[酒店预订Agent] 已选择酒店 {selected['hotel_name']}（{selected['star_rating']}星），"
        f"单价 {selected['price_per_night']} 元/晚，共 {selected['nights']} 晚。理由：{choice.reason}"
    )
    return {
        "hotel_options": options,
        "selected_hotel": selected,
        "messages": [("ai", summary)],
        "iteration": state.get("iteration", 0) + 1,
    }
