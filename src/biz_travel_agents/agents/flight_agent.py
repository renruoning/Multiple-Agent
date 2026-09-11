"""机票预订Agent：查询候选航班并结合需求/预算选出最合适的一个。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import build_chat_model
from ..state import TravelState
from ..tools.flight_tools import search_flights

SYSTEM_PROMPT = """你是企业智能商旅系统中的「机票预订Agent」。
请根据出差需求和候选航班列表，选择最合适的一个航班：
- 优先满足预算约束
- 其次考虑时间是否合理、航司口碑
- 若收到「预算审批未通过，需要重新选择更经济方案」的提示，必须选择候选中价格更低的选项。
"""


class FlightChoice(BaseModel):
    flight_no: str = Field(description="选择的航班号，必须来自候选列表")
    reason: str = Field(description="选择理由，一句话")


def flight_node(state: TravelState) -> dict:
    req = state["requirements"]
    options = search_flights(
        origin=req["origin"],
        destination=req["destination"],
        date=req["depart_date"],
        cabin=req.get("cabin_preference", "economy"),
    )

    budget_check = state.get("budget_check")
    retry_hint = ""
    if budget_check and not budget_check.get("approved", True):
        retry_hint = (
            "\n\n注意：上一轮预算审批未通过，违规原因："
            f"{budget_check.get('violations')}。请重新选择更经济的方案。"
        )

    llm = build_chat_model()
    structured_llm = llm.with_structured_output(FlightChoice)
    human_prompt = f"出差需求: {req}\n候选航班: {options}{retry_hint}"
    choice: FlightChoice = structured_llm.invoke(
        [("system", SYSTEM_PROMPT), ("human", human_prompt)]
    )

    selected = next(
        (o for o in options if o["flight_no"] == choice.flight_no), options[0]
    )
    summary = (
        f"[机票预订Agent] 已选择航班 {selected['flight_no']}（{selected['airline']}，"
        f"{selected['depart_time']} 出发），票价 {selected['price']} 元。理由：{choice.reason}"
    )
    return {
        "flight_options": options,
        "selected_flight": selected,
        "messages": [("ai", summary)],
        "iteration": state.get("iteration", 0) + 1,
    }
