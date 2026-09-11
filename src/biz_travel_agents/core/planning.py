"""通用任务规划（Plan-and-Execute）组件。

`supervisor.py` 用的是"每轮重新问 LLM 下一步做什么"的反应式（Reactive）
调度方式；另一种常见范式是"先生成完整计划，再按依赖关系逐步执行"
（Plan-and-Execute）。两者可以结合：先用 Planner 生成 Plan，再依据
`next_ready_tasks()` 做路由，只有遇到异常/审批打回时才重新规划。

这里给出通用、与具体业务场景无关的 Plan/Task 数据结构与调度逻辑；
`build_default_travel_plan()` 是把它应用到商旅场景的一个示例。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    description: str
    assignee: str  # 负责执行该任务的 Agent 名称
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class Plan:
    tasks: list[Task]

    def _by_id(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(f"任务不存在: {task_id}")

    def next_ready_tasks(self) -> list[Task]:
        """返回所有依赖已完成、且自身尚未开始的任务（可并行执行）。"""
        done_ids = {t.id for t in self.tasks if t.status == TaskStatus.DONE}
        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING and all(dep in done_ids for dep in t.depends_on)
        ]

    def mark(self, task_id: str, status: TaskStatus) -> None:
        self._by_id(task_id).status = status

    def is_complete(self) -> bool:
        return all(t.status == TaskStatus.DONE for t in self.tasks)

    def has_failed(self) -> bool:
        return any(t.status == TaskStatus.FAILED for t in self.tasks)


def build_default_travel_plan() -> Plan:
    """构造智能商旅场景的典型任务计划，可作为 Planner Agent 的默认输出。

    flight / hotel 都依赖 requirement 完成，彼此互不依赖，可并行执行；
    itinerary 依赖两者都完成；budget 依赖 itinerary 完成。
    """
    return Plan(
        tasks=[
            Task(id="requirement", description="解析结构化出差需求", assignee="requirement"),
            Task(id="flight", description="预订机票", assignee="flight", depends_on=["requirement"]),
            Task(id="hotel", description="预订酒店", assignee="hotel", depends_on=["requirement"]),
            Task(
                id="itinerary",
                description="生成逐日行程",
                assignee="itinerary",
                depends_on=["flight", "hotel"],
            ),
            Task(id="budget", description="预算审批", assignee="budget", depends_on=["itinerary"]),
        ]
    )
