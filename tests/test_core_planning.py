"""Plan/Task（Plan-and-Execute）的单元测试。"""
from biz_travel_agents.core.planning import Plan, Task, TaskStatus, build_default_travel_plan


def test_next_ready_tasks_respects_dependencies():
    plan = build_default_travel_plan()
    ready = {t.id for t in plan.next_ready_tasks()}
    assert ready == {"requirement"}

    plan.mark("requirement", TaskStatus.DONE)
    ready = {t.id for t in plan.next_ready_tasks()}
    assert ready == {"flight", "hotel"}  # 二者互不依赖，可并行


def test_plan_completion_flow():
    plan = build_default_travel_plan()
    for task_id in ["requirement", "flight", "hotel", "itinerary", "budget"]:
        assert not plan.is_complete()
        plan.mark(task_id, TaskStatus.DONE)
    assert plan.is_complete()


def test_has_failed():
    plan = Plan(tasks=[Task(id="a", description="d", assignee="x")])
    assert not plan.has_failed()
    plan.mark("a", TaskStatus.FAILED)
    assert plan.has_failed()


def test_unknown_task_id_raises_key_error():
    plan = Plan(tasks=[Task(id="a", description="d", assignee="x")])
    try:
        plan.mark("missing", TaskStatus.DONE)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
