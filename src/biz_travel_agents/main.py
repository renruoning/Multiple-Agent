"""命令行入口：输入一句自然语言出差需求，跑完整个多智能体流程。

用法：
    python -m biz_travel_agents.main "帮我订下周一从北京到上海的机票，住两晚，预算4000元"

未提供参数时会进入交互式输入模式。
"""
from __future__ import annotations

import sys

from .core.tracing import Tracer
from .graph import build_graph


def run(user_request: str) -> None:
    tracer = Tracer()
    app = build_graph(tracer=tracer)
    initial_state = {
        "user_request": user_request,
        "iteration": 0,
    }

    print(f"\n收到出差请求：{user_request}\n")
    print("=" * 60)

    final_state = None
    for step in app.stream(initial_state, config={"recursion_limit": 60}):
        for node_name, node_output in step.items():
            messages = node_output.get("messages") or []
            for msg in messages:
                # messages 中的元素形如 ("ai", "文本内容")
                content = msg[1] if isinstance(msg, tuple) else getattr(msg, "content", msg)
                print(f"[{node_name}] {content}")
            final_state = node_output

    print("=" * 60)
    if final_state and final_state.get("final_report"):
        print("\n最终出差方案：\n")
        print(final_state["final_report"])

    print("\n" + "=" * 60)
    print("[Trace] 各节点执行耗时（见 core/tracing.py）：")
    for event in tracer.events:
        status = "OK" if event.status == "success" else f"ERROR: {event.error}"
        print(f"  - {event.node}: {event.duration_ms:.1f}ms [{status}]")


def main() -> None:
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
    else:
        user_request = input("请输入你的出差需求：")
    run(user_request)


if __name__ == "__main__":
    main()
