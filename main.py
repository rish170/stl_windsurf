from __future__ import annotations

from langchain_core.messages import HumanMessage

from autostream.state import AgentState
from autostream.graph import build_graph


def run_cli():
    graph = build_graph()
    state: AgentState = {
        "messages": [],
        "intent": "",
        "retrieved": [],
        "lead_info": {},
        "lead_captured": False,
        "plan_choice": "",
    }

    print("AutoStream Assistant (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state)
        intent = state.get("intent", "")
        ai_msg = state["messages"][-1].content if state.get("messages") else ""
        intent_str = f"[intent: {intent}]" if intent else ""
        print(f"{intent_str}")
        print(f"Assistant: {ai_msg}\n")


if __name__ == "__main__":
    run_cli()
