from __future__ import annotations

from langgraph.graph import END, StateGraph

from autostream.state import AgentState
from autostream.agent import classify, retrieve, respond, handle_high_intent


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("respond", respond)
    graph.add_node("handle_high_intent", handle_high_intent)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")

    def route(state: AgentState):
        return "handle_high_intent" if state.get("intent") == "high_intent" else "respond"

    graph.add_conditional_edges("retrieve", route, {"handle_high_intent": "handle_high_intent", "respond": "respond"})
    graph.add_edge("respond", END)
    graph.add_edge("handle_high_intent", END)

    return graph.compile()
