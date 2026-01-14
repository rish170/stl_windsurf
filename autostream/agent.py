"""LangGraph-based conversational agent for AutoStream (ServiceHive assignment)."""
from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from autostream.intents import classify_intent, detect_plan_choice
from autostream.llm import CHAT_MODEL
from autostream.rag import build_retriever
from autostream.state import AgentState, REQUIRED_FIELDS
from autostream.tools import (
    extract_lead_fields,
    mock_lead_capture,
    onboarding_steps,
    plan_pitch,
    wants_onboarding,
)

RETRIEVER = build_retriever()


def classify(state: AgentState):
    return classify_intent(state)


def retrieve(state: AgentState):
    last_user = state["messages"][-1].content if state.get("messages") else ""
    docs = RETRIEVER.invoke(last_user)
    contexts = [d.page_content for d in docs]
    return {"retrieved": contexts}


def respond(state: AgentState):
    context_text = "\n".join(state.get("retrieved", []))
    system = SystemMessage(
        content=(
            "You are AutoStream's assistant. Use ONLY the provided context. "
            "If the answer is not in context, say you don't have that info. "
            "Be concise and helpful.\n\n"
            f"Context:\n{context_text}"
        )
    )
    messages: List[BaseMessage] = [system] + state["messages"]
    ai_reply = CHAT_MODEL.invoke(messages)
    return {"messages": [ai_reply]}


def handle_high_intent(state: AgentState):
    last_user = state["messages"][-1].content if state.get("messages") else ""
    plan_choice = state.get("plan_choice", "") or detect_plan_choice(last_user)
    collected = dict(state.get("lead_info", {}))
    extracted = extract_lead_fields(last_user)
    for k in REQUIRED_FIELDS:
        if extracted.get(k):
            collected[k] = extracted[k]
    missing = [f for f in REQUIRED_FIELDS if not collected.get(f)]

    context_text = "\n".join(state.get("retrieved", []))
    responses: List[str] = []

    # If lead already captured and user asks/acknowledges onboarding, provide checklist directly.
    if wants_onboarding(last_user, state.get("lead_captured", False)):
        checklist = onboarding_steps(plan_choice)
        return {
            "messages": [AIMessage(content=checklist)],
            "lead_info": collected,
            "lead_captured": True,
            "plan_choice": plan_choice,
        }

    if missing:
        pitch = plan_pitch(plan_choice)
        ask_parts = []
        if "name" in missing:
            ask_parts.append("your name")
        if "email" in missing:
            ask_parts.append("a work email")
        if "platform" in missing:
            ask_parts.append("your creator platform (e.g., YouTube, Instagram)")
        ask_text = "Could you share " + " and ".join(ask_parts) + " to set you up?"
        responses.append(f"Great! I can help you start with AutoStream. {pitch}")
        if context_text:
            responses.append(f"Context:\n{context_text}")
        responses.append(ask_text)
        reply = "\n".join(responses)
        return {
            "messages": [AIMessage(content=reply)],
            "lead_info": collected,
            "lead_captured": False,
            "plan_choice": plan_choice,
        }

    # All fields present and not yet captured
    if not state.get("lead_captured"):
        mock_lead_capture(
            collected.get("name", ""), collected.get("email", ""), collected.get("platform", "")
        )
        pitch = plan_pitch(plan_choice)
        reply = (
            f"Awesome, you're set for the {plan_choice or 'selected'} plan! "
            "I've captured your details and will send next steps to your email. "
            f"{pitch} Want a quick onboarding checklist?"
        )
        return {
            "messages": [AIMessage(content=reply)],
            "lead_info": collected,
            "lead_captured": True,
            "retrieved": state.get("retrieved", []),
            "plan_choice": plan_choice,
        }

    # Already captured, continue helpful chat
    follow_up = "All set! Anything else you'd like to know about AutoStream?"
    return {
        "messages": [AIMessage(content=follow_up)],
        "lead_info": collected,
        "lead_captured": True,
        "plan_choice": plan_choice,
    }
