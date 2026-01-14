from __future__ import annotations

from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from autostream.llm import CHAT_MODEL
from autostream.state import AgentState, CONFUSION_KEYWORDS

intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intent classifier for the AutoStream sales assistant. "
            "Classify the latest user message into one of: greeting, inquiry, high_intent, other. "
            "High intent means the user wants to sign up, try, buy, or is ready to proceed. "
            "Respond with only the intent word in lowercase.",
        ),
        ("human", "{input}"),
    ]
)
intent_chain = intent_prompt | CHAT_MODEL | StrOutputParser()


def detect_plan_choice(text: str) -> str:
    lower = text.lower()
    if "basic" in lower:
        return "basic"
    if "pro" in lower:
        return "pro"
    return ""


def classify_intent(state: AgentState) -> Dict[str, str]:
    last_user = state["messages"][-1].content.lower() if state.get("messages") else ""
    plan_choice = detect_plan_choice(last_user) or state.get("plan_choice", "")

    if any(keyword in last_user for keyword in CONFUSION_KEYWORDS):
        return {"intent": "inquiry", "plan_choice": plan_choice}

    if state.get("intent") == "high_intent" and not state.get("lead_captured"):
        return {"intent": "high_intent", "plan_choice": plan_choice}

    intent = intent_chain.invoke({"input": last_user}).strip().lower()
    if intent not in {"greeting", "inquiry", "high_intent", "other"}:
        intent = "other"
    return {"intent": intent, "plan_choice": plan_choice}
