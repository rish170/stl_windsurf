from __future__ import annotations

from pathlib import Path
from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

DATA_PATH = Path("data/knowledge_base.json")
REQUIRED_FIELDS = ["name", "email", "platform"]
CONFUSION_KEYWORDS = ["confused", "compare", "difference", "which plan", "not sure", "decide", "vs"]
PLAN_SUMMARIES = {
    "basic": "Basic plan: $29/month, 10 videos/month, 720p resolution.",
    "pro": "Pro plan: $79/month, unlimited videos, 4K, AI captions, 24/7 support.",
}
ONBOARDING_KEYWORDS = ["onboarding", "checklist", "next steps", "setup", "get started"]
ONBOARDING_ACK = {"yes", "yes sure", "sure", "yeah", "ok", "okay", "yup"}


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    retrieved: List[str]
    lead_info: Dict[str, str]
    lead_captured: bool
    plan_choice: str  # "basic" | "pro" | ""
