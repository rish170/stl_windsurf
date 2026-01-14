from __future__ import annotations

import json
import re
from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from autostream.llm import CHAT_MODEL
from autostream.state import ONBOARDING_ACK, ONBOARDING_KEYWORDS, PLAN_SUMMARIES, REQUIRED_FIELDS

lead_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract lead details from the latest user message. "
            "Return a JSON object with keys name, email, platform. Use empty string if not provided. "
            "Platform examples: YouTube, Instagram, TikTok, podcast, etc.",
        ),
        ("human", "{input}"),
    ]
)
lead_chain = lead_prompt | CHAT_MODEL | StrOutputParser()


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API call to capture a lead."""
    msg = f"Lead captured successfully: {name}, {email}, {platform}"
    print(msg)
    return msg


def extract_lead_fields(text: str) -> Dict[str, str]:
    try:
        raw = lead_chain.invoke({"input": text})
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return {k: (v or "").strip() for k, v in parsed.items() if k in REQUIRED_FIELDS}
    except Exception:
        pass
    return {k: "" for k in REQUIRED_FIELDS}


def plan_pitch(plan: str) -> str:
    if plan in PLAN_SUMMARIES:
        return PLAN_SUMMARIES[plan]
    return "We have two plans: Basic ($29/mo, 10 videos, 720p) and Pro ($79/mo, unlimited, 4K, AI captions, 24/7 support)."


def onboarding_steps(plan: str) -> str:
    plan_label = "Pro" if plan == "pro" else "Basic" if plan == "basic" else "your"
    return (
        f"Here’s a quick {plan_label} onboarding checklist:\n"
        "1) Connect your storage (Drive/Dropbox) and import a sample video.\n"
        "2) Pick an editing template (cuts + captions) and set aspect ratio for your platform.\n"
        "3) Enable AI captions and audio leveling; review the preview.\n"
        "4) Export and publish to your creator platform (YouTube/Instagram/Twitch).\n"
        "5) Turn on 24/7 support (Pro only) if you need live help.\n"
    )


def wants_onboarding(text: str, lead_captured: bool) -> bool:
    if not lead_captured:
        return False
    lower = text.lower().strip()
    return any(k in lower for k in ONBOARDING_KEYWORDS) or lower in ONBOARDING_ACK
