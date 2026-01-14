from __future__ import annotations

from model_config import load_chat_model

# Single shared chat model instance used across nodes
CHAT_MODEL = load_chat_model()
