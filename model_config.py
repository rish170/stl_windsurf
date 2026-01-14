from __future__ import annotations

import os
from typing import Optional

import warnings

from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
# Suppress deprecation warning for langchain_community HuggingFaceEmbeddings (since stack is pinned to 0.2.x)
warnings.filterwarnings(
    "ignore",
    message=".*HuggingFaceEmbeddings was deprecated.*",
    category=LangChainDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*HuggingFaceEmbeddings was deprecated.*",
    category=DeprecationWarning,
    module="langchain_community.embeddings",
)
# Also silence generic DeprecationWarning variants
warnings.simplefilter("ignore", DeprecationWarning)

DEFAULT_CHAT_MODEL = "models/gemini-2.5-flash-lite"
# Use a local embedding model by default to avoid API quota issues; switch via ENV.
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def get_model_name() -> str:
    """Return the chat model name, allowing override via env."""
    return os.getenv("MODEL_NAME", DEFAULT_CHAT_MODEL)


def load_chat_model(**kwargs) -> ChatGoogleGenerativeAI:
    """Create a chat model instance with sensible defaults."""
    model_name = get_model_name()
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    max_retries = int(os.getenv("MODEL_MAX_RETRIES", "1"))
    request_timeout = float(os.getenv("MODEL_TIMEOUT", "60"))
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=None,
        max_retries=max_retries,
        request_timeout=request_timeout,
        **kwargs,
    )


def load_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Create an embedding model instance (Google when prefixed, else local HF)."""
    model_name = os.getenv("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL)
    if model_name.startswith("models/"):
        return GoogleGenerativeAIEmbeddings(model=model_name)
    # Local fallback avoids Gemini embed quotas. Suppress deprecation noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", LangChainDeprecationWarning)
        return HuggingFaceEmbeddings(model_name=model_name)
