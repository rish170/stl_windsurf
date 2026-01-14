from __future__ import annotations

import json
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from autostream.state import DATA_PATH
from model_config import load_embedding_model


def load_knowledge_texts() -> List[str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {DATA_PATH}")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    texts: List[str] = []
    pricing = raw.get("pricing", [])
    for plan in pricing:
        details = [
            f"Plan: {plan.get('plan')}",
            f"Price: {plan.get('price')}",
            f"Limits: {plan.get('limits')}",
            f"Quality: {plan.get('quality')}",
        ]
        features = plan.get("features") or []
        if features:
            details.append("Features: " + ", ".join(features))
        texts.append(" | ".join(details))
    policies = raw.get("policies", [])
    if policies:
        texts.append("Policies: " + "; ".join(policies))
    product = raw.get("product", [])
    if product:
        texts.append("Product: " + " | ".join(product))
    return texts


def build_retriever():
    embeddings: GoogleGenerativeAIEmbeddings = load_embedding_model()
    texts = load_knowledge_texts()
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    return vector_store.as_retriever(k=3)
