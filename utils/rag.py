"""
rag.py – Lightweight RAG pipeline for AutoStream's knowledge base.

Uses TF-IDF similarity to find the most relevant KB chunk, then
returns it as context for the LLM.  No vector DB dependency needed.
"""

import json
import math
import re
from pathlib import Path
from typing import List, Tuple


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


# ─────────────────────────────────────────────
# Load & chunk knowledge base
# ─────────────────────────────────────────────

def _load_kb() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)


def _build_chunks(kb: dict) -> List[str]:
    """Convert the JSON KB into flat text chunks for retrieval."""
    chunks = []

    # Company overview
    c = kb["company"]
    chunks.append(f"Company: {c['name']}. {c['description']}")

    # Pricing plans
    for plan in kb["pricing_plans"]:
        features = "; ".join(plan["features"])
        chunks.append(
            f"{plan['name']}: ${plan['price_monthly']}/month. "
            f"Features: {features}. Best for: {plan['best_for']}."
        )

    # Policies
    for p in kb["policies"]:
        chunks.append(f"{p['policy']}: {p['details']}")

    # FAQs
    for faq in kb["faqs"]:
        chunks.append(f"Q: {faq['question']} A: {faq['answer']}")

    return chunks


# ─────────────────────────────────────────────
# TF-IDF helpers
# ─────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf(tokens: List[str]) -> dict:
    counts: dict = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _idf(term: str, corpus: List[List[str]]) -> float:
    df = sum(1 for doc in corpus if term in doc)
    return math.log((len(corpus) + 1) / (df + 1)) + 1


def _tfidf_score(query_tokens: List[str], doc_tokens: List[str], corpus: List[List[str]]) -> float:
    tf = _tf(doc_tokens)
    score = 0.0
    for t in set(query_tokens):
        score += tf.get(t, 0) * _idf(t, corpus)
    return score


# ─────────────────────────────────────────────
# Public retrieval function
# ─────────────────────────────────────────────

_kb = _load_kb()
_chunks = _build_chunks(_kb)
_tokenized_chunks = [_tokenize(c) for c in _chunks]


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Return the top-k most relevant KB chunks as a single context string.
    """
    query_tokens = _tokenize(query)
    scored: List[Tuple[float, str]] = []

    for chunk, tokens in zip(_chunks, _tokenized_chunks):
        score = _tfidf_score(query_tokens, tokens, _tokenized_chunks)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k]]
    return "\n\n".join(top_chunks)


def get_full_context() -> str:
    """Return the entire knowledge base as a formatted string (for system prompt)."""
    return "\n\n".join(_chunks)
