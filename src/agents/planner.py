"""Planner agent for Phase 3.

This agent turns the incoming research question into a small set of academic
search queries. It uses Gemini when configured, and falls back to a local
heuristic when the model is unavailable or the response cannot be parsed.
"""
from __future__ import annotations

import json
import re
from typing import List

from src.core.config import settings
from src.core.exceptions import AgentError
from src.core.logger import get_logger
from src.graph.state import ResearchState

logger = get_logger(__name__)

try:  # pragma: no cover - import depends on environment
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None


def _fallback_queries(question: str) -> List[str]:
    normalized = " ".join(question.strip().split())
    if not normalized:
        return []

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", normalized)
    focus = " ".join(tokens[:10]) if tokens else normalized
    queries = [
        focus,
        f"{focus} survey",
        f"{focus} methods",
        f"{focus} evaluation",
    ]

    deduped: List[str] = []
    for query in queries:
        candidate = " ".join(query.split())
        if candidate and candidate.lower() not in {item.lower() for item in deduped}:
            deduped.append(candidate)
    return deduped[:5]


def _parse_queries(payload: str) -> List[str]:
    text = payload.strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            raw_queries = data.get("search_queries", [])
        else:
            raw_queries = data
        if not isinstance(raw_queries, list):
            return []
        queries = [" ".join(str(item).split()) for item in raw_queries if str(item).strip()]
        return queries[:5]
    except Exception:
        lines = [line.strip("- •\t ") for line in text.splitlines()]
        queries = [line for line in lines if line]
        return queries[:5]


def _call_gemini(question: str) -> List[str]:
    if ChatGoogleGenerativeAI is None:
        raise AgentError("langchain-google-genai is not installed.")
    if not settings.gemini_api_key:
        raise AgentError("Gemini API key is not configured.")

    prompt = (
        "You are an academic research planner. Generate 3 to 5 concise search queries "
        "for scholarly search engines. Return JSON only with a top-level key named "
        '"search_queries". Each query should be distinct, short, and directly tied to '
        "the question. Avoid duplicates and avoid complete sentences.\n\n"
        f"Research question: {question}"
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    response = model.invoke(prompt)
    content = getattr(response, "content", response)
    queries = _parse_queries(str(content))
    if not queries:
        raise AgentError("Gemini returned no usable search queries.")
    return queries


def planner_agent(state: ResearchState) -> dict:
    try:
        question = str(state.get("question", "") or "").strip()
        if not question:
            raise AgentError("planner_agent requires a non-empty research question.")

        try:
            queries = _call_gemini(question)
        except AgentError as exc:
            logger.warning("planner_agent Gemini planning failed; using fallback queries: %s", exc)
            queries = _fallback_queries(question)

        if not queries:
            raise AgentError("planner_agent could not derive any search queries.")

        step_log = [f"planner: generated {len(queries)} queries"]
        logger.info("planner_agent generated %s queries", len(queries))
        return {"search_queries": queries, "step_log": step_log}
    except AgentError:
        raise
    except Exception as exc:
        raise AgentError(f"Planner agent failed: {exc}") from exc
