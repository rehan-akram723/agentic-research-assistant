"""Planner agent: deterministic query generator for Phase 1.

This lightweight implementation creates 3 short search queries derived from
the user's question. It's deliberately deterministic and avoids making
external LLM calls so the codebase remains runnable in CI.
"""
from __future__ import annotations

from typing import List
from src.graph.state import ResearchState


def planner_agent(state: ResearchState) -> dict:
    question = state.get("question", "")
    base = question.strip()
    queries: List[str] = []
    if not base:
        queries = []
    else:
        # Simple heuristics to produce 3 queries
        queries = [
            base,
            f"{base} review",
            f"{base} methods results",
        ]

    step_log = [f"planner: generated {len(queries)} queries"]
    return {"search_queries": queries, "step_log": step_log}
