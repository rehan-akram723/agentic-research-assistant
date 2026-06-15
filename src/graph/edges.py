"""Decision edges for the research pipeline graph.

Small helpers used by the graph builder to decide routing.
"""
from __future__ import annotations

from src.graph.state import ResearchState


def should_verify_citations(state: ResearchState) -> str:
    if state.get("include_citations", True):
        return "citation_verifier"
    return "report_generator"


def handle_retrieval_failure(state: ResearchState) -> str:
    retrieved = state.get("retrieved_docs") or []
    if not retrieved:
        return "end"
    return "summarizer"
