"""Summarizer agent: lightweight deterministic summaries for Phase 1.

Extracts the key points from each Document's title and abstract and
produces a short structured summary string for downstream agents.
"""
from __future__ import annotations

from typing import List
from src.graph.state import ResearchState
from src.schemas.document import Document


def _summarize_doc(doc: Document) -> str:
    parts = [f"Title: {doc.title}", f"Authors: {', '.join(doc.authors)}"]
    if doc.abstract:
        # take the first 2 sentences as a naive summary
        abstract = doc.abstract.strip()
        sentences = abstract.split('.')
        summary_text = '.'.join([s.strip() for s in sentences[:2] if s.strip()])
        parts.append(f"Abstract Summary: {summary_text}.")
    return "\n".join(parts)


def summarizer_agent(state: ResearchState) -> dict:
    docs: List[Document] = state.get("retrieved_docs", []) or []
    summaries: List[str] = []
    for d in docs:
        if isinstance(d, dict):
            d = Document.model_validate(d) if hasattr(Document, "model_validate") else Document(**d)
        summaries.append(_summarize_doc(d))

    step_log = [f"summarizer: produced {len(summaries)} summaries"]
    return {"summaries": summaries, "step_log": step_log}
