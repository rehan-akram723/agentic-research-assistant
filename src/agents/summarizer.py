"""Summarizer agent for Phase 3.

The summarizer produces one structured summary per retrieved document. An
OpenAI-compatible chat model is used when available, while a deterministic
fallback keeps the pipeline runnable without external model access.
"""
from __future__ import annotations

import json
import re
from typing import List

from pydantic import BaseModel, Field

from src.core.exceptions import AgentError
from src.core.logger import get_logger
from src.core.llm import call_openrouter_completion
from src.core.metrics import track_latency
from src.graph.state import ResearchState
from src.schemas.document import Document

logger = get_logger(__name__)

class SummaryPayload(BaseModel):
    summary: str = Field(default="")
    key_points: List[str] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)


def _coerce_document(item: object) -> Document:
    if isinstance(item, Document):
        return item

    if isinstance(item, dict):
        if hasattr(Document, "model_validate"):
            return Document.model_validate(item)  # type: ignore[attr-defined]
        return Document(**item)

    raise AgentError("summarizer_agent received an invalid retrieved document.")


def _split_sentences(text: str) -> List[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    return sentences


def _fallback_payload(document: Document) -> SummaryPayload:
    abstract = " ".join((document.abstract or "").split())
    sentences = _split_sentences(abstract)
    summary_text = " ".join(sentences[:2]).strip() or abstract[:400].strip() or "No abstract available."
    key_points = [summary_text[:240]]
    if document.published_date:
        key_points.append(f"Published: {document.published_date}")
    claims = [sentence[:240] for sentence in sentences[:3]] if sentences else [summary_text[:240]]
    return SummaryPayload(summary=summary_text, key_points=key_points, claims=claims)


def _parse_payload(payload: str) -> SummaryPayload:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Summary payload must be a JSON object")
    return SummaryPayload.model_validate(data)


def _call_model(document: Document) -> SummaryPayload:
    prompt = (
        "You are summarizing a research paper for downstream citation verification. "
        "Return JSON only with keys summary, key_points, and claims. The summary should be 2 to 4 sentences. "
        "Key points should contain 2 to 4 concise bullets, and claims should contain 1 to 3 factual claims "
        "drawn only from the supplied title, authors, abstract, and metadata. "
        "Your entire response must be one valid JSON object and nothing else.\n\n"
        f"Document ID: {document.id}\n"
        f"Title: {document.title}\n"
        f"Authors: {', '.join(document.authors)}\n"
        f"Abstract: {document.abstract}\n"
        f"Published: {document.published_date or 'Unknown'}\n"
        f"Source: {document.source}\n"
    )

    content = call_openrouter_completion(prompt, temperature=0.2)
    return _parse_payload(content)


def _format_summary(document: Document, payload: SummaryPayload) -> str:
    lines = [
        f"Document ID: {document.id}",
        f"Title: {document.title}",
        f"Authors: {', '.join(document.authors) if document.authors else 'Unknown authors'}",
        f"Source: {document.source}",
        f"Published: {document.published_date or 'Unknown'}",
        f"Summary: {payload.summary.strip()}",
        "Key Points:",
    ]
    lines.extend(f"- {point.strip()}" for point in payload.key_points if point.strip())
    lines.append("Claims:")
    lines.extend(f"- {claim.strip()}" for claim in payload.claims if claim.strip())
    if document.url:
        lines.append(f"URL: {document.url}")
    return "\n".join(lines)


@track_latency("summarizer")
def summarizer_agent(state: ResearchState) -> dict:
    try:
        documents = [_coerce_document(item) for item in (state.get("retrieved_docs") or [])]
        summaries: List[str] = []

        for document in documents:
            try:
                payload = _call_model(document)
            except Exception as exc:
                logger.warning("summarizer_agent falling back to deterministic summary: %s", exc)
                payload = _fallback_payload(document)
            summaries.append(_format_summary(document, payload))

        step_log = [f"summarizer: produced {len(summaries)} summaries"]
        logger.info("summarizer_agent produced %s summaries", len(summaries))
        return {"summaries": summaries, "step_log": step_log}
    except AgentError:
        raise
    except Exception as exc:
        raise AgentError(f"Summarizer agent failed: {exc}") from exc
