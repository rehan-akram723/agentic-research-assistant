"""Citation verifier agent for Phase 3.

The verifier checks the generated claims against source documents using FAISS
similarity and, when configured, Gemini-assisted judgment. It emits state-only
patches containing `Citation` objects.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.exceptions import AgentError, CitationError
from src.core.logger import get_logger
from src.graph.state import ResearchState
from src.schemas.document import Citation, Document
from src.tools.faiss_memory_tool import faiss_memory

logger = get_logger(__name__)

try:  # pragma: no cover - import depends on environment
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None


class VerificationPayload(BaseModel):
    is_verified: bool = Field(default=False)
    quote: str | None = Field(default=None)


def _coerce_document(item: object) -> Document:
    if isinstance(item, Document):
        return item

    if isinstance(item, dict):
        if hasattr(Document, "model_validate"):
            return Document.model_validate(item)  # type: ignore[attr-defined]
        return Document(**item)

    raise CitationError("citation_verifier_agent received an invalid retrieved document.")


def _build_llm() -> object | None:
    if ChatGoogleGenerativeAI is None or not settings.gemini_api_key:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )


def _document_lookup(documents: List[Document]) -> Dict[str, Document]:
    lookup: Dict[str, Document] = {}
    for document in documents:
        if document.id:
            lookup[document.id.strip().lower()] = document
        if document.title:
            lookup[document.title.strip().lower()] = document
    return lookup


def _parse_summary(summary: str) -> tuple[str | None, List[str]]:
    document_id: str | None = None
    claims: List[str] = []
    in_claims = False

    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("document id:"):
            document_id = line.split(":", 1)[1].strip() or None
            continue
        if line.lower().startswith("claims:"):
            in_claims = True
            continue
        if line.lower().startswith(("summary:", "title:", "authors:", "source:", "published:", "key points:", "url:")):
            if line.lower().startswith("key points:"):
                in_claims = False
            continue
        if in_claims and line.startswith("-"):
            claim = line.lstrip("- ").strip()
            if claim:
                claims.append(claim)

    if not claims:
        summary_line = next((line.split(":", 1)[1].strip() for line in summary.splitlines() if line.lower().startswith("summary:") and ":" in line), "")
        if summary_line:
            claims = [summary_line]

    return document_id, claims


def _fallback_verification(claim: str, document: Document) -> VerificationPayload:
    text = f"{document.title}\n{document.abstract}".lower()
    claim_tokens = [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", claim.lower()) if len(token) > 3]
    overlap = sum(1 for token in claim_tokens if token in text)
    verified = overlap >= max(2, min(5, len(claim_tokens))) or claim[:80].lower() in text
    quote = document.abstract[:280] if verified and document.abstract else None
    return VerificationPayload(is_verified=verified, quote=quote)


def _parse_payload(payload: str) -> VerificationPayload:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Verification payload must be a JSON object")
    return VerificationPayload.model_validate(data)


def _call_gemini(claim: str, document: Document) -> VerificationPayload:
    if ChatGoogleGenerativeAI is None:
        raise AgentError("langchain-google-genai is not installed.")
    if not settings.gemini_api_key:
        raise AgentError("Gemini API key is not configured.")

    prompt = (
        "You are verifying whether a claim is explicitly supported by a research paper. "
        "Return JSON only with keys is_verified and quote. The quote should be the shortest exact supporting "
        "span available from the abstract or metadata, or null if the claim is not supported.\n\n"
        f"Claim: {claim}\n"
        f"Document title: {document.title}\n"
        f"Document authors: {', '.join(document.authors)}\n"
        f"Document abstract: {document.abstract}\n"
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    response = model.invoke(prompt)
    content = getattr(response, "content", response)
    return _parse_payload(str(content))


def citation_verifier_agent(state: ResearchState) -> dict:
    try:
        summaries = [str(summary).strip() for summary in (state.get("summaries") or []) if str(summary).strip()]
        documents = [_coerce_document(item) for item in (state.get("retrieved_docs") or [])]
        if not summaries:
            raise CitationError("No summaries available for citation verification.")
        if not documents:
            raise CitationError("No retrieved documents available for citation verification.")

        lookup = _document_lookup(documents)
        llm = _build_llm()
        verified: List[Citation] = []

        for summary in summaries:
            document_id, claims = _parse_summary(summary)
            document = lookup.get(document_id.lower()) if document_id else None
            if document is None:
                try:
                    hits = faiss_memory.similarity_search(summary, k=1)
                    document = hits[0] if hits else None
                except Exception as exc:
                    logger.warning("citation_verifier_agent similarity search failed: %s", exc)
                    document = None

            if document is None:
                for claim in claims or [summary[:240]]:
                    verified.append(
                        Citation(
                            claim=claim[:300],
                            source_doc_id="",
                            source_title="",
                            is_verified=False,
                            quote=None,
                        )
                    )
                continue

            for claim in claims or [summary[:240]]:
                try:
                    if llm is not None:
                        payload = _call_gemini(claim, document)
                    else:
                        payload = _fallback_verification(claim, document)
                except Exception as exc:
                    logger.warning("citation_verifier_agent falling back to deterministic verification: %s", exc)
                    payload = _fallback_verification(claim, document)

                verified.append(
                    Citation(
                        claim=claim[:300],
                        source_doc_id=document.id,
                        source_title=document.title,
                        is_verified=bool(payload.is_verified),
                        quote=(payload.quote[:300] if payload.quote else None),
                    )
                )

        step_log = [f"citation_verifier: produced {len(verified)} citations"]
        logger.info("citation_verifier_agent produced %s citations", len(verified))
        return {"verified_citations": verified, "step_log": step_log}
    except (CitationError, AgentError):
        raise
    except Exception as exc:
        raise CitationError(f"Citation verifier failed: {exc}") from exc
