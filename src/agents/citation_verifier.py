"""Citation verifier: uses FAISS to find candidate supporting documents.

Phase 1 implementation does not call an LLM. Instead it queries the FAISS
memory for the most similar document to each summary and produces a
`Citation` record linking the summary to that document. This provides a
deterministic, testable baseline for the citation pipeline.
"""
from __future__ import annotations

from typing import List
from src.graph.state import ResearchState
from src.tools.faiss_memory_tool import faiss_memory
from src.schemas.document import Citation


def citation_verifier_agent(state: ResearchState) -> dict:
    summaries: List[str] = state.get("summaries", []) or []
    verified: List[Citation] = []

    for s in summaries:
        hits = faiss_memory.similarity_search(s, k=1)
        if hits:
            doc = hits[0]
            claim = s.split('\n')[0][:300]
            c = Citation(
                claim=claim,
                source_doc_id=doc.id,
                source_title=doc.title,
                is_verified=True,
                quote=(doc.abstract[:300] if doc.abstract else None),
            )
        else:
            c = Citation(
                claim=(s.split('\n')[0][:300]),
                source_doc_id="",
                source_title="",
                is_verified=False,
                quote=None,
            )
        verified.append(c)

    step_log = [f"citation_verifier: produced {len(verified)} citations"]
    return {"verified_citations": verified, "step_log": step_log}
