"""Retrieval agent: queries external tools and populates `retrieved_docs`.

This agent calls the arXiv and Semantic Scholar tool wrappers, deduplicates
by document id, converts results to `Document` instances and registers them
with the FAISS memory tool.
"""
from __future__ import annotations

from typing import List, Dict
from src.graph.state import ResearchState
from src.tools import arxiv_tool, semantic_scholar_tool, faiss_memory_tool
from src.schemas.document import Document


def retrieval_agent(state: ResearchState) -> dict:
    queries: List[str] = state.get("search_queries", []) or []
    max_results = int(state.get("max_results", 10))
    collected: Dict[str, Document] = {}

    for q in queries:
        a_docs = arxiv_tool.search_arxiv(q, max_results=max_results // 2 or 1)
        s_docs = semantic_scholar_tool.search_semantic_scholar(q, max_results=max_results // 2 or 1)

        for d in (a_docs or []) + (s_docs or []):
            try:
                doc = Document.model_validate(d) if hasattr(Document, "model_validate") else Document(**d)
            except Exception:
                # Skip malformed documents
                continue
            collected[doc.id] = doc

    docs = list(collected.values())
    # register into FAISS memory
    from src.tools.faiss_memory_tool import faiss_memory

    faiss_memory.reset()
    faiss_memory.add_documents(docs)

    step_log = [f"retrieval: retrieved {len(docs)} unique documents"]
    return {"retrieved_docs": docs, "step_log": step_log}
