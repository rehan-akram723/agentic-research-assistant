"""Retrieval agent for academic source collection.

This agent queries arXiv and Semantic Scholar for every planned search query,
deduplicates results, stores them in FAISS memory, and returns a state patch.
"""
from __future__ import annotations

from typing import Dict, List

from src.core.exceptions import AgentError, RetrievalError
from src.core.logger import get_logger
from src.graph.state import ResearchState
from src.schemas.document import Document
from src.tools.arxiv_tool import search_arxiv
from src.tools.faiss_memory_tool import faiss_memory
from src.tools.semantic_scholar_tool import search_semantic_scholar

logger = get_logger(__name__)


def _coerce_document(item: object) -> Document:
    if isinstance(item, Document):
        return item

    if isinstance(item, dict):
        if hasattr(Document, "model_validate"):
            return Document.model_validate(item)  # type: ignore[attr-defined]
        return Document(**item)

    raise RetrievalError("Retrieved item is not a valid document payload.")


def retrieval_agent(state: ResearchState) -> dict:
    try:
        queries: List[str] = [str(query).strip() for query in (state.get("search_queries") or []) if str(query).strip()]
        if not queries:
            raise RetrievalError("No search queries available for retrieval.")

        max_results = max(1, int(state.get("max_results", 10) or 10))
        per_source_limit = max(1, max_results // 2)

        collected: Dict[str, Document] = {}
        for query in queries:
            arxiv_docs = search_arxiv(query, max_results=per_source_limit)
            scholar_docs = search_semantic_scholar(query, max_results=per_source_limit)

            for raw_doc in list(arxiv_docs or []) + list(scholar_docs or []):
                try:
                    document = _coerce_document(raw_doc)
                except RetrievalError:
                    continue
                collected.setdefault(document.id, document)

        documents = list(collected.values())[:max_results]
        faiss_memory.reset()
        if documents:
            faiss_memory.add_documents(documents)

        # FIX: Remove the Exception raise. Log the empty retrieval state instead.
        if not documents:
            logger.warning("Retrieval returned zero documents across all sources and queries.")

        step_log = [f"retrieval: retrieved {len(documents)} unique documents"]
        logger.info("retrieval_agent retrieved %s documents", len(documents))
        
        # This will now safely return {"retrieved_docs": []} to the graph
        return {"retrieved_docs": documents, "step_log": step_log}
        
    except (RetrievalError, AgentError):
        raise
    except Exception as exc:
        raise RetrievalError(f"Retrieval agent failed: {exc}") from exc