"""Retrieval agent — concurrent document collection via DocumentAggregator."""
from __future__ import annotations

from typing import List

from src.core.exceptions import AgentError, RetrievalError
from src.core.logger import get_logger
from src.core.metrics import track_latency
from src.graph.state import ResearchState
from src.retrieval.aggregator import DocumentAggregator
from src.tools.faiss_memory_tool import faiss_memory

logger = get_logger(__name__)


@track_latency("retrieval")
def retrieval_agent(state: ResearchState) -> dict:
    try:
        queries: List[str] = [
            str(q).strip()
            for q in (state.get("search_queries") or [])
            if str(q).strip()
        ]
        if not queries:
            raise RetrievalError("No search queries available for retrieval.")

        max_results = max(1, int(state.get("max_results", 10) or 10))

        aggregator = DocumentAggregator(max_results=max_results)
        documents = aggregator.run(queries)  # concurrent, metrics recorded inside

        faiss_memory.reset()
        if documents:
            faiss_memory.add_documents(documents)
            question = str(state.get("question", "")).strip()
            if question:
                documents = faiss_memory.similarity_search(question, k=max_results)

        if not documents:
            logger.warning("Retrieval returned zero documents across all sources and queries.")

        step_log = [f"retrieval: {len(documents)} unique documents via concurrent aggregator"]
        logger.info("retrieval_agent: %d documents retrieved.", len(documents))
        return {"retrieved_docs": documents, "step_log": step_log}

    except (RetrievalError, AgentError):
        raise
    except Exception as exc:
        raise RetrievalError(f"Retrieval agent failed: {exc}") from exc