"""ResearchService: orchestrates a single end-to-end research run.

This module is the only boundary between the API routing layer and the
graph/agent internals. No route handler imports from src.graph or src.agents.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from src.core.config import settings
from src.core.exceptions import AgentError, RetrievalError
from src.core.logger import get_logger
from src.core.metrics import RunMetrics, set_current_metrics
from src.graph.graph_builder import build_graph, CompiledGraph
from src.graph.state import ResearchState
from src.schemas.document import Citation
from src.schemas.response import ResearchResponse
from src.tools.faiss_memory_tool import faiss_memory

logger = get_logger(__name__)


def _build_initial_state(
    question: str,
    max_results: int,
    include_citations: bool,
) -> dict[str, Any]:
    capped = min(max_results, settings.max_results_cap)
    return {
        "question": question,
        "max_results": capped,
        "include_citations": include_citations,
        "search_queries": [],
        "retrieved_docs": [],
        "summaries": [],
        "verified_citations": [],
        "final_report": "",
        "error": None,
        "step_log": [],
    }


def _classify_status(final_state: dict[str, Any]) -> str:
    has_error = bool(final_state.get("error"))
    has_report = bool(final_state.get("final_report", "").strip())
    if has_error and not has_report:
        return "error"
    if has_error and has_report:
        return "partial"
    return "success"


def _coerce_citations(raw: list[Any]) -> list[Citation]:
    result: list[Citation] = []
    for item in raw:
        if isinstance(item, Citation):
            result.append(item)
        elif isinstance(item, dict):
            try:
                result.append(Citation.model_validate(item))
            except Exception as exc:
                logger.warning("Skipping malformed citation dict: %s | error: %s", item, exc)
        else:
            logger.warning("Unexpected citation type %s — skipping.", type(item).__name__)
    return result


def _run_graph_with_metrics(
    graph: CompiledGraph,
    initial_state: dict[str, Any],
    metrics: RunMetrics,
) -> dict[str, Any]:
    """Bind metrics to the thread context and invoke the graph synchronously.

    Called inside asyncio.to_thread — the ContextVar is inherited by the thread
    automatically, so every agent and tool in the call stack can write to the
    same RunMetrics instance without argument plumbing.
    """
    set_current_metrics(metrics)
    return graph.invoke(initial_state)


class ResearchService:
    """Stateless service wiring the graph to the API layer."""

    def __init__(self) -> None:
        logger.info("Initialising ResearchService — building LangGraph pipeline...")
        self.graph: CompiledGraph = build_graph()
        logger.info("ResearchService ready.")

    async def run(
        self,
        question: str,
        max_results: int,
        include_citations: bool,
    ) -> ResearchResponse:
        """Execute a full research pipeline and return a validated response.

        The synchronous CompiledGraph.invoke is offloaded to a thread pool via
        asyncio.to_thread so the FastAPI event loop is never blocked.

        A fresh RunMetrics is created for this request, bound to the thread
        context, and attached to the response on completion.
        """
        request_id = f"{int(time.time() * 1000)}"
        logger.info(
            "[%s] Research request — question=%r, max_results=%d, include_citations=%s",
            request_id, question[:80], max_results, include_citations,
        )

        faiss_memory.reset()

        metrics = RunMetrics()
        initial_state = _build_initial_state(question, max_results, include_citations)
        start_time = time.perf_counter()

        try:
            final_state: dict[str, Any] = await asyncio.to_thread(
                _run_graph_with_metrics, self.graph, initial_state, metrics
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 3)
            logger.error(
                "[%s] Graph raised an unexpected exception after %.3fs: %s",
                request_id, elapsed, exc, exc_info=True,
            )
            raise RuntimeError(f"Research pipeline failed: {exc}") from exc

        elapsed = round(time.perf_counter() - start_time, 3)
        status = _classify_status(final_state)
        docs_retrieved = len(final_state.get("retrieved_docs", []))
        queries_used = final_state.get("search_queries", [])
        error_message = final_state.get("error")

        logger.info(
            "[%s] Graph completed in %.3fs — status=%s, docs=%d, queries=%d, "
            "tokens=%d, cost=$%.6f, api_failures=%d",
            request_id, elapsed, status, docs_retrieved, len(queries_used),
            metrics.total_prompt_tokens + metrics.total_completion_tokens,
            metrics.estimated_cost_usd,
            metrics.api_failures,
        )

        if status == "error":
            if error_message and "retrieval" in error_message.lower():
                raise RetrievalError(error_message)
            raise AgentError(error_message or "An unspecified agent error occurred.")

        citations = _coerce_citations(final_state.get("verified_citations", []))

        return ResearchResponse(
            status=status,
            question=question,
            report=final_state.get("final_report", ""),
            citations=citations,
            queries_used=queries_used,
            docs_retrieved=docs_retrieved,
            processing_time_seconds=round(elapsed, 2),
            error_message=error_message,
            metrics=metrics.to_dict(),
        )
