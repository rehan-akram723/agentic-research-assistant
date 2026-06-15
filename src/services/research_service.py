"""ResearchService: orchestrates a single research run using the compiled graph.

This service builds the graph, resets FAISS memory, invokes the graph and
packages the final state into a `ResearchResponse` model.
"""
from __future__ import annotations

import time
from src.graph.graph_builder import build_graph
from src.schemas.response import ResearchResponse
from src.tools.faiss_memory_tool import faiss_memory
from typing import Any


class ResearchService:
    def __init__(self) -> None:
        self.graph = build_graph()

    async def run(self, question: str, max_results: int, include_citations: bool) -> ResearchResponse:
        faiss_memory.reset()
        start = time.time()
        initial_state: dict[str, Any] = {
            "question": question,
            "max_results": max_results,
            "include_citations": include_citations,
            "search_queries": [],
            "retrieved_docs": [],
            "summaries": [],
            "verified_citations": [],
            "final_report": "",
            "error": None,
            "step_log": [],
        }

        final_state = self.graph.invoke(initial_state)  # synchronous for Phase 1

        processing_time = round(time.time() - start, 2)
        return ResearchResponse(
            status=("error" if final_state.get("error") else "success"),
            question=question,
            report=final_state.get("final_report", ""),
            citations=final_state.get("verified_citations", []),
            queries_used=final_state.get("search_queries", []),
            docs_retrieved=len(final_state.get("retrieved_docs", [])),
            processing_time_seconds=processing_time,
            error_message=final_state.get("error"),
        )
