"""Lightweight graph builder implementing the pipeline without langgraph.

For Phase 3 we provide a simple `CompiledGraph` with an `invoke` method that
runs the planner → retrieval → summarizer → (optional citation verifier)
→ report_generator sequence. The graph handles exceptions from agents and sets
the error field appropriately.
"""
from __future__ import annotations

from typing import Any
from src.core.logger import get_logger
from src.graph.state import ResearchState
from src.agents.planner import planner_agent
from src.agents.retrieval import retrieval_agent
from src.agents.summarizer import summarizer_agent
from src.agents.citation_verifier import citation_verifier_agent
from src.agents.report_generator import report_generator_agent
from src.graph.edges import should_verify_citations, handle_retrieval_failure

logger = get_logger(__name__)


class CompiledGraph:
    """Single-threaded research pipeline orchestrator."""

    def __init__(self) -> None:
        self.steps = [
            ("planner", planner_agent),
            ("retrieval", retrieval_agent),
            ("summarizer", summarizer_agent),
        ]

    def invoke(self, state: ResearchState) -> ResearchState:
        """Execute the research pipeline sequentially.
        
        Args:
            state: The initial ResearchState with a question.
        
        Returns:
            The final ResearchState with all fields populated or errors set.
        """
        if state.get("step_log") is None:
            state["step_log"] = []

        try:
            # planner: generate search queries
            try:
                p_out = planner_agent(state)
                state["search_queries"] = p_out.get("search_queries", [])
                state.setdefault("step_log", []).extend(p_out.get("step_log", []))
            except Exception as exc:
                logger.error("planner_agent failed: %s", exc)
                state["error"] = f"planner_error: {exc}"
                return state

            # retrieval: fetch documents from sources
            try:
                r_out = retrieval_agent(state)
                state["retrieved_docs"] = r_out.get("retrieved_docs", [])
                state.setdefault("step_log", []).extend(r_out.get("step_log", []))
            except Exception as exc:
                logger.error("retrieval_agent failed: %s", exc)
                state["error"] = f"retrieval_error: {exc}"
                return state

            # decide if retrieval succeeded
            next_node = handle_retrieval_failure(state)
            if next_node == "end":
                state["error"] = "no_documents_retrieved"
                state.setdefault("step_log", []).append("graph: retrieval returned zero documents")
                return state

            # summarizer: summarize each document
            try:
                s_out = summarizer_agent(state)
                state["summaries"] = s_out.get("summaries", [])
                state.setdefault("step_log", []).extend(s_out.get("step_log", []))
            except Exception as exc:
                logger.error("summarizer_agent failed: %s", exc)
                state["error"] = f"summarizer_error: {exc}"
                return state

            # conditional citation verification
            next_node = should_verify_citations(state)
            if next_node == "citation_verifier":
                try:
                    c_out = citation_verifier_agent(state)
                    state["verified_citations"] = c_out.get("verified_citations", [])
                    state.setdefault("step_log", []).extend(c_out.get("step_log", []))
                except Exception as exc:
                    logger.error("citation_verifier_agent failed: %s", exc)
                    state["error"] = f"citation_verifier_error: {exc}"
                    return state
            else:
                state["verified_citations"] = []

            # report generator: synthesize final report
            try:
                rg_out = report_generator_agent(state)
                state["final_report"] = rg_out.get("final_report", "")
                state.setdefault("step_log", []).extend(rg_out.get("step_log", []))
            except Exception as exc:
                logger.error("report_generator_agent failed: %s", exc)
                state["error"] = f"report_generator_error: {exc}"
                return state

            # Success
            state["error"] = None
            state.setdefault("step_log", []).append("graph: all stages completed successfully")
            return state

        except Exception as exc:  # pragma: no cover
            logger.error("Graph invoke encountered an unexpected error: %s", exc)
            state["error"] = f"graph_error: {exc}"
            return state


def build_graph() -> CompiledGraph:
    """Factory for creating a new compiled graph instance.
    
    Returns:
        A new CompiledGraph ready for invocation.
    """
    return CompiledGraph()
