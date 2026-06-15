"""Lightweight graph builder implementing the pipeline without langgraph.

For Phase 1 we provide a simple `CompiledGraph` with an `invoke` method that
runs the planner → retrieval → summarizer → (optional citation verifier)
→ report_generator sequence. This avoids a hard dependency on langgraph
for early development and keeps the code easy to test.
"""
from __future__ import annotations

from typing import Callable, Dict, Any
from src.graph.state import ResearchState
from src.agents.planner import planner_agent
from src.agents.retrieval import retrieval_agent
from src.agents.summarizer import summarizer_agent
from src.agents.citation_verifier import citation_verifier_agent
from src.agents.report_generator import report_generator_agent
from src.graph.edges import should_verify_citations, handle_retrieval_failure


class CompiledGraph:
    def __init__(self) -> None:
        self.steps = [
            ("planner", planner_agent),
            ("retrieval", retrieval_agent),
            ("summarizer", summarizer_agent),
        ]

    def invoke(self, state: ResearchState) -> ResearchState:
        # planner
        p_out = planner_agent(state)
        state["search_queries"] = p_out.get("search_queries", [])
        state.setdefault("step_log", []).extend(p_out.get("step_log", []))

        # retrieval
        r_out = retrieval_agent(state)
        state["retrieved_docs"] = r_out.get("retrieved_docs", [])
        state.setdefault("step_log", []).extend(r_out.get("step_log", []))

        # decide if retrieval failed
        next_node = handle_retrieval_failure(state)
        if next_node == "end":
            state["error"] = "no_documents_retrieved"
            return state

        # summarizer
        s_out = summarizer_agent(state)
        state["summaries"] = s_out.get("summaries", [])
        state.setdefault("step_log", []).extend(s_out.get("step_log", []))

        # conditional citation verification
        next_node = should_verify_citations(state)
        if next_node == "citation_verifier":
            c_out = citation_verifier_agent(state)
            state["verified_citations"] = c_out.get("verified_citations", [])
            state.setdefault("step_log", []).extend(c_out.get("step_log", []))
        else:
            state["verified_citations"] = []

        # report generator
        rg_out = report_generator_agent(state)
        state["final_report"] = rg_out.get("final_report", "")
        state.setdefault("step_log", []).extend(rg_out.get("step_log", []))

        return state


def build_graph() -> CompiledGraph:
    return CompiledGraph()
