"""Unit tests for the planner agent."""
from __future__ import annotations

import pytest

from src.agents.planner import planner_agent
from src.core.exceptions import AgentError
from src.graph.state import ResearchState


class TestPlannerAgent:
    """Test suite for planner_agent."""

    def test_planner_generates_queries_from_question(self, base_state: ResearchState) -> None:
        """Test that planner generates search queries from a research question."""
        result = planner_agent(base_state)
        
        assert "search_queries" in result
        assert isinstance(result["search_queries"], list)
        assert len(result["search_queries"]) > 0
        assert all(isinstance(q, str) and len(q) > 0 for q in result["search_queries"])

    def test_planner_generates_distinct_queries(self, base_state: ResearchState) -> None:
        """Test that planner generates distinct (non-duplicate) queries."""
        result = planner_agent(base_state)
        queries = result["search_queries"]
        
        unique_queries = list(set(q.lower() for q in queries))
        assert len(unique_queries) == len(queries), "Planner should generate distinct queries"

    def test_planner_logs_step(self, base_state: ResearchState) -> None:
        """Test that planner adds entries to step_log."""
        result = planner_agent(base_state)
        
        assert "step_log" in result
        assert isinstance(result["step_log"], list)
        assert len(result["step_log"]) > 0
        assert "planner" in result["step_log"][0].lower()

    def test_planner_raises_on_empty_question(self) -> None:
        """Test that planner raises AgentError when question is empty."""
        state: ResearchState = ResearchState(
            question="",
            max_results=10,
            include_citations=True,
            search_queries=[],
            retrieved_docs=[],
            summaries=[],
            verified_citations=[],
            final_report="",
            error=None,
            step_log=[],
        )
        
        with pytest.raises(AgentError):
            planner_agent(state)

    def test_planner_raises_on_none_question(self) -> None:
        """Test that planner raises AgentError when question is None."""
        state = ResearchState(
            question=None,  # type: ignore
            max_results=10,
            include_citations=True,
            search_queries=[],
            retrieved_docs=[],
            summaries=[],
            verified_citations=[],
            final_report="",
            error=None,
            step_log=[],
        )
        
        with pytest.raises(AgentError):
            planner_agent(state)

    def test_planner_query_count_reasonable(self, base_state: ResearchState) -> None:
        """Test that planner generates a reasonable number of queries."""
        result = planner_agent(base_state)
        query_count = len(result["search_queries"])
        
        assert 1 <= query_count <= 5, "Planner should generate 1 to 5 queries"

    def test_planner_updates_state_correctly(self, base_state: ResearchState) -> None:
        """Test that planner returns the correct state patch structure."""
        result = planner_agent(base_state)
        
        assert isinstance(result, dict)
        assert "search_queries" in result
        assert "step_log" in result
        assert len(result) == 2, "Planner should only return search_queries and step_log"
