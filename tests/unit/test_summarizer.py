"""Unit tests for the summarizer agent."""
from __future__ import annotations

import pytest

from src.agents.summarizer import summarizer_agent
from src.core.exceptions import AgentError
from src.graph.state import ResearchState


class TestSummarizerAgent:
    """Test suite for summarizer_agent."""

    def test_summarizer_generates_summaries(self, state_with_documents: ResearchState) -> None:
        """Test that summarizer generates one summary per document."""
        result = summarizer_agent(state_with_documents)
        
        assert "summaries" in result
        assert isinstance(result["summaries"], list)
        assert len(result["summaries"]) == len(state_with_documents["retrieved_docs"])

    def test_summarizer_creates_structured_format(self, state_with_documents: ResearchState) -> None:
        """Test that summarizer produces structured summary format."""
        result = summarizer_agent(state_with_documents)
        summaries = result["summaries"]
        
        for summary in summaries:
            assert isinstance(summary, str)
            assert "Title:" in summary
            assert "Authors:" in summary
            assert "Source:" in summary

    def test_summarizer_logs_step(self, state_with_documents: ResearchState) -> None:
        """Test that summarizer adds entries to step_log."""
        result = summarizer_agent(state_with_documents)
        
        assert "step_log" in result
        assert isinstance(result["step_log"], list)
        assert len(result["step_log"]) > 0
        assert "summarizer" in result["step_log"][0].lower()

    def test_summarizer_raises_on_invalid_documents(self, base_state: ResearchState) -> None:
        """Test that summarizer raises AgentError on malformed documents."""
        base_state["retrieved_docs"] = [{"invalid": "document"}]  # type: ignore
        
        with pytest.raises(AgentError):
            summarizer_agent(base_state)

    def test_summarizer_handles_empty_abstracts(self, base_state: ResearchState, sample_document) -> None:
        """Test that summarizer handles documents without abstracts."""
        sample_document.abstract = ""
        base_state["retrieved_docs"] = [sample_document]
        
        result = summarizer_agent(base_state)
        
        assert len(result["summaries"]) == 1
        assert "No abstract available" in result["summaries"][0]

    def test_summarizer_includes_document_id(self, state_with_documents: ResearchState) -> None:
        """Test that summarizer includes document ID in summaries."""
        result = summarizer_agent(state_with_documents)
        summaries = result["summaries"]
        
        for summary, doc in zip(summaries, state_with_documents["retrieved_docs"]):
            assert doc.id in summary or "Document ID:" in summary

    def test_summarizer_returns_correct_structure(self, state_with_documents: ResearchState) -> None:
        """Test that summarizer returns the correct state patch structure."""
        result = summarizer_agent(state_with_documents)
        
        assert isinstance(result, dict)
        assert "summaries" in result
        assert "step_log" in result
        assert len(result) == 2, "Summarizer should only return summaries and step_log"

    def test_summarizer_works_with_multiple_documents(self, sample_documents) -> None:
        """Test that summarizer processes multiple documents correctly."""
        state: ResearchState = ResearchState(
            question="test",
            max_results=10,
            include_citations=True,
            search_queries=[],
            retrieved_docs=sample_documents,
            summaries=[],
            verified_citations=[],
            final_report="",
            error=None,
            step_log=[],
        )
        
        result = summarizer_agent(state)
        
        assert len(result["summaries"]) == len(sample_documents)
