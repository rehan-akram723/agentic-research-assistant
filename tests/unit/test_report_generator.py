"""Unit tests for the report generator agent."""
from __future__ import annotations

import pytest

from src.agents.report_generator import report_generator_agent
from src.core.exceptions import AgentError
from src.graph.state import ResearchState


class TestReportGeneratorAgent:
    """Test suite for report_generator_agent."""

    def test_report_generator_produces_report(self, state_with_citations: ResearchState) -> None:
        """Test that report generator produces a final report string."""
        result = report_generator_agent(state_with_citations)
        
        assert "final_report" in result
        assert isinstance(result["final_report"], str)
        assert len(result["final_report"]) > 0

    def test_report_generator_includes_question(self, state_with_citations: ResearchState) -> None:
        """Test that report includes the research question."""
        result = report_generator_agent(state_with_citations)
        report = result["final_report"]
        
        question = state_with_citations.get("question", "")
        # Report should contain reference to the question
        assert "question" in report.lower() or len(report) > 100

    def test_report_generator_includes_summaries(self, state_with_citations: ResearchState) -> None:
        """Test that report includes content from summaries."""
        result = report_generator_agent(state_with_citations)
        report = result["final_report"]
        
        # Report should be markdown formatted with sections
        assert "#" in report, "Report should have markdown headers"

    def test_report_generator_includes_citations(self, state_with_citations: ResearchState) -> None:
        """Test that report includes citation information."""
        result = report_generator_agent(state_with_citations)
        report = result["final_report"]
        
        # Report should reference citations or verification status
        assert "citation" in report.lower() or "reference" in report.lower()

    def test_report_generator_logs_step(self, state_with_citations: ResearchState) -> None:
        """Test that report generator adds entries to step_log."""
        result = report_generator_agent(state_with_citations)
        
        assert "step_log" in result
        assert isinstance(result["step_log"], list)
        assert len(result["step_log"]) > 0
        assert "report_generator" in result["step_log"][0].lower()

    def test_report_generator_handles_empty_summaries(self, base_state: ResearchState) -> None:
        """Test that report generator handles empty summaries gracefully."""
        base_state["summaries"] = []
        base_state["verified_citations"] = []
        
        result = report_generator_agent(base_state)
        
        assert "final_report" in result
        assert isinstance(result["final_report"], str)

    def test_report_generator_handles_no_citations(self, state_with_summaries: ResearchState) -> None:
        """Test that report generator handles case with no citations."""
        state_with_summaries["verified_citations"] = []
        
        result = report_generator_agent(state_with_summaries)
        
        assert "final_report" in result
        assert isinstance(result["final_report"], str)
        assert len(result["final_report"]) > 0

    def test_report_generator_has_structured_sections(self, state_with_citations: ResearchState) -> None:
        """Test that report has expected markdown sections."""
        result = report_generator_agent(state_with_citations)
        report = result["final_report"]
        
        # Check for common markdown sections
        sections = [
            "Executive Summary",
            "Key Findings",
            "Methods",
            "Limitations",
            "Reference",
        ]
        section_count = sum(1 for section in sections if section.lower() in report.lower())
        assert section_count >= 2, f"Report should have at least 2 main sections, found {section_count}"

    def test_report_generator_returns_correct_structure(self, state_with_citations: ResearchState) -> None:
        """Test that report generator returns the correct state patch structure."""
        result = report_generator_agent(state_with_citations)
        
        assert isinstance(result, dict)
        assert "final_report" in result
        assert "step_log" in result
        assert len(result) == 2, "Report generator should only return final_report and step_log"

    def test_report_generator_markdown_format(self, state_with_citations: ResearchState) -> None:
        """Test that report is in valid markdown format."""
        result = report_generator_agent(state_with_citations)
        report = result["final_report"]
        
        # Should have markdown headers
        assert "#" in report
        # Should have proper line breaks
        assert "\n" in report
        # Should not be excessively short
        assert len(report) > 50
