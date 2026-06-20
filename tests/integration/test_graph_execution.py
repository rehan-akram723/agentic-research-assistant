"""Integration tests for the research graph execution."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.graph.graph_builder import build_graph
from src.graph.state import ResearchState
from src.schemas.document import Document


class TestGraphExecution:
    """Integration tests for the complete research pipeline."""

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_full_pipeline_execution(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that the full research pipeline executes end-to-end."""
        mock_arxiv.return_value = [sample_documents[0]]
        mock_scholar.return_value = [sample_documents[1]]
        mock_faiss.add_documents = MagicMock()
        mock_faiss.similarity_search = MagicMock(return_value=[sample_documents[0]])

        graph = build_graph()
        result = graph.invoke(base_state)

        assert result["error"] is None
        assert len(result["search_queries"]) > 0
        assert len(result["retrieved_docs"]) > 0
        assert len(result["summaries"]) > 0
        assert len(result["verified_citations"]) > 0
        assert len(result["final_report"]) > 0
        assert len(result["step_log"]) > 0

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_pipeline_handles_retrieval_failure(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
    ) -> None:
        """Test that pipeline gracefully handles retrieval failure."""
        mock_arxiv.side_effect = Exception("arXiv API error")
        mock_scholar.side_effect = Exception("Semantic Scholar API error")

        graph = build_graph()
        result = graph.invoke(base_state)

        assert result["error"] is not None
        assert "retrieval" in result["error"].lower()

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_pipeline_handles_zero_documents(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
    ) -> None:
        """Test that pipeline handles case where no documents are retrieved."""
        mock_arxiv.return_value = []
        mock_scholar.return_value = []

        graph = build_graph()
        result = graph.invoke(base_state)

        assert result["error"] is not None
        assert "no_documents_retrieved" in result["error"] or "zero" in result["error"].lower()
        assert len(result["summaries"]) == 0

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_pipeline_skips_citation_verification_when_disabled(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that pipeline skips citation verification when include_citations is False."""
        mock_arxiv.return_value = [sample_documents[0]]
        mock_scholar.return_value = [sample_documents[1]]
        mock_faiss.add_documents = MagicMock()
        mock_faiss.similarity_search = MagicMock(return_value=[sample_documents[0]])

        base_state["include_citations"] = False
        graph = build_graph()
        result = graph.invoke(base_state)

        assert result["error"] is None
        assert len(result["verified_citations"]) == 0

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_pipeline_accumulates_step_log(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that pipeline accumulates logs from each agent."""
        mock_arxiv.return_value = [sample_documents[0]]
        mock_scholar.return_value = [sample_documents[1]]
        mock_faiss.add_documents = MagicMock()
        mock_faiss.similarity_search = MagicMock(return_value=[sample_documents[0]])

        graph = build_graph()
        result = graph.invoke(base_state)

        step_log = result["step_log"]
        assert len(step_log) >= 5
        assert any("planner" in log for log in step_log)
        assert any("retrieval" in log for log in step_log)
        assert any("summarizer" in log for log in step_log)
        assert any("report_generator" in log for log in step_log)

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_pipeline_with_max_results_limit(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that pipeline respects max_results limit."""
        mock_arxiv.return_value = sample_documents * 5
        mock_scholar.return_value = sample_documents * 5
        mock_faiss.add_documents = MagicMock()
        mock_faiss.similarity_search = MagicMock(return_value=[sample_documents[0]])

        base_state["max_results"] = 3
        graph = build_graph()
        result = graph.invoke(base_state)

        assert len(result["retrieved_docs"]) <= 3

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_pipeline_returns_complete_state(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that pipeline returns a complete and valid ResearchState."""
        mock_arxiv.return_value = [sample_documents[0]]
        mock_scholar.return_value = [sample_documents[1]]
        mock_faiss.add_documents = MagicMock()
        mock_faiss.similarity_search = MagicMock(return_value=[sample_documents[0]])

        graph = build_graph()
        result = graph.invoke(base_state)

        # Check all required state fields are present
        assert "question" in result
        assert "search_queries" in result
        assert "retrieved_docs" in result
        assert "summaries" in result
        assert "verified_citations" in result
        assert "final_report" in result
        assert "error" in result
        assert "step_log" in result

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_pipeline_with_different_question(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that pipeline works with different research questions."""
        mock_arxiv.return_value = [sample_documents[0]]
        mock_scholar.return_value = [sample_documents[1]]
        mock_faiss.add_documents = MagicMock()
        mock_faiss.similarity_search = MagicMock(return_value=[sample_documents[0]])

        questions = [
            "What is artificial intelligence?",
            "How does deep learning work?",
            "What are neural networks?",
        ]

        for question in questions:
            state = base_state.copy()
            state["question"] = question
            graph = build_graph()
            result = graph.invoke(state)

            assert result["error"] is None
            assert result["question"] == question
            assert len(result["search_queries"]) > 0

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_pipeline_report_quality(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that the generated report has expected quality markers."""
        mock_arxiv.return_value = [sample_documents[0]]
        mock_scholar.return_value = [sample_documents[1]]
        mock_faiss.add_documents = MagicMock()
        mock_faiss.similarity_search = MagicMock(return_value=[sample_documents[0]])

        graph = build_graph()
        result = graph.invoke(base_state)
        report = result["final_report"]

        # Report should be substantial
        assert len(report) > 100
        # Report should have markdown formatting
        assert "#" in report
        # Report should have multiple lines
        lines = report.split("\n")
        assert len(lines) > 5
