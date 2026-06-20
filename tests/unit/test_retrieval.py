"""Unit tests for the retrieval agent."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.agents.retrieval import retrieval_agent
from src.core.exceptions import RetrievalError
from src.graph.state import ResearchState
from src.schemas.document import Document


class TestRetrievalAgent:
    """Test suite for retrieval_agent."""

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_retrieval_fetches_documents(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        state_with_queries: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that retrieval agent fetches and returns documents."""
        mock_arxiv.return_value = [sample_documents[0]]
        mock_scholar.return_value = [sample_documents[1]]
        mock_faiss.add_documents = MagicMock()
        
        result = retrieval_agent(state_with_queries)
        
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)
        assert len(result["retrieved_docs"]) > 0

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_retrieval_deduplicates_documents(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        state_with_queries: ResearchState,
        sample_document: Document,
    ) -> None:
        """Test that retrieval agent deduplicates documents by ID."""
        mock_arxiv.return_value = [sample_document]
        mock_scholar.return_value = [sample_document]  # Same document ID
        mock_faiss.add_documents = MagicMock()
        
        result = retrieval_agent(state_with_queries)
        docs = result["retrieved_docs"]
        
        # Should have only one document despite appearing in both sources
        doc_ids = [doc.id for doc in docs]
        assert len(doc_ids) == len(set(doc_ids)), "Retrieval should deduplicate by ID"

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_retrieval_logs_step(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        state_with_queries: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that retrieval agent adds entries to step_log."""
        mock_arxiv.return_value = [sample_documents[0]]
        mock_scholar.return_value = [sample_documents[1]]
        mock_faiss.add_documents = MagicMock()
        
        result = retrieval_agent(state_with_queries)
        
        assert "step_log" in result
        assert isinstance(result["step_log"], list)
        assert len(result["step_log"]) > 0
        assert "retrieval" in result["step_log"][0].lower()

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    def test_retrieval_raises_on_no_queries(
        self,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        base_state: ResearchState,
    ) -> None:
        """Test that retrieval raises RetrievalError when no queries are available."""
        base_state["search_queries"] = []
        
        with pytest.raises(RetrievalError):
            retrieval_agent(base_state)

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_retrieval_raises_on_zero_documents(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        state_with_queries: ResearchState,
    ) -> None:
        """Test that retrieval raises RetrievalError when no documents are retrieved."""
        mock_arxiv.return_value = []
        mock_scholar.return_value = []
        
        with pytest.raises(RetrievalError):
            retrieval_agent(state_with_queries)

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_retrieval_respects_max_results(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        state_with_queries: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that retrieval agent respects max_results limit."""
        mock_arxiv.return_value = sample_documents
        mock_scholar.return_value = sample_documents
        mock_faiss.add_documents = MagicMock()
        
        state_with_queries["max_results"] = 2
        result = retrieval_agent(state_with_queries)
        
        assert len(result["retrieved_docs"]) <= 2

    @patch("src.agents.retrieval.search_arxiv")
    @patch("src.agents.retrieval.search_semantic_scholar")
    @patch("src.agents.retrieval.faiss_memory")
    def test_retrieval_adds_to_faiss(
        self,
        mock_faiss: MagicMock,
        mock_scholar: MagicMock,
        mock_arxiv: MagicMock,
        state_with_queries: ResearchState,
        sample_documents: list[Document],
    ) -> None:
        """Test that retrieval agent populates FAISS memory."""
        mock_arxiv.return_value = [sample_documents[0]]
        mock_scholar.return_value = [sample_documents[1]]
        mock_faiss.add_documents = MagicMock()
        
        result = retrieval_agent(state_with_queries)
        
        mock_faiss.add_documents.assert_called_once()
        called_docs = mock_faiss.add_documents.call_args[0][0]
        assert len(called_docs) == 2
