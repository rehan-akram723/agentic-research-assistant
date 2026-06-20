"""Unit tests for the citation verifier agent."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.agents.citation_verifier import citation_verifier_agent
from src.core.exceptions import CitationError
from src.graph.state import ResearchState
from src.schemas.document import Citation


class TestCitationVerifierAgent:
    """Test suite for citation_verifier_agent."""

    @patch("src.agents.citation_verifier.faiss_memory")
    def test_citation_verifier_creates_citations(
        self,
        mock_faiss: MagicMock,
        state_with_summaries: ResearchState,
    ) -> None:
        """Test that citation verifier generates Citation objects."""
        mock_faiss.similarity_search = MagicMock(return_value=[state_with_summaries["retrieved_docs"][0]])
        
        result = citation_verifier_agent(state_with_summaries)
        
        assert "verified_citations" in result
        assert isinstance(result["verified_citations"], list)
        assert len(result["verified_citations"]) > 0
        assert all(isinstance(c, Citation) for c in result["verified_citations"])

    @patch("src.agents.citation_verifier.faiss_memory")
    def test_citation_verifier_sets_verification_status(
        self,
        mock_faiss: MagicMock,
        state_with_summaries: ResearchState,
    ) -> None:
        """Test that citation verifier sets is_verified flag."""
        mock_faiss.similarity_search = MagicMock(return_value=[state_with_summaries["retrieved_docs"][0]])
        
        result = citation_verifier_agent(state_with_summaries)
        citations = result["verified_citations"]
        
        assert any(c.is_verified for c in citations), "At least some citations should be verified"

    @patch("src.agents.citation_verifier.faiss_memory")
    def test_citation_verifier_logs_step(
        self,
        mock_faiss: MagicMock,
        state_with_summaries: ResearchState,
    ) -> None:
        """Test that citation verifier adds entries to step_log."""
        mock_faiss.similarity_search = MagicMock(return_value=[state_with_summaries["retrieved_docs"][0]])
        
        result = citation_verifier_agent(state_with_summaries)
        
        assert "step_log" in result
        assert isinstance(result["step_log"], list)
        assert len(result["step_log"]) > 0
        assert "citation_verifier" in result["step_log"][0].lower()

    def test_citation_verifier_raises_on_no_summaries(self, base_state: ResearchState) -> None:
        """Test that citation verifier raises CitationError when no summaries are available."""
        base_state["summaries"] = []
        
        with pytest.raises(CitationError):
            citation_verifier_agent(base_state)

    def test_citation_verifier_raises_on_no_documents(self, state_with_summaries: ResearchState) -> None:
        """Test that citation verifier raises CitationError when no documents are available."""
        state_with_summaries["retrieved_docs"] = []
        
        with pytest.raises(CitationError):
            citation_verifier_agent(state_with_summaries)

    @patch("src.agents.citation_verifier.faiss_memory")
    def test_citation_verifier_creates_unverified_citations(
        self,
        mock_faiss: MagicMock,
        state_with_summaries: ResearchState,
    ) -> None:
        """Test that citation verifier creates unverified citations when no match found."""
        mock_faiss.similarity_search = MagicMock(return_value=[])
        
        result = citation_verifier_agent(state_with_summaries)
        citations = result["verified_citations"]
        
        assert any(not c.is_verified for c in citations), "At least some citations should be unverified"

    @patch("src.agents.citation_verifier.faiss_memory")
    def test_citation_verifier_includes_source_info(
        self,
        mock_faiss: MagicMock,
        state_with_summaries: ResearchState,
    ) -> None:
        """Test that citation verifier includes source document information."""
        mock_faiss.similarity_search = MagicMock(return_value=[state_with_summaries["retrieved_docs"][0]])
        
        result = citation_verifier_agent(state_with_summaries)
        citations = result["verified_citations"]
        
        for citation in citations:
            assert citation.source_doc_id or citation.source_title or not citation.is_verified

    @patch("src.agents.citation_verifier.faiss_memory")
    def test_citation_verifier_returns_correct_structure(
        self,
        mock_faiss: MagicMock,
        state_with_summaries: ResearchState,
    ) -> None:
        """Test that citation verifier returns the correct state patch structure."""
        mock_faiss.similarity_search = MagicMock(return_value=[state_with_summaries["retrieved_docs"][0]])
        
        result = citation_verifier_agent(state_with_summaries)
        
        assert isinstance(result, dict)
        assert "verified_citations" in result
        assert "step_log" in result
        assert len(result) == 2, "Citation verifier should only return verified_citations and step_log"

    @patch("src.agents.citation_verifier.faiss_memory")
    def test_citation_verifier_extracts_claims(
        self,
        mock_faiss: MagicMock,
        state_with_summaries: ResearchState,
    ) -> None:
        """Test that citation verifier extracts and uses summary claims."""
        mock_faiss.similarity_search = MagicMock(return_value=[state_with_summaries["retrieved_docs"][0]])
        
        result = citation_verifier_agent(state_with_summaries)
        citations = result["verified_citations"]
        
        # Claims should be extracted from summaries
        assert all(len(c.claim) > 0 for c in citations)
