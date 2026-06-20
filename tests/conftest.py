"""Shared test fixtures and utilities."""
from __future__ import annotations

import pytest
from typing import Dict, Any, List

from src.graph.state import ResearchState
from src.schemas.document import Document, Citation


@pytest.fixture
def base_state() -> ResearchState:
    """Return a minimal valid ResearchState for testing."""
    return ResearchState(
        question="What is machine learning?",
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


@pytest.fixture
def sample_document() -> Document:
    """Return a sample Document for testing."""
    return Document(
        id="arxiv:2301.00001",
        title="An Introduction to Machine Learning",
        authors=["Alice Author", "Bob Bob"],
        abstract="This paper introduces the fundamentals of machine learning. We discuss supervised learning, unsupervised learning, and reinforcement learning techniques.",
        url="https://arxiv.org/abs/2301.00001",
        published_date="2023-01-01",
        source="arxiv",
        relevance_score=0.95,
    )


@pytest.fixture
def sample_documents(sample_document: Document) -> List[Document]:
    """Return a list of sample Documents for testing."""
    return [
        sample_document,
        Document(
            id="scholar:456",
            title="Deep Learning Fundamentals",
            authors=["Charlie Chen"],
            abstract="Deep learning has revolutionized machine learning. Convolutional networks and transformers are key architectures.",
            url="https://example.com/deep-learning",
            published_date="2023-06-15",
            source="semantic_scholar",
            relevance_score=0.87,
        ),
    ]


@pytest.fixture
def state_with_queries(base_state: ResearchState) -> ResearchState:
    """Return a state with search queries already generated."""
    state = base_state.copy()
    state["search_queries"] = [
        "machine learning algorithms",
        "machine learning survey",
        "machine learning applications",
    ]
    return state


@pytest.fixture
def state_with_documents(state_with_queries: ResearchState, sample_documents: List[Document]) -> ResearchState:
    """Return a state with retrieved documents."""
    state = state_with_queries.copy()
    state["retrieved_docs"] = sample_documents
    return state


@pytest.fixture
def state_with_summaries(state_with_documents: ResearchState) -> ResearchState:
    """Return a state with document summaries."""
    state = state_with_documents.copy()
    state["summaries"] = [
        "Document ID: arxiv:2301.00001\nTitle: An Introduction to Machine Learning\n"
        "Authors: Alice Author, Bob Bob\nSource: arxiv\nPublished: 2023-01-01\n"
        "Summary: This paper introduces machine learning fundamentals including supervised, unsupervised, "
        "and reinforcement learning.\nKey Points:\n- Core ML concepts\nClaims:\n- ML has three main paradigms",
        "Document ID: scholar:456\nTitle: Deep Learning Fundamentals\n"
        "Authors: Charlie Chen\nSource: semantic_scholar\nPublished: 2023-06-15\n"
        "Summary: Deep learning has transformed the field with CNNs and transformers.\n"
        "Key Points:\n- Deep learning impact\nClaims:\n- Transformers are key architectures",
    ]
    return state


@pytest.fixture
def state_with_citations(state_with_summaries: ResearchState) -> ResearchState:
    """Return a state with verified citations."""
    state = state_with_summaries.copy()
    state["verified_citations"] = [
        Citation(
            claim="ML has three main paradigms",
            source_doc_id="arxiv:2301.00001",
            source_title="An Introduction to Machine Learning",
            is_verified=True,
            quote="supervised learning, unsupervised learning, and reinforcement learning",
        ),
        Citation(
            claim="Transformers are key architectures",
            source_doc_id="scholar:456",
            source_title="Deep Learning Fundamentals",
            is_verified=True,
            quote="Convolutional networks and transformers are key architectures",
        ),
    ]
    return state
