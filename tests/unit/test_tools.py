from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.schemas.document import Document
from src.tools import arxiv_tool, semantic_scholar_tool
from src.tools.faiss_memory_tool import FAISSMemory


@dataclass
class _FakeAuthor:
    name: str


@dataclass
class _FakeArxivResult:
    entry_id: str = "arxiv:1234.5678"
    title: str = "A Study of Large Language Models"
    authors: list[_FakeAuthor] = None  # type: ignore[assignment]
    summary: str = "This paper studies large language models and their applications."
    pdf_url: str = "https://arxiv.org/pdf/1234.5678.pdf"
    published: object = None

    def __post_init__(self) -> None:
        if self.authors is None:
            self.authors = [_FakeAuthor("Ada Lovelace"), _FakeAuthor("Alan Turing")]


@dataclass
class _FakeSearch:
    results_payload: list[_FakeArxivResult]

    def results(self):
        return iter(self.results_payload)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def clean_memory() -> FAISSMemory:
    memory = FAISSMemory()
    memory.reset()
    yield memory
    memory.reset()


def test_arxiv_search_returns_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        arxiv_tool,
        "arxiv",
        type(
            "FakeArxivModule",
            (),
            {"Search": lambda **kwargs: _FakeSearch([_FakeArxivResult()])},
        ),
    )

    results = arxiv_tool.search_arxiv("large language models", max_results=1)

    assert isinstance(results, list)
    assert results
    assert all(isinstance(item, Document) for item in results)

    first = results[0]
    assert first.id
    assert first.title
    assert first.abstract


def test_semantic_scholar_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_payload = {
        "data": [
            {
                "paperId": "ss:abc123",
                "title": "Large Language Models for Research",
                "abstract": "This work uses large language models for scientific research.",
                "authors": [{"name": "Grace Hopper"}],
                "url": "https://example.org/paper",
                "year": 2024,
            }
        ]
    }
    monkeypatch.setattr(semantic_scholar_tool.requests, "get", lambda *args, **kwargs: _FakeResponse(fake_payload))

    results = semantic_scholar_tool.search_semantic_scholar("large language models", max_results=1)

    assert isinstance(results, list)
    assert results
    assert all(isinstance(item, Document) for item in results)

    first = results[0]
    assert first.id
    assert first.title
    assert first.abstract


def test_faiss_stores_and_retrieves(clean_memory: FAISSMemory) -> None:
    documents = [
        Document(
            id="doc-1",
            title="Large Language Models in Science",
            authors=["A. Author"],
            abstract="Large language models are useful for science tasks.",
            url="https://example.org/1",
            published_date="2024-01-01",
            source="arxiv",
            relevance_score=None,
        ),
        Document(
            id="doc-2",
            title="Embedding Methods for Research",
            authors=["B. Author"],
            abstract="Embedding methods improve retrieval of scientific documents.",
            url="https://example.org/2",
            published_date="2024-01-02",
            source="semantic_scholar",
            relevance_score=None,
        ),
    ]

    clean_memory.add_documents(documents)
    results = clean_memory.similarity_search("large language models for science", k=2)

    assert isinstance(results, list)
    assert results
    assert all(isinstance(item, Document) for item in results)
    assert {item.id for item in results}.issubset({"doc-1", "doc-2"})

    clean_memory.reset()
    assert clean_memory.similarity_search("large language models for science", k=2) == []
