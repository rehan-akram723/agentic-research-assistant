from __future__ import annotations

from typing import List

import pytest

from src.schemas.document import Document
from src.tools.arxiv_tool import search_arxiv
from src.tools.faiss_memory_tool import faiss_memory, FAISSMemory


@pytest.fixture(autouse=False)
def clean_faiss_memory() -> FAISSMemory:
    """Provide a clean singleton memory store for each test."""
    faiss_memory.reset()
    yield faiss_memory
    faiss_memory.reset()


def _as_documents(items: List[object]) -> List[Document]:
    documents: List[Document] = []
    for item in items:
        if isinstance(item, Document):
            documents.append(item)
        else:
            documents.append(Document.model_validate(item))
    return documents


def test_tools_end_to_end_retrieval_embedding_and_reset(clean_faiss_memory: FAISSMemory) -> None:
    # 1-3. End-to-end retrieval from arXiv and contract validation.
    retrieved = search_arxiv("Large Language Models", max_results=5)

    assert isinstance(retrieved, list)
    assert retrieved, "Expected at least one arXiv result for a broad query"

    documents = _as_documents(retrieved)
    assert all(isinstance(doc, Document) for doc in documents)
    assert len({doc.id for doc in documents}) == len(documents)

    for doc in documents:
        assert doc.id
        assert doc.title
        assert doc.abstract

    # 4-7. Embedding and memory flow.
    clean_faiss_memory.add_documents(documents)
    matches = clean_faiss_memory.similarity_search("large language model applications", k=3)

    assert isinstance(matches, list)
    assert matches, "Expected similarity_search to return at least one match after adding documents"
    assert all(isinstance(doc, Document) for doc in matches)

    for doc in matches:
        assert doc.id
        assert doc.title
        assert doc.abstract

    # 8-9. Reset functionality.
    clean_faiss_memory.reset()
    after_reset = clean_faiss_memory.similarity_search("large language model applications", k=3)
    assert after_reset == []
