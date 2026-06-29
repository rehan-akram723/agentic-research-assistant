"""Simple arXiv search tool that returns results mapped to the Document schema.

This module implements the ArxivTool architecture with retry logic and full
Pydantic model mapping.
"""
from __future__ import annotations

from http import client
import time
from typing import List, Optional

from src.schemas.document import Document
from src.core.exceptions import RetrievalError
from src.core.logger import get_logger

logger = get_logger(__name__)

try:
    import arxiv
except ImportError:  # pragma: no cover
    arxiv = None


class ArxivTool:
    """
    A tool for querying the arXiv API to retrieve academic research papers.
    Maps raw arXiv results into the internal `Document` schema.
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def run(self, query: str, limit: Optional[int] = None) -> List[Document]:
        """Alias for search_arxiv to support generic agent tool interfaces."""
        return self.search_arxiv(query, limit)

    def search_arxiv(self, query: str, limit: Optional[int] = None) -> List[Document]:
        """
        Executes a search against the arXiv API using v2.1.0+ client syntax.
        """
        if arxiv is None:
            logger.error("arxiv package is not installed.")
            raise RetrievalError("arxiv package is required but not installed.")

        search_limit = limit if limit is not None else self.max_results
        
        # Initialize the client required for v2.1.0+
        client = arxiv.Client()
        search_query = arxiv.Search(query=query, max_results=search_limit)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use the client to execute the search
                results_generator = client.results(search_query)
                results: List[Document] = []
                for entry in results_generator:
                    results.append(self._map_to_document(entry))
                return results
            except Exception as e:
                logger.warning(f"arXiv search attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"arXiv search failed for query: '{query}'. Degrading to empty list.")
                    return []
                time.sleep(2 ** attempt)
        
        return []

    def _map_to_document(self, raw_result: 'arxiv.Result') -> Document:
        """
        Internal helper to transform an arxiv.Result object into our Pydantic Document.
        """
        doc_id = getattr(raw_result, "entry_id", None) or getattr(raw_result, "id", "")
        title = getattr(raw_result, "title", "").strip()
        authors = [getattr(a, "name", str(a)) for a in getattr(raw_result, "authors", [])]
        abstract = getattr(raw_result, "summary", getattr(raw_result, "abstract", "")).strip()
        url = getattr(raw_result, "pdf_url", None) or getattr(raw_result, "entry_id", "")
        
        published = getattr(raw_result, "published", None)
        published_str = str(published.date()) if hasattr(published, 'date') else str(published) if published else None

        return Document(
            id=str(doc_id),
            title=title,
            authors=authors,
            abstract=abstract,
            url=str(url),
            published_date=published_str,
            source="arxiv",
            relevance_score=None
        )


def search_arxiv(query: str, max_results: int = 5) -> List[Document]:
    """Helper function to execute a search using ArxivTool."""
    tool = ArxivTool(max_results=max_results)
    return tool.search_arxiv(query)