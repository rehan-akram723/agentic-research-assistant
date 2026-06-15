"""Semantic Scholar search wrapper returning Document-compatible schemas.

Uses the public Semantic Scholar Graph API. Features retry mechanisms for rate limits
and robust mapping to the core Document schema.
"""
from __future__ import annotations

import time
import requests
from typing import List, Optional

from src.schemas.document import Document
from src.core.exceptions import RetrievalError
from src.core.logger import get_logger

logger = get_logger(__name__)


class SemanticScholarTool:
    """
    A tool for querying the Semantic Scholar API for academic papers.
    Handles HTTP requests, retries, and mapping to the Document schema.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "paperId,title,authors,abstract,url,year"

    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        self.api_key = api_key
        self.max_results = max_results

    def run(self, query: str, limit: Optional[int] = None) -> List[Document]:
        """Alias for search_semantic_scholar to support generic agent tool interfaces."""
        return self.search_semantic_scholar(query, limit)

    def search_semantic_scholar(self, query: str, limit: Optional[int] = None) -> List[Document]:
        """
        Executes a search against Semantic Scholar.

        Args:
            query (str): The search query string.
            limit (int, optional): Max results to retrieve. Overrides self.max_results if provided.

        Returns:
            List[Document]: A list of mapped Document objects.

        Raises:
            RetrievalError: If the API responds with an unrecoverable error after retries.
        """
        search_limit = limit if limit is not None else self.max_results
        params = {
            "query": query,
            "limit": search_limit,
            "fields": self.FIELDS
        }
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=10)
                
                if resp.status_code == 429:
                    logger.warning("Semantic Scholar API rate limit exceeded. Retrying...")
                    time.sleep(2 ** (attempt + 1))
                    continue
                    
                resp.raise_for_status()
                data = resp.json()
                
                hits = data.get("data", [])
                results: List[Document] = []
                for item in hits:
                    results.append(self._map_to_document(item))
                return results
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Semantic Scholar search attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Semantic Scholar search completely failed for query: '{query}'")
                    raise RetrievalError(f"Semantic Scholar search failed after {max_retries} attempts: {e}") from e
                time.sleep(2 ** attempt)

        return []

    def _map_to_document(self, raw_dict: dict) -> Document:
        """
        Internal helper to map a single Semantic Scholar JSON response dict to the Document schema.
        """
        pid = raw_dict.get("paperId") or raw_dict.get("id") or ""
        title = raw_dict.get("title", "").strip()
        abstract = raw_dict.get("abstract", "") or ""
        
        raw_authors = raw_dict.get("authors", [])
        authors = [a.get("name") for a in raw_authors if isinstance(a, dict) and a.get("name")]
        
        url_field = raw_dict.get("url") or ""
        year = raw_dict.get("year")

        return Document(
            id=str(pid),
            title=title,
            authors=authors,
            abstract=abstract,
            url=str(url_field),
            published_date=str(year) if year is not None else None,
            source="semantic_scholar",
            relevance_score=None
        )


def search_semantic_scholar(query: str, max_results: int = 5) -> List[Document]:
    """Helper function to execute a search using SemanticScholarTool."""
    tool = SemanticScholarTool(max_results=max_results)
    return tool.search_semantic_scholar(query)