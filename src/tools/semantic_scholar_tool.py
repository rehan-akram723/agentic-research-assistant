"""Semantic Scholar search wrapper returning Document-compatible schemas.

Uses the public Semantic Scholar Graph API (no auth required for ≤100 RPM).

Throttling strategy
-------------------
* **Proactive** — ``time.sleep(1.1)`` is called *before every HTTP attempt* to
  ensure we never exceed 1 request per second, regardless of retry count.
* **Reactive** — If a 429 is still received (e.g. in multi-threaded scenarios),
  exponential back-off is applied (5 s × attempt) before the next retry.

This dual-layer approach prevents 429 warnings in normal single-threaded use
while still recovering gracefully if the limit is hit.
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
            # ----------------------------------------------------------------
            # Proactive throttle — enforce ≤1 RPS *before* every HTTP request.
            # At 1.1 s per call we stay well inside the 1 RPS free-tier limit
            # without needing to track wall-clock state across calls.
            # ----------------------------------------------------------------
            logger.debug(
                "Semantic Scholar: sleeping 1.1 s before attempt %d (query=%r)",
                attempt + 1,
                query[:60],
            )
            time.sleep(1.1)

            try:
                resp = requests.get(
                    self.BASE_URL, params=params, headers=headers, timeout=15
                )

                if resp.status_code == 429:
                    # Reactive back-off: 5 s, 10 s, 15 s for attempts 0, 1, 2.
                    backoff = 5 * (attempt + 1)
                    logger.warning(
                        "Semantic Scholar API rate limit (429) on attempt %d — "
                        "backing off %d s before retry.",
                        attempt + 1,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue

                resp.raise_for_status()
                data = resp.json()

                hits = data.get("data", [])
                results: List[Document] = [self._map_to_document(item) for item in hits]
                logger.debug(
                    "Semantic Scholar: returned %d results for query %r (attempt %d).",
                    len(results),
                    query[:60],
                    attempt + 1,
                )
                return results

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Semantic Scholar search attempt %d failed: %s", attempt + 1, exc
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "Semantic Scholar search exhausted %d retries for query %r — "
                        "degrading to empty list.",
                        max_retries,
                        query[:60],
                    )
                    return []
                # Short jittered sleep before the next attempt (separate from the
                # proactive 1.1 s above, which will also fire at the top of the loop).
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