"""src/retrieval/aggregator.py - Concurrent document retrieval across all sources and queries.

Runs every (query × source) pair in parallel using asyncio.gather + asyncio.to_thread,
then deduplicates by document ID and caps at max_results.

The aggregator is a pure function — it has no state and creates no singletons.
The retrieval_agent is the only consumer.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List

from src.core.logger import get_logger
from src.core.metrics import get_current_metrics
from src.schemas.document import Document
from src.tools.arxiv_tool import search_arxiv
from src.tools.semantic_scholar_tool import search_semantic_scholar

logger = get_logger(__name__)


def _safe_fetch(fn, query: str, limit: int) -> List[Document]:
    """Call a synchronous search function and swallow exceptions, returning []."""
    try:
        return list(fn(query, max_results=limit) or [])
    except Exception as exc:
        logger.warning("Fetch failed for query=%r: %s", query[:60], exc)
        return []


async def _gather_all(queries: List[str], per_source_limit: int) -> tuple[List[Document], int]:
    """Fan out all (query, source) pairs concurrently.

    Returns:
        (documents, raw_count) where documents is already deduplicated and
        raw_count is the total number of results before deduplication.
    """
    # Build one coroutine per (query, source) combination.
    tasks = []
    for query in queries:
        tasks.append(
            asyncio.to_thread(_safe_fetch, search_arxiv, query, per_source_limit)
        )
        tasks.append(
            asyncio.to_thread(_safe_fetch, search_semantic_scholar, query, per_source_limit)
        )

    # All tasks run concurrently; blocking I/O inside each is off the event loop.
    batches: List[List[Document]] = await asyncio.gather(*tasks)

    collected: Dict[str, Document] = {}
    raw_count = 0
    for batch in batches:
        raw_count += len(batch)
        for doc in batch:
            collected.setdefault(doc.id, doc)

    return list(collected.values()), raw_count


class DocumentAggregator:
    """Concurrent multi-source document retrieval with deduplication.

    Usage:
        aggregator = DocumentAggregator(max_results=10)
        documents = aggregator.run(queries)
    """

    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results

    def run(self, queries: List[str]) -> List[Document]:
        """Execute concurrent retrieval and return at most max_results documents.

        Drives its own event loop via asyncio.run() so it is safe to call from
        any synchronous context (including inside asyncio.to_thread).
        """
        if not queries:
            return []

        per_source_limit = max(1, self.max_results // 2)

        try:
            docs, raw_count = asyncio.run(
                _gather_all(queries, per_source_limit)
            )
        except RuntimeError:
            # asyncio.run() fails if a loop is already running (e.g. in tests
            # that use an explicit event loop). Fall back to a new thread.
            docs, raw_count = asyncio.get_event_loop().run_until_complete(
                _gather_all(queries, per_source_limit)
            )

        documents = docs[: self.max_results]

        # Record to active RunMetrics if available.
        m = get_current_metrics()
        if m:
            m.papers_retrieved = len(documents)
            m.duplicates_removed = max(0, raw_count - len(documents))

        logger.info(
            "DocumentAggregator: %d unique docs from %d raw results "
            "(%d queries, 2 sources each).",
            len(documents),
            raw_count,
            len(queries),
        )
        return documents
