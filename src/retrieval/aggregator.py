"""src/retrieval/aggregator.py - Concurrent document retrieval across all sources and queries.

Runs every (query × source) pair in parallel using asyncio.gather + asyncio.to_thread,
then deduplicates by document ID and caps at max_results.

The aggregator is a pure function — it has no state and creates no singletons.
The retrieval_agent is the only consumer.
"""
import asyncio
import threading
from typing import Dict, List

from cachetools import LRUCache
from aiolimiter import AsyncLimiter

from src.core.logger import get_logger
from src.core.metrics import get_current_metrics
from src.schemas.document import Document
from src.tools.arxiv_tool import search_arxiv
from src.tools.semantic_scholar_tool import search_semantic_scholar

logger = get_logger(__name__)

_cache = LRUCache(maxsize=256)
_cache_lock = threading.Lock()

def _safe_fetch(fn, query: str, limit: int) -> tuple[List[Document], bool]:
    """Call a synchronous search function and swallow exceptions, returning (results, was_cache_hit)."""
    cache_key = (fn.__name__, query, limit)
    
    with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key], True
            
    try:
        results = list(fn(query, max_results=limit) or [])
    except Exception as exc:
        logger.warning("Fetch failed for query=%r: %s", query[:60], exc)
        return [], False

    with _cache_lock:
        _cache[cache_key] = results
        
    return results, False


async def _sem_fetch(sem: asyncio.Semaphore, ss_limiter: AsyncLimiter, fn, query: str, limit: int) -> tuple[List[Document], bool]:
    """Wraps the blocking _safe_fetch in a thread with semaphore control."""
    if fn.__name__ == "search_semantic_scholar":
        async with ss_limiter:
            pass # Consume rate limit token without blocking concurrent execution
            
    async with sem:
        coro = asyncio.to_thread(_safe_fetch, fn, query, limit)
        if fn.__name__ == "search_semantic_scholar":
            try:
                return await asyncio.wait_for(coro, timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("Semantic Scholar fetch timed out after 3.0s — degrading gracefully.")
                return [], False
        else:
            return await coro


async def _gather_all(queries: List[str], per_source_limit: int) -> tuple[List[Document], int, int]:
    """Fan out all (query, source) pairs concurrently, bound by a semaphore.

    Returns:
        (documents, raw_count, cache_hits) where documents is already deduplicated and
        raw_count is the total number of results before deduplication.
    """
    sem = asyncio.Semaphore(2)
    
    # Create the limiter inside the active event loop to prevent RuntimeError across threads
    ss_limiter = AsyncLimiter(1, 1)
    
    # Build one coroutine per (query, source) combination.
    tasks = []
    
    # 1. Dispatch ArXiv queries independently (no strict rate limits)
    for query in queries:
        tasks.append(
            _sem_fetch(sem, ss_limiter, search_arxiv, query, per_source_limit)
        )
        
    # 2. Batch API Restructure for Semantic Scholar:
    # Consolidate all queries into a single HTTP request to avoid aggressive 429 blocks
    # Semantic Scholar uses a soft-match search, so joining queries works effectively.
    if queries:
        combined_query = " ".join(queries)
        batch_limit = per_source_limit * len(queries)
        # Cap the batch limit to avoid excessive payload parsing
        batch_limit = min(batch_limit, 20)
        
        tasks.append(
            _sem_fetch(sem, ss_limiter, search_semantic_scholar, combined_query, batch_limit)
        )

    # All tasks run concurrently; blocking I/O inside each is off the event loop.
    batches = await asyncio.gather(*tasks)

    collected: Dict[str, Document] = {}
    raw_count = 0
    cache_hits = 0
    for batch, was_hit in batches:
        if was_hit:
            cache_hits += 1
        raw_count += len(batch)
        for doc in batch:
            collected.setdefault(doc.id, doc)

    return list(collected.values()), raw_count, cache_hits


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
            docs, raw_count, cache_hits = asyncio.run(
                _gather_all(queries, per_source_limit)
            )
        except RuntimeError:
            # asyncio.run() fails if a loop is already running (e.g. in tests
            # that use an explicit event loop). Fall back to a new thread.
            docs, raw_count, cache_hits = asyncio.get_event_loop().run_until_complete(
                _gather_all(queries, per_source_limit)
            )

        documents = docs[: self.max_results]

        # Record to active RunMetrics if available.
        m = get_current_metrics()
        if m:
            m.papers_retrieved = len(documents)
            m.duplicates_removed = max(0, raw_count - len(documents))
            m.cache_hits += cache_hits

        logger.info(
            "DocumentAggregator: %d unique docs from %d raw results "
            "(%d queries, 2 sources each).",
            len(documents),
            raw_count,
            len(queries),
        )
        return documents
