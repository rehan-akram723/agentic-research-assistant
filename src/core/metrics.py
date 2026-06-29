"""src/core/metrics.py - Per-request metrics collection.

One RunMetrics instance is created per research request and threaded
through the pipeline via a lightweight context variable so every agent
and tool can record to it without explicit argument passing.
"""
from __future__ import annotations

import time
import functools
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# OpenRouter pricing (USD per 1k tokens) - update as model pricing changes.
_COST_PER_1K: dict[str, tuple[float, float]] = {
    # model_id: (input_cost, output_cost)
    "openai/gpt-4.1-nano": (0.0001, 0.0004),
    "openai/gpt-4o-mini":  (0.00015, 0.0006),
    "openai/gpt-4o":       (0.005,   0.015),
}
_DEFAULT_COST = (0.001, 0.003)  # conservative fallback


@dataclass
class RunMetrics:
    """All instrumentation data captured during one research pipeline run."""
    papers_retrieved: int = 0
    duplicates_removed: int = 0

    embedding_time_seconds: float = 0.0

    llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    api_failures: int = 0

    # Per-agent latency samples (seconds)
    agent_latencies: dict[str, list[float]] = field(default_factory=dict)

    cache_hits: int = 0

    # Lock for safe cross-thread writes (summarizer runs sequentially, but
    # embedding_service may be called from multiple threads in future).
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool = True,
    ) -> None:
        with self._lock:
            if not success:
                self.api_failures += 1
                return
            self.llm_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            inp, out = _COST_PER_1K.get(model, _DEFAULT_COST)
            self.estimated_cost_usd += (
                prompt_tokens / 1000 * inp + completion_tokens / 1000 * out
            )

    def record_embedding(self, elapsed: float) -> None:
        with self._lock:
            self.embedding_time_seconds += elapsed

    def record_agent_latency(self, agent: str, elapsed: float) -> None:
        with self._lock:
            self.agent_latencies.setdefault(agent, []).append(round(elapsed, 4))

    def avg_summary_latency(self) -> float | None:
        samples = self.agent_latencies.get("summarizer", [])
        return round(sum(samples) / len(samples), 4) if samples else None

    def to_dict(self) -> dict[str, Any]:
        """Serialisable snapshot — safe to embed in API responses / logs."""
        return {
            "papers_retrieved": self.papers_retrieved,
            "duplicates_removed": self.duplicates_removed,
            "embedding_time_seconds": round(self.embedding_time_seconds, 4),
            "llm_calls": self.llm_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "api_failures": self.api_failures,
            "avg_summary_latency_seconds": self.avg_summary_latency(),
            "cache_hits": self.cache_hits,
            "agent_latencies": {
                k: {"calls": len(v), "avg": round(sum(v) / len(v), 4)}
                for k, v in self.agent_latencies.items()
            },
        }


# Context variable — holds the active RunMetrics for the current async task.
# asyncio.to_thread() copies the context automatically, so agents running in
# the thread pool see the same RunMetrics instance as their parent coroutine.
_active_metrics: ContextVar[RunMetrics | None] = ContextVar(
    "_active_metrics", default=None
)


def get_current_metrics() -> RunMetrics | None:
    """Return the RunMetrics bound to the current execution context, or None."""
    return _active_metrics.get()


def set_current_metrics(m: RunMetrics) -> None:
    """Bind a RunMetrics to the current execution context."""
    _active_metrics.set(m)


def track_latency(agent_name: str) -> Callable[[F], F]:
    """Decorator that records wall-clock time for a synchronous agent function.

    Usage:
        @track_latency("summarizer")
        def summarizer_agent(state): ...
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                m = get_current_metrics()
                if m is not None:
                    m.record_agent_latency(agent_name, elapsed)
        return wrapper  # type: ignore[return-value]
    return decorator
