"""Embedding service — sentence-transformers wrapper with HF_TOKEN support.

Falls back to a deterministic character n-gram embedding when the package
is unavailable so tests and CI remain lightweight.
"""
from __future__ import annotations

import os
import time
import threading
from typing import List

import numpy as np

from src.core.config import settings
from src.core.logger import get_logger
from src.core.metrics import get_current_metrics

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def _apply_hf_token() -> None:
    """Set HF_TOKEN env var so sentence-transformers / huggingface_hub pick it up."""
    token = settings.hf_token
    if not token:
        return
    # huggingface_hub reads HUGGING_FACE_HUB_TOKEN (legacy) and HF_TOKEN (current).
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    logger.info("HF_TOKEN applied to environment (first %s…).", token[:8])


class EmbeddingService:
    """Thread-safe singleton for generating dense vector embeddings.

    Automatically configures the Hugging Face token before loading the model,
    enabling authenticated downloads of gated or private models.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name

        if SentenceTransformer is not None:
            _apply_hf_token()  # must run before model download attempt
            try:
                self.model = SentenceTransformer(model_name)
                logger.info("Loaded SentenceTransformer model: %s", model_name)
            except Exception as exc:
                logger.error("Failed to load model %s: %s", model_name, exc)
                raise RuntimeError(f"Could not load embedding model {model_name}: {exc}") from exc
        else:
            logger.warning("sentence-transformers not installed — using fallback embedder.")
            self.model = None

        self._initialized = True

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return dense vector embeddings for each text in *texts*."""
        if not texts:
            return []

        safe = [t if t.strip() else " " for t in texts]

        if self.model is not None:
            with self._lock:
                try:
                    t0 = time.perf_counter()
                    vectors = self.model.encode(safe)
                    elapsed = time.perf_counter() - t0
                    m = get_current_metrics()
                    if m:
                        m.record_embedding(elapsed)
                    return vectors.tolist()
                except Exception as exc:
                    logger.error("Embedding encode failed: %s", exc)
                    raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

        # Fallback: character n-gram hashing, 384-dim (matches all-MiniLM-L6-v2).
        out: List[List[float]] = []
        t0 = time.perf_counter()
        for t in safe:
            arr = np.zeros(384, dtype=float)
            for i, ch in enumerate(t[:1024]):
                arr[i % 384] += ord(ch)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            out.append(arr.tolist())
        m = get_current_metrics()
        if m:
            m.record_embedding(time.perf_counter() - t0)
        return out


embedding_service = EmbeddingService()
