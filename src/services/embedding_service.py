"""Embedding service wrapper around sentence-transformers.

Provides a robust, thread-safe `EmbeddingService` that returns vector embeddings for
text inputs. Falls back to a deterministic, low-quality embedding when the
SentenceTransformer package is unavailable to keep tests and CI lightweight.
"""
from __future__ import annotations

import threading
from typing import List
import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


class EmbeddingService:
    """
    Thread-safe service for generating dense vector embeddings using SentenceTransformers.
    Designed as a module-level singleton.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if getattr(self, "_initialized", False):
            return
            
        self.model_name = model_name
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded SentenceTransformer model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
                raise RuntimeError(f"Could not load embedding model {model_name}: {e}") from e
        else:
            logger.warning("SentenceTransformer not installed. Using fallback deterministic embedder.")
            self.model = None
            
        self._initialized = True

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of vector embeddings for `texts`.

        Args:
            texts (List[str]): The batch of strings to embed.

        Returns:
            List[List[float]]: A list of vector representations.
        """
        if not texts:
            return []

        # Safely handle empty strings to avoid downstream errors
        safe_texts = [text if text.strip() else " " for text in texts]

        if self.model is not None:
            with self._lock:
                try:
                    vectors = self.model.encode(safe_texts)
                    return vectors.tolist()
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    raise RuntimeError(f"Failed to generate embeddings: {e}") from e

        # Fallback: use normalized character n-gram hashing to produce stable vectors
        # Dimension is 384 to match 'all-MiniLM-L6-v2' ensuring compatibility with FAISS
        out: List[List[float]] = []
        for t in safe_texts:
            arr = np.zeros(384, dtype=float)
            for i, ch in enumerate(t[:1024]):
                arr[i % 384] += ord(ch)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            out.append(arr.tolist())
        return out


embedding_service = EmbeddingService()
