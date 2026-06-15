"""FAISS-backed memory tool used for similarity search over documents.

Implements a robust thread-safe Singleton wrapper around FAISS (when available) 
and falls back to brute-force numpy-based nearest neighbors if FAISS is not installed.
"""
from __future__ import annotations

import threading
import numpy as np
from typing import List, Optional

from src.schemas.document import Document
from src.services.embedding_service import EmbeddingService, embedding_service
from src.core.logger import get_logger
from src.core.exceptions import RetrievalError

logger = get_logger(__name__)

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class FAISSMemory:
    """
    A thread-safe Singleton vector database wrapper for storing and retrieving Documents
    based on semantic similarity.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, embedding_svc: Optional[EmbeddingService] = None, dimension: int = 384):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FAISSMemory, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, embedding_svc: Optional[EmbeddingService] = None, dimension: int = 384) -> None:
        if getattr(self, "_initialized", False):
            return
            
        self.embedding_service = embedding_svc or embedding_service
        self.dimension = dimension
        self.documents: List[Document] = []
        self.index = None
        self._vectors = None
        
        if faiss is not None:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            logger.warning("FAISS is not installed. Similarity search will use exact numpy fallback.")

        self._initialized = True

    def add_documents(self, documents: List[Document]) -> None:
        """
        Embeds and adds a batch of Documents to the FAISS index.

        Args:
            documents (List[Document]): The documents to store.
        """
        if not documents:
            return

        texts = [doc.abstract if doc.abstract else doc.title for doc in documents]
        
        try:
            embeddings = self.embedding_service.embed(texts)
            vecs = np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise RetrievalError(f"Embedding failed during add_documents: {e}") from e

        if vecs.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {vecs.shape[1]}")

        with self._lock:
            self.documents.extend(documents)
            if self.index is not None:
                self.index.add(vecs)
            else:
                # Fallback numpy store
                if self._vectors is None:
                    self._vectors = vecs.copy()
                else:
                    self._vectors = np.vstack([self._vectors, vecs])
            logger.info(f"Added {len(documents)} documents to memory.")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Searches the FAISS index for the most semantically similar documents.

        Args:
            query (str): The search string.
            k (int): Number of top results to return.

        Returns:
            List[Document]: A list of the matching Document objects.
        """
        with self._lock:
            num_docs = len(self.documents)
            
        if num_docs == 0:
            logger.debug("FAISS memory is empty. Returning no results.")
            return []

        try:
            query_embedding = self.embedding_service.embed([query])
            q_vec = np.array(query_embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RetrievalError(f"Embedding failed during similarity_search: {e}") from e

        search_k = min(k, num_docs)

        with self._lock:
            if self.index is not None:
                # index.search returns D (distances) and I (indices)
                distances, indices_arr = self.index.search(q_vec, search_k)
                indices = indices_arr[0].tolist()
            else:
                if self._vectors is None:
                    return []
                # Fallback L2 distance
                diff = self._vectors - q_vec
                distances_arr = np.linalg.norm(diff, axis=1)
                indices = np.argsort(distances_arr)[:search_k].tolist()

            results: List[Document] = []
            for idx in indices:
                if 0 <= idx < len(self.documents):
                    results.append(self.documents[idx])
            
            return results

    def clear(self) -> None:
        """Resets the FAISS index and clears the document store."""
        with self._lock:
            self.documents = []
            if self.index is not None:
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                self._vectors = None
            logger.info("FAISS memory cleared.")

    def reset(self) -> None:
        """Alias for clear for backwards compatibility."""
        self.clear()


faiss_memory = FAISSMemory()
