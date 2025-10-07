"""Base vector store interface.

Defines the abstract contract our services depend on, independent of the
backing implementation (PgVector, Pinecone, OpenSearch, etc.).

All methods are asynchronous to support high‑throughput services.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Implementations should ensure idempotent upserts, consistent typing of
    metadata, and predictable similarity semantics (cosine or inner product).
    """
    
    @abstractmethod
    async def store_embedding(
        self,
        entity_type: str,
        entity_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        model_version: str = "default",
        tenant_id: str = "default"
    ) -> bool:
        """Store an embedding vector.

        Returns
        - ``True`` on success, ``False`` on recoverable failure
        """
        pass
    
    @abstractmethod
    async def get_embedding(
        self,
        entity_type: str,
        entity_id: str,
        model_version: str = "default",
        tenant_id: str = "default"
    ) -> Optional[np.ndarray]:
        """Get an embedding vector.

        Returns
        - ``np.ndarray`` when found, else ``None``
        """
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        query_vector: np.ndarray,
        entity_type: Optional[str] = None,
        tenant_id: str = "default",
        limit: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Search for similar vectors.

        Returns
        - A list of tuples ``(entity_type, entity_id, similarity, metadata)``
        sorted by descending similarity (implementation defines the metric).
        """
        pass
    
    @abstractmethod
    async def delete_embedding(
        self,
        entity_type: str,
        entity_id: str,
        model_version: str = "default",
        tenant_id: str = "default"
    ) -> bool:
        """Delete an embedding vector.

        Returns ``True`` if an item was deleted, else ``False``.
        """
        pass
    
    @abstractmethod
    async def batch_store_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        tenant_id: str = "default"
    ) -> int:
        """Store multiple embeddings in batch.

        Returns the number of embeddings successfully stored.
        """
        pass
    
    @abstractmethod
    async def get_embedding_count(
        self,
        entity_type: Optional[str] = None,
        tenant_id: str = "default"
    ) -> int:
        """Get count of stored embeddings."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        pass


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Connection error to vector store."""
    pass


class VectorStoreQueryError(VectorStoreError):
    """Query error in vector store."""
    pass


class VectorStoreNotFoundError(VectorStoreError):
    """Vector not found in store."""
    pass
