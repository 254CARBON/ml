"""Dual-write vector store for migration between backends."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import structlog

from .base import VectorStore

logger = structlog.get_logger("vector_store.dual_write")


class DualWriteVectorStore(VectorStore):
    """Vector store that writes to both primary and secondary stores."""
    
    def __init__(
        self,
        primary_store: VectorStore,
        secondary_store: VectorStore,
        read_from_secondary: bool = False,
        write_to_secondary: bool = True
    ):
        """Initialize dual-write vector store.
        
        Args:
            primary_store: Primary vector store (e.g., PgVector)
            secondary_store: Secondary vector store (e.g., OpenSearch)
            read_from_secondary: Whether to read from secondary store
            write_to_secondary: Whether to write to secondary store
        """
        self.primary_store = primary_store
        self.secondary_store = secondary_store
        self.read_from_secondary = read_from_secondary
        self.write_to_secondary = write_to_secondary
    
    async def store_embedding(
        self,
        entity_type: str,
        entity_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        model_version: str = "default",
        tenant_id: str = "default"
    ) -> bool:
        """Store embedding in both stores."""
        try:
            # Always write to primary
            primary_success = await self.primary_store.store_embedding(
                entity_type=entity_type,
                entity_id=entity_id,
                vector=vector,
                metadata=metadata,
                model_version=model_version,
                tenant_id=tenant_id
            )
            
            # Write to secondary if enabled
            secondary_success = True
            if self.write_to_secondary:
                try:
                    secondary_success = await self.secondary_store.store_embedding(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        vector=vector,
                        metadata=metadata,
                        model_version=model_version,
                        tenant_id=tenant_id
                    )
                except Exception as e:
                    logger.warning(
                        "Secondary store write failed",
                        entity_type=entity_type,
                        entity_id=entity_id,
                        error=str(e)
                    )
                    secondary_success = False
            
            # Return success if primary succeeded
            return primary_success
            
        except Exception as e:
            logger.error(
                "Dual write failed",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(e)
            )
            return False
    
    async def get_embedding(
        self,
        entity_type: str,
        entity_id: str,
        model_version: str = "default",
        tenant_id: str = "default"
    ) -> Optional[np.ndarray]:
        """Get embedding from primary or secondary store."""
        try:
            # Read from secondary if enabled, otherwise primary
            if self.read_from_secondary:
                try:
                    result = await self.secondary_store.get_embedding(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        model_version=model_version,
                        tenant_id=tenant_id
                    )
                    if result is not None:
                        return result
                except Exception as e:
                    logger.warning(
                        "Secondary store read failed, falling back to primary",
                        entity_type=entity_type,
                        entity_id=entity_id,
                        error=str(e)
                    )
            
            # Read from primary
            return await self.primary_store.get_embedding(
                entity_type=entity_type,
                entity_id=entity_id,
                model_version=model_version,
                tenant_id=tenant_id
            )
            
        except Exception as e:
            logger.error(
                "Dual read failed",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(e)
            )
            return None
    
    async def search_similar(
        self,
        query_vector: np.ndarray,
        entity_type: Optional[str] = None,
        tenant_id: str = "default",
        limit: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        try:
            # Read from secondary if enabled, otherwise primary
            if self.read_from_secondary:
                try:
                    return await self.secondary_store.search_similar(
                        query_vector=query_vector,
                        entity_type=entity_type,
                        tenant_id=tenant_id,
                        limit=limit,
                        similarity_threshold=similarity_threshold
                    )
                except Exception as e:
                    logger.warning(
                        "Secondary store search failed, falling back to primary",
                        error=str(e)
                    )
            
            # Search primary
            return await self.primary_store.search_similar(
                query_vector=query_vector,
                entity_type=entity_type,
                tenant_id=tenant_id,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
        except Exception as e:
            logger.error("Dual search failed", error=str(e))
            return []
    
    async def delete_embedding(
        self,
        entity_type: str,
        entity_id: str,
        model_version: str = "default",
        tenant_id: str = "default"
    ) -> bool:
        """Delete embedding from both stores."""
        try:
            # Delete from primary
            primary_success = await self.primary_store.delete_embedding(
                entity_type=entity_type,
                entity_id=entity_id,
                model_version=model_version,
                tenant_id=tenant_id
            )
            
            # Delete from secondary if enabled
            secondary_success = True
            if self.write_to_secondary:
                try:
                    secondary_success = await self.secondary_store.delete_embedding(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        model_version=model_version,
                        tenant_id=tenant_id
                    )
                except Exception as e:
                    logger.warning(
                        "Secondary store delete failed",
                        entity_type=entity_type,
                        entity_id=entity_id,
                        error=str(e)
                    )
                    secondary_success = False
            
            return primary_success
            
        except Exception as e:
            logger.error(
                "Dual delete failed",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(e)
            )
            return False
    
    async def batch_store_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        tenant_id: str = "default"
    ) -> int:
        """Batch store embeddings in both stores."""
        try:
            # Store in primary
            primary_count = await self.primary_store.batch_store_embeddings(
                embeddings=embeddings,
                tenant_id=tenant_id
            )
            
            # Store in secondary if enabled
            if self.write_to_secondary:
                try:
                    await self.secondary_store.batch_store_embeddings(
                        embeddings=embeddings,
                        tenant_id=tenant_id
                    )
                except Exception as e:
                    logger.warning(
                        "Secondary store batch write failed",
                        count=len(embeddings),
                        error=str(e)
                    )
            
            return primary_count
            
        except Exception as e:
            logger.error("Dual batch store failed", count=len(embeddings), error=str(e))
            return 0
    
    async def get_embedding_count(
        self,
        entity_type: Optional[str] = None,
        tenant_id: str = "default"
    ) -> int:
        """Get embedding count from primary store."""
        return await self.primary_store.get_embedding_count(
            entity_type=entity_type,
            tenant_id=tenant_id
        )
    
    async def health_check(self) -> bool:
        """Check health of both stores."""
        try:
            primary_healthy = await self.primary_store.health_check()
            
            if self.write_to_secondary or self.read_from_secondary:
                secondary_healthy = await self.secondary_store.health_check()
                return primary_healthy and secondary_healthy
            
            return primary_healthy
            
        except Exception as e:
            logger.error("Dual health check failed", error=str(e))
            return False
    
    async def close(self) -> None:
        """Close both stores."""
        try:
            if hasattr(self.primary_store, 'close'):
                await self.primary_store.close()
            
            if hasattr(self.secondary_store, 'close'):
                await self.secondary_store.close()
                
            logger.info("Dual-write vector store closed")
            
        except Exception as e:
            logger.error("Failed to close dual-write vector store", error=str(e))


def create_dual_write_store(
    primary_store: VectorStore,
    secondary_store: VectorStore,
    read_from_secondary: bool = False,
    write_to_secondary: bool = True
) -> DualWriteVectorStore:
    """Create a dual-write vector store."""
    return DualWriteVectorStore(
        primary_store=primary_store,
        secondary_store=secondary_store,
        read_from_secondary=read_from_secondary,
        write_to_secondary=write_to_secondary
    )
