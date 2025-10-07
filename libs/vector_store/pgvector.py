"""PgVector implementation of vector store.

This implementation stores vectors in PostgreSQL using the pgvector extension.
Cosine similarity is computed using the ``<=>`` operator and converted to a
``similarity`` score in ``[0, 1]`` for consistency with typical ranking logic.

Connection management
- A shared asyncpg pool is created on demand and reused across calls
- Queries are funneled through ``_execute_query`` for uniform error handling
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import asyncpg
import numpy as np
import structlog

try:
    from pgvector.asyncpg import register_vector as register_pgvector_codec
except ImportError:  # pragma: no cover - optional dependency guarded at runtime
    register_pgvector_codec = None

from asyncpg import Connection, Pool

from .base import (
    VectorStore,
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreQueryError,
    VectorStoreNotFoundError,
)

logger = structlog.get_logger("vector_store.pgvector")


class PgVectorStore(VectorStore):
    """PgVector implementation of vector store."""
    
    def __init__(
        self,
        dsn: str,
        pool_size: int = 10,
        max_queries: int = 50000,
        command_timeout: int = 60,
        vector_dimension: Optional[int] = None,
    ):
        """Configure a PgVector-backed vector store.

        Parameters
        - dsn: PostgreSQL DSN including database and credentials
        - pool_size: Max size of asyncpg connection pool
        - max_queries: Queries per connection before recycling
        - command_timeout: Seconds to allow per DB command
        - vector_dimension: Expected dimensionality for stored vectors
        """
        self.dsn = dsn
        self.pool_size = pool_size
        self.max_queries = max_queries
        self.command_timeout = command_timeout
        self.vector_dimension = vector_dimension
        self._pool: Optional[Pool] = None
    
    async def _init_connection(self, conn: Connection) -> None:
        """Register pgvector codec for asyncpg connections."""
        if register_pgvector_codec:
            await register_pgvector_codec(conn)
            return
        
        # Fallback to a text codec if pgvector.asyncpg is unavailable
        def _encoder(value: Iterable[float]) -> str:
            array = np.asarray(value, dtype=np.float32)
            return "[" + ",".join(str(float(v)) for v in array.tolist()) + "]"
        
        def _decoder(value: str) -> np.ndarray:
            if value is None:
                return value
            stripped = value.strip("[]")
            if not stripped:
                return np.array([], dtype=np.float32)
            return np.fromstring(stripped, sep=",", dtype=np.float32)
        
        await conn.set_type_codec(
            "vector",
            encoder=_encoder,
            decoder=_decoder,
            schema="pg_catalog",
            format="text",
        )
    
    async def _get_pool(self) -> Pool:
        """Get or create connection pool.

        Lazily initializes an asyncpg pool so callers don't pay startup cost
        unless/until they make a call that requires the database.
        """
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    self.dsn,
                    min_size=1,
                    max_size=self.pool_size,
                    max_queries=self.max_queries,
                    command_timeout=self.command_timeout,
                    init=self._init_connection,
                )
                logger.info("Created PgVector connection pool", pool_size=self.pool_size)
            except Exception as e:
                logger.error("Failed to create PgVector connection pool", error=str(e))
                raise VectorStoreConnectionError(f"Failed to create connection pool: {e}")
        
        return self._pool
    
    async def _execute_query(
        self,
        query: str,
        *args: Any,
        fetch: bool = False,
        fetch_one: bool = False
    ) -> Any:
        """Execute a query with error handling.

        The ``fetch``/``fetch_one`` flags control how results are retrieved.
        All failures are wrapped in ``VectorStoreQueryError`` for consistency.
        """
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                if fetch_one:
                    result = await conn.fetchrow(query, *args)
                elif fetch:
                    result = await conn.fetch(query, *args)
                else:
                    result = await conn.execute(query, *args)
                return result
        except Exception as e:
            logger.error("Query execution failed", query=query, error=str(e))
            raise VectorStoreQueryError(f"Query failed: {e}")
    
    async def store_embedding(
        self,
        entity_type: str,
        entity_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        model_version: str = "default",
        tenant_id: str = "default"
    ) -> bool:
        """Store an embedding vector."""
        try:
            vector_array = self._ensure_vector_dimension(vector)
            
            query = """
                INSERT INTO embeddings (entity_type, entity_id, vector, meta, model_version, tenant_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (entity_type, entity_id, model_version, tenant_id)
                DO UPDATE SET
                    vector = EXCLUDED.vector,
                    meta = EXCLUDED.meta,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            await self._execute_query(
                query,
                entity_type,
                entity_id,
                vector_array,
                metadata,
                model_version,
                tenant_id
            )
            
            logger.info(
                "Stored embedding",
                entity_type=entity_type,
                entity_id=entity_id,
                model_version=model_version,
                tenant_id=tenant_id
            )
            return True
            
        except Exception as e:
            logger.error(
                "Failed to store embedding",
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
        """Get an embedding vector."""
        try:
            query = """
                SELECT vector FROM embeddings
                WHERE entity_type = $1 AND entity_id = $2 AND model_version = $3 AND tenant_id = $4
            """
            
            result = await self._execute_query(
                query,
                entity_type,
                entity_id,
                model_version,
                tenant_id,
                fetch_one=True
            )
            
            if result:
                vector_array = np.asarray(result["vector"], dtype=np.float32)
                return vector_array
            else:
                logger.warning(
                    "Embedding not found",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    model_version=model_version,
                    tenant_id=tenant_id
                )
                return None
                
        except Exception as e:
            logger.error(
                "Failed to get embedding",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(e)
            )
            raise VectorStoreQueryError(f"Failed to get embedding: {e}")
    
    async def search_similar(
        self,
        query_vector: np.ndarray,
        entity_type: Optional[str] = None,
        tenant_id: str = "default",
        limit: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Search for similar vectors using cosine similarity."""
        try:
            vector_array = self._ensure_vector_dimension(query_vector)
            
            # Build query with optional entity_type filter
            if entity_type:
                query = """
                    SELECT entity_type, entity_id, 
                           1 - (vector <=> $1) as similarity,
                           meta
                    FROM embeddings
                    WHERE tenant_id = $2 AND entity_type = $3
                    ORDER BY vector <=> $1
                    LIMIT $4
                """
                args = (vector_array, tenant_id, entity_type, limit)
            else:
                query = """
                    SELECT entity_type, entity_id,
                           1 - (vector <=> $1) as similarity,
                           meta
                    FROM embeddings
                    WHERE tenant_id = $2
                    ORDER BY vector <=> $1
                    LIMIT $3
                """
                args = (vector_array, tenant_id, limit)
            
            results = await self._execute_query(query, *args, fetch=True)
            
            # Filter by similarity threshold and return results
            similar_vectors = []
            for row in results:
                similarity = float(row['similarity'])
                if similarity >= similarity_threshold:
                    similar_vectors.append((
                        row['entity_type'],
                        row['entity_id'],
                        similarity,
                        row['meta'] or {}
                    ))
            
            logger.info(
                "Vector similarity search completed",
                query_vector_dim=len(vector_array),
                entity_type=entity_type,
                tenant_id=tenant_id,
                limit=limit,
                results_count=len(similar_vectors)
            )
            
            return similar_vectors
            
        except Exception as e:
            logger.error(
                "Vector similarity search failed",
                error=str(e)
            )
            raise VectorStoreQueryError(f"Similarity search failed: {e}")
    
    async def delete_embedding(
        self,
        entity_type: str,
        entity_id: str,
        model_version: str = "default",
        tenant_id: str = "default"
    ) -> bool:
        """Delete an embedding vector."""
        try:
            query = """
                DELETE FROM embeddings
                WHERE entity_type = $1 AND entity_id = $2 AND model_version = $3 AND tenant_id = $4
            """
            
            result = await self._execute_query(
                query,
                entity_type,
                entity_id,
                model_version,
                tenant_id
            )
            
            # Check if any rows were affected
            deleted = result.split()[-1] == "1"
            
            if deleted:
                logger.info(
                    "Deleted embedding",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    model_version=model_version,
                    tenant_id=tenant_id
                )
            else:
                logger.warning(
                    "Embedding not found for deletion",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    model_version=model_version,
                    tenant_id=tenant_id
                )
            
            return deleted
            
        except Exception as e:
            logger.error(
                "Failed to delete embedding",
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
        """Store multiple embeddings in batch."""
        if not embeddings:
            return 0
        
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Prepare batch data – we perform an upsert to keep entries fresh
                batch_data = []
                for emb in embeddings:
                    vector_array = self._ensure_vector_dimension(emb['vector'])
                    batch_data.append((
                        emb['entity_type'],
                        emb['entity_id'],
                        vector_array,
                        emb.get('metadata'),
                        emb.get('model_version', 'default'),
                        tenant_id
                    ))
                
                # Execute batch insert
                query = """
                    INSERT INTO embeddings (entity_type, entity_id, vector, meta, model_version, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (entity_type, entity_id, model_version, tenant_id)
                    DO UPDATE SET
                        vector = EXCLUDED.vector,
                        meta = EXCLUDED.meta,
                        updated_at = CURRENT_TIMESTAMP
                """
                
                result = await conn.executemany(query, batch_data)
                stored_count = len(batch_data)
                
                logger.info(
                    "Batch stored embeddings",
                    count=stored_count,
                    tenant_id=tenant_id
                )
                
                return stored_count
                
        except Exception as e:
            logger.error(
                "Batch store embeddings failed",
                count=len(embeddings),
                error=str(e)
            )
            raise VectorStoreQueryError(f"Batch store failed: {e}")
    
    async def get_embedding_count(
        self,
        entity_type: Optional[str] = None,
        tenant_id: str = "default"
    ) -> int:
        """Get count of stored embeddings."""
        try:
            if entity_type:
                query = """
                    SELECT COUNT(*) FROM embeddings
                    WHERE tenant_id = $1 AND entity_type = $2
                """
                result = await self._execute_query(query, tenant_id, entity_type, fetch_one=True)
            else:
                query = """
                    SELECT COUNT(*) FROM embeddings
                    WHERE tenant_id = $1
                """
                result = await self._execute_query(query, tenant_id, fetch_one=True)
            
            count = result['count'] if result else 0
            return count
            
        except Exception as e:
            logger.error("Failed to get embedding count", error=str(e))
            raise VectorStoreQueryError(f"Failed to get count: {e}")
    
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        try:
            query = "SELECT 1"
            await self._execute_query(query, fetch_one=True)
            return True
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Closed PgVector connection pool")
    
    def _ensure_vector_dimension(self, vector: Iterable[float]) -> np.ndarray:
        """Ensure a vector matches the expected dimensionality."""
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("Vector must be one-dimensional")
        
        if self.vector_dimension is not None and array.shape[0] != self.vector_dimension:
            raise ValueError(
                f"Expected vector dimension {self.vector_dimension}, "
                f"got {array.shape[0]}"
            )
        return array


def create_pgvector_store(dsn: str, **kwargs: Any) -> PgVectorStore:
    """Create a PgVector store instance."""
    return PgVectorStore(dsn, **kwargs)
