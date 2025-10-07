"""Search manager for hybrid semantic and lexical search.

Combines vector similarity (semantic) with full‑text ranking (lexical) and
merges results using Reciprocal Rank Fusion (RRF). Designed for practical,
incremental search quality improvements without heavy infrastructure.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable
from contextlib import suppress
import asyncpg
import numpy as np
import httpx
import structlog

from libs.common.config import SearchConfig
from libs.common.events import EventSubscriber, EventType
from libs.vector_store.factory import create_vector_store_from_env
from ..ranking.fusion import create_fusion_algorithm, create_query_preprocessor, create_result_ranker
from ..retrievers.cache_manager import create_search_cache_manager

# Import circuit breaker from model serving (shared component)
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from service_model_serving.app.adapters.circuit_breaker import (
    get_circuit_breaker,
    CircuitBreakerError,
)

logger = structlog.get_logger("search_service.search_manager")


class SearchManager:
    """Manages hybrid search operations.

    Responsibilities
    - Keep a DB pool and HTTP client for dependencies
    - Generate query embeddings via the embedding service
    - Query lexical results using PostgreSQL full‑text search
    - Merge, filter, and return ranked results
    """
    
    def __init__(self, config: SearchConfig):
        """Construct a search manager.

        Parameters
        - config: ``SearchConfig`` providing DSN, Redis URL, and service URLs
        """
        self.config = config
        self.db_pool: Optional[asyncpg.Pool] = None
        self.embedding_service_url = config.ml_embedding_service_url
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Initialize vector store
        env_config = {
            "ML_VECTOR_DB_DSN": config.ml_vector_db_dsn,
            "ML_VECTOR_POOL_SIZE": "10",
            "ML_VECTOR_MAX_QUERIES": "50000",
            "ML_VECTOR_COMMAND_TIMEOUT": "60",
            "ML_VECTOR_DIMENSION": str(config.ml_vector_dimension)
        }
        self.vector_store = create_vector_store_from_env(env_config)
        
        # Initialize fusion algorithm, preprocessor, and ranker
        fusion_algorithm = os.getenv("ML_SEARCH_FUSION_ALGORITHM", "rrf")
        self.fusion_algorithm = create_fusion_algorithm(fusion_algorithm)
        self.query_preprocessor = create_query_preprocessor()
        self.result_ranker = create_result_ranker()
        
        # Initialize cache manager
        self.cache_manager = create_search_cache_manager(
            redis_url=config.ml_redis_url,
            query_cache_ttl=int(os.getenv("ML_SEARCH_QUERY_CACHE_TTL", "300")),
            embedding_cache_ttl=int(os.getenv("ML_SEARCH_EMBEDDING_CACHE_TTL", "3600")),
            result_cache_ttl=int(os.getenv("ML_SEARCH_RESULT_CACHE_TTL", "600"))
        )
        
        # Initialize circuit breaker for embedding service calls
        self.embedding_circuit_breaker = get_circuit_breaker(
            name="embedding_service",
            failure_threshold=5,
            recovery_timeout=30.0,
            expected_exception=Exception
        )
        self.embedding_retry_attempts = int(os.getenv("ML_SEARCH_EMBEDDING_RETRY_ATTEMPTS", "3"))
        self.embedding_retry_base_delay = float(os.getenv("ML_SEARCH_EMBEDDING_RETRY_BASE_DELAY", "1.0"))
        self.embedding_retry_max_delay = float(os.getenv("ML_SEARCH_EMBEDDING_RETRY_MAX_DELAY", "8.0"))
        
        # Initialize event subscriber for index updates
        self.event_subscriber = EventSubscriber(config.ml_redis_url)
        self._setup_event_handlers()
        self._event_listener_task: Optional[asyncio.Task] = None
    
    def _setup_event_handlers(self):
        """Set up event handlers for index updates."""
        def handle_search_index_updated(event_data: Dict[str, Any]):
            """React to SEARCH_INDEX_UPDATED events (logging/refresh hooks)."""
            asyncio.create_task(self._handle_search_index_updated(event_data))
        
        self.event_subscriber.subscribe(EventType.SEARCH_INDEX_UPDATED, handle_search_index_updated)
    
    async def initialize(self):
        """Initialize the search manager.

        Acquires DB pool, HTTP client, and starts the background event loop.
        """
        try:
            # Initialize database connection pool
            self.db_pool = await asyncpg.create_pool(
                self.config.ml_vector_db_dsn,
                min_size=1,
                max_size=10
            )
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Start event listener in background
            asyncio.create_task(self._start_event_listener())
            
            logger.info("Search manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize search manager", error=str(e))
            raise
    
    async def _start_event_listener(self):
        """Start the event listener in background."""
        try:
            if self._event_listener_task and not self._event_listener_task.done():
                logger.info("Event listener already running")
                return
            
            # Start event listener in background task with retry logic
            self._event_listener_task = asyncio.create_task(
                self._run_event_listener_with_retry()
            )
            logger.info("Event listener started")
        except Exception as e:
            logger.error("Failed to start event listener", error=str(e))
    
    async def _run_event_listener_with_retry(self):
        """Run event listener with retry logic."""
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                await self.event_subscriber.start_listening()
                break  # Success, exit retry loop
            except asyncio.CancelledError:
                logger.info("Event listener cancelled")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error("Event listener failed after all retries", error=str(e))
                    break
                
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    "Event listener failed, retrying",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e)
                )
                await asyncio.sleep(delay)

    async def _call_with_retry(
        self,
        func: Callable[[], Awaitable[Any]],
        operation_name: str,
        max_attempts: int,
        base_delay: float,
        max_delay: float
    ) -> Any:
        """Execute a coroutine-returning callable with retry and backoff."""
        last_exception: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                return await func()
            except CircuitBreakerError as cb_error:
                logger.error(
                    "Circuit breaker open, aborting retries",
                    operation=operation_name,
                    error=str(cb_error)
                )
                raise
            except Exception as exc:
                last_exception = exc

                if attempt == max_attempts:
                    logger.error(
                        "Operation failed after retries",
                        operation=operation_name,
                        attempts=attempt,
                        error=str(exc)
                    )
                    raise

                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    "Operation failed, retrying",
                    operation=operation_name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_seconds=delay,
                    error=str(exc)
                )
                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Retry logic failed for {operation_name}")
    
    async def _handle_search_index_updated(self, event_data: Dict[str, Any]):
        """Handle search index updated event."""
        try:
            entity_type = event_data.get("entity_type")
            count = event_data.get("count")
            
            logger.info(
                "Handled search index updated event",
                entity_type=entity_type,
                count=count
            )
            
        except Exception as e:
            logger.error("Failed to handle search index updated event", error=str(e))
    
    async def search(
        self,
        query: str,
        semantic: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0,
        tenant_id: Optional[str] = None
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Perform hybrid search.

        Returns a list of tuples ``(type, id, score, metadata)`` sorted by
        final fused score. The score is a relative ranking signal, not an
        absolute probability or similarity.
        """
        try:
            # Check cache first
            cached_results = await self.cache_manager.get_cached_search_results(
                query=query,
                filters=filters,
                semantic=semantic,
                limit=limit
            )
            
            if cached_results is not None:
                logger.info("Search cache hit", query=query[:50])
                return cached_results
            
            # Preprocess query
            query_info = self.query_preprocessor.preprocess_query(query)
            processed_query = query_info["expanded_query"]
            
            semantic_results = []
            if semantic:
                # Perform semantic search
                semantic_results = await self._semantic_search(
                    query=processed_query,
                    filters=filters,
                    limit=limit * 2,  # Get more for fusion
                    similarity_threshold=similarity_threshold,
                    tenant_id=tenant_id
                )
            
            # Perform lexical search
            lexical_results = await self._lexical_search(
                query=processed_query,
                filters=filters,
                limit=limit * 2,  # Get more for fusion
                tenant_id=tenant_id
            )
            
            # Fuse results using the fusion library
            fused_results = self.fusion_algorithm.fuse_results(
                semantic_results=semantic_results,
                lexical_results=lexical_results,
                query=query
            )
            
            # Apply additional ranking
            ranked_results = self.result_ranker.rerank_results(
                results=fused_results,
                query_info=query_info
            )
            
            # Apply filters
            if filters:
                ranked_results = await self._apply_filters(ranked_results, filters)
            
            # Limit results
            final_results = ranked_results[:limit]
            
            # Cache results
            await self.cache_manager.cache_search_results(
                query=query,
                results=final_results,
                filters=filters,
                semantic=semantic,
                limit=limit
            )
            
            logger.info(
                "Search completed",
                query=query,
                semantic=semantic,
                results_count=len(final_results),
                cache_miss=True
            )
            
            return final_results
            
        except Exception as e:
            logger.error("Search failed", query=query, error=str(e))
            raise
    
    async def _semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0,
        tenant_id: Optional[str] = None
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Perform semantic search using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)
            
            if query_embedding is None:
                return []
            
            # Search for similar vectors
            similar_vectors = await self.vector_store.search_similar(
                query_vector=query_embedding,
                entity_type=filters.get("type") if filters else None,
                tenant_id=tenant_id or filters.get("tenant_id", "default") if filters else "default",
                limit=limit * 2,  # Get more results for merging
                similarity_threshold=similarity_threshold
            )
            
            # Convert to search results format
            results = []
            for entity_type, entity_id, similarity, metadata in similar_vectors:
                results.append((entity_type, entity_id, similarity, metadata))
            
            logger.info("Semantic search completed", results_count=len(results))
            return results
            
        except Exception as e:
            logger.error("Semantic search failed", error=str(e))
            return []
    
    async def _lexical_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        tenant_id: Optional[str] = None
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Perform lexical search using PostgreSQL full-text search."""
        try:
            if not self.db_pool:
                return []
            
            # Build SQL query
            sql = """
                SELECT entity_type, entity_id, 
                       ts_rank(to_tsvector('english', text), plainto_tsquery('english', $1)) as rank,
                       COALESCE(meta, '{}'::jsonb) as meta
                FROM search_items
                WHERE to_tsvector('english', text) @@ plainto_tsquery('english', $1)
            """
            
            params = [query]
            param_count = 1
            
            # Add filters
            if filters:
                if "type" in filters:
                    param_count += 1
                    sql += f" AND entity_type = ${param_count}"
                    params.append(filters["type"])
                
                if "tenant_id" in filters:
                    param_count += 1
                    sql += f" AND tenant_id = ${param_count}"
                    params.append(filters["tenant_id"])
                else:
                    param_count += 1
                    sql += f" AND tenant_id = ${param_count}"
                    params.append(tenant_id or "default")
            else:
                param_count += 1
                sql += f" AND tenant_id = ${param_count}"
                params.append(tenant_id or "default")
            
            sql += " ORDER BY rank DESC LIMIT $2"
            params.append(limit * 2)  # Get more results for merging
            
            # Execute query
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            
            # Convert to search results format
            results = []
            for row in rows:
                results.append((
                    row['entity_type'],
                    row['entity_id'],
                    float(row['rank']),
                    row['meta'] or {}
                ))
            
            logger.info("Lexical search completed", results_count=len(results))
            return results
            
        except Exception as e:
            logger.error("Lexical search failed", error=str(e))
            return []
    
    
    async def _apply_filters(
        self,
        results: List[Tuple[str, str, float, Dict[str, Any]]],
        filters: Dict[str, Any]
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Apply additional filters to search results."""
        try:
            filtered_results = []
            
            for entity_type, entity_id, score, metadata in results:
                # Apply type filter
                if "type" in filters:
                    if isinstance(filters["type"], list):
                        if entity_type not in filters["type"]:
                            continue
                    else:
                        if entity_type != filters["type"]:
                            continue
                
                # Apply tenant filter
                if "tenant_id" in filters:
                    if metadata.get("tenant_id") != filters["tenant_id"]:
                        continue
                
                # Apply tag filters
                if "tags" in filters:
                    entity_tags = metadata.get("tags", [])
                    if not any(tag in entity_tags for tag in filters["tags"]):
                        continue
                
                filtered_results.append((entity_type, entity_id, score, metadata))
            
            logger.info("Filters applied", original_count=len(results), filtered_count=len(filtered_results))
            return filtered_results
            
        except Exception as e:
            logger.error("Filter application failed", error=str(e))
            return results
    
    async def _generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for search query."""
        try:
            # Check cache first
            cached_embedding = await self.cache_manager.get_cached_query_embedding(query)
            if cached_embedding is not None:
                return np.array(cached_embedding)
            
            if not self.http_client:
                return None
            
            # Call embedding service with circuit breaker
            async def call_embedding_service():
                """POST to embedding service to obtain query vector."""
                response = await self.http_client.post(
                    f"{self.embedding_service_url}/api/v1/embed",
                    json={
                        "items": [{"text": query}],
                        "model": "default"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    vectors = data.get("vectors", [])
                    if vectors:
                        embedding = vectors[0]
                        # Cache the embedding
                        await self.cache_manager.cache_query_embedding(query, embedding)
                        return np.array(embedding)
                
                raise Exception(f"Embedding service returned status {response.status_code}")
            
            try:
                result = await self._call_with_retry(
                    lambda: self.embedding_circuit_breaker.call(call_embedding_service),
                    operation_name="embedding_service_request",
                    max_attempts=self.embedding_retry_attempts,
                    base_delay=self.embedding_retry_base_delay,
                    max_delay=self.embedding_retry_max_delay
                )
                return result
            except Exception as e:
                logger.error("Embedding service call failed", error=str(e))
                return None
            
        except Exception as e:
            logger.error("Query embedding generation failed", error=str(e))
            return None
    
    async def index_entity(
        self,
        entity_type: str,
        entity_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        tenant_id: Optional[str] = None
    ):
        """Index an entity for search."""
        try:
            if not self.db_pool:
                raise ValueError("Database pool not initialized")
            
            # Insert into search_items table
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO search_items (entity_type, entity_id, text, meta, tags, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (entity_type, entity_id, tenant_id)
                    DO UPDATE SET
                        text = EXCLUDED.text,
                        meta = EXCLUDED.meta,
                        tags = EXCLUDED.tags,
                        updated_at = CURRENT_TIMESTAMP
                """, entity_type, entity_id, text, metadata or {}, tags, tenant_id or "default")
            
            # Invalidate related cache entries
            await self.cache_manager.invalidate_entity_cache(entity_type, entity_id)
            
            logger.info("Entity indexed", entity_type=entity_type, entity_id=entity_id)
            
        except Exception as e:
            logger.error("Entity indexing failed", entity_type=entity_type, entity_id=entity_id, error=str(e))
            raise
    
    async def remove_entity(self, entity_type: str, entity_id: str):
        """Remove an entity from the search index."""
        try:
            if not self.db_pool:
                raise ValueError("Database pool not initialized")
            
            # Remove from search_items table
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM search_items
                    WHERE entity_type = $1 AND entity_id = $2 AND tenant_id = $3
                """, entity_type, entity_id, "default")
            
            # Invalidate related cache entries
            await self.cache_manager.invalidate_entity_cache(entity_type, entity_id)
            
            logger.info("Entity removed from index", entity_type=entity_type, entity_id=entity_id)
            
        except Exception as e:
            logger.error("Entity removal failed", entity_type=entity_type, entity_id=entity_id, error=str(e))
            raise
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get search index statistics."""
        try:
            if not self.db_pool:
                return {}
            
            async with self.db_pool.acquire() as conn:
                # Get total count
                total_count = await conn.fetchval("SELECT COUNT(*) FROM search_items")
                
                # Get count by entity type
                type_counts = await conn.fetch("""
                    SELECT entity_type, COUNT(*) as count
                    FROM search_items
                    GROUP BY entity_type
                    ORDER BY count DESC
                """)
                
                # Get count by tenant
                tenant_counts = await conn.fetch("""
                    SELECT tenant_id, COUNT(*) as count
                    FROM search_items
                    GROUP BY tenant_id
                    ORDER BY count DESC
                """)
            
            return {
                "total_entities": total_count,
                "by_entity_type": {row['entity_type']: row['count'] for row in type_counts},
                "by_tenant": {row['tenant_id']: row['count'] for row in tenant_counts}
            }
            
        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            return {}
    
    async def rebuild_index(self, entity_type: Optional[str] = None):
        """Rebuild the search index."""
        try:
            # This would typically:
            # 1. Clear existing index
            # 2. Re-index all entities from the source database
            # 3. Update vector embeddings
            
            logger.info("Index rebuild triggered", entity_type=entity_type)
            
            # For now, just log the request
            # In a real implementation, this would start a background job
            
        except Exception as e:
            logger.error("Index rebuild failed", entity_type=entity_type, error=str(e))
            raise
    
    async def health_check(self) -> bool:
        """Check if the search manager is healthy."""
        try:
            # Check database connection
            if not self.db_pool:
                return False
            
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            # Check vector store
            vector_store_healthy = await self.vector_store.health_check()
            
            return vector_store_healthy
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Stop event listener task if running
            if self._event_listener_task and not self._event_listener_task.done():
                self._event_listener_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._event_listener_task
            
            # Close event subscriber connection
            with suppress(Exception):
                await self.event_subscriber.close()
            
            # Close database pool
            if self.db_pool:
                await self.db_pool.close()
            
            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
            
            # Close vector store connection
            if hasattr(self.vector_store, 'close'):
                await self.vector_store.close()
            
            # Close cache manager
            if hasattr(self.cache_manager, 'close'):
                await self.cache_manager.close()
            
            logger.info("Search manager cleanup completed")
            
        except Exception as e:
            logger.error("Search manager cleanup failed", error=str(e))
