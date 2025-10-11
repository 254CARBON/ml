"""Indexer worker for large-scale embedding operations.

Coordinates batch embedding generation and storage into the vector store.
Consumes events (e.g., reindex requests), queries source data in batches,
invokes the embedding service, and upserts results.

Execution model
- Async worker with an HTTP client and DB pool
- Background event listener sets tasks in motion
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from contextlib import suppress
import asyncpg
import httpx
import numpy as np
import structlog
from asyncpg import exceptions as pg_exceptions

from libs.common.config import IndexerConfig
from libs.common.events import EventSubscriber, EventType, create_event_publisher
from libs.common.metrics import get_metrics_collector
from libs.vector_store.factory import create_vector_store_from_env

logger = structlog.get_logger("indexer_worker")


class IndexerWorker:
    """Worker for large-scale embedding and indexing operations.

    The worker is designed to be resilient and to surface progress via logs.
    All long‑running operations are chunked into batches to limit memory and
    service pressure.
    """
    
    def __init__(self, config: IndexerConfig, enable_events: bool = True):
        self.config = config
        self.db_pool: Optional[asyncpg.Pool] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.embedding_service_url = config.ml_embedding_service_url
        self.enable_events = enable_events
        
        # Initialize vector store
        env_config = {
            "ML_VECTOR_DB_DSN": config.ml_vector_db_dsn,
            "ML_VECTOR_POOL_SIZE": "20",
            "ML_VECTOR_MAX_QUERIES": "100000",
            "ML_VECTOR_COMMAND_TIMEOUT": "120",
            "ML_VECTOR_DIMENSION": str(config.ml_vector_dimension)
        }
        self.vector_store = create_vector_store_from_env(env_config)
        
        # Initialize event subscriber and publisher
        self.event_publisher = create_event_publisher(config.ml_redis_url)
        self.event_subscriber: Optional[EventSubscriber] = None
        self._event_listener_task: Optional[asyncio.Task] = None
        if self.enable_events:
            self.event_subscriber = EventSubscriber(config.ml_redis_url)
            self._setup_event_handlers()
        
        # Initialize metrics collector
        self.metrics_collector = get_metrics_collector("indexer-worker")
    
    def _setup_event_handlers(self):
        """Set up event handlers for reindex requests."""
        if not self.event_subscriber:
            return
        def handle_reindex_request(event_data: Dict[str, Any]):
            asyncio.create_task(self._handle_reindex_request(event_data))
        
        self.event_subscriber.subscribe(EventType.EMBEDDING_REINDEX_REQUEST, handle_reindex_request)
    
    async def initialize(self):
        """Initialize the indexer worker.

        Creates DB connection pool and a shared HTTP client, then starts the
        event listener in the background.
        """
        try:
            # Initialize database connection pool
            self.db_pool = await asyncpg.create_pool(
                self.config.ml_vector_db_dsn,
                min_size=2,
                max_size=20
            )
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=60.0)
            
            # Start event listener in background
            if self.enable_events and self.event_subscriber:
                asyncio.create_task(self._start_event_listener())
            
            logger.info("Indexer worker initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize indexer worker", error=str(e))
            raise
    
    async def _start_event_listener(self):
        """Start the event listener in background."""
        if not self.enable_events or not self.event_subscriber:
            logger.info("Event listener disabled for this worker instance")
            return
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
        if not self.enable_events or not self.event_subscriber:
            return
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
    
    async def _handle_reindex_request(self, event_data: Dict[str, Any]):
        """Handle embedding reindex request event.

        Kicks off a full reindex for ``entity_type`` with a configurable batch
        size. Errors are logged and do not crash the worker process.
        """
        try:
            entity_type = event_data.get("entity_type")
            batch_size = event_data.get("batch_size", 100)
            model_version = event_data.get("model_version", "default")
            
            # Trigger reindexing
            await self.reindex_entities(entity_type, batch_size, model_version)
            
            logger.info(
                "Handled reindex request event",
                entity_type=entity_type,
                batch_size=batch_size,
                model_version=model_version
            )
            
        except Exception as e:
            logger.error("Failed to handle reindex request event", error=str(e))
    
    async def reindex_entities(
        self,
        entity_type: str,
        batch_size: int = 100,
        model_version: str = "default"
    ) -> Dict[str, Any]:
        """Reindex entities of a specific type.

        Strategy
        - Page through source rows deterministically (``ORDER BY entity_id``)
        - Generate embeddings in batches to control memory/throughput
        - Upsert into the vector store; log progress and aggregate counters
        """
        try:
            if not self.db_pool:
                raise ValueError("Database pool not initialized")
            
            start_time = time.time()
            total_processed = 0
            total_errors = 0
            
            # Get total count for progress tracking
            async with self.db_pool.acquire() as conn:
                total_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM search_items 
                    WHERE entity_type = $1 AND tenant_id = $2
                """, entity_type, "default")
            
            logger.info(
                "Starting reindexing",
                entity_type=entity_type,
                total_count=total_count,
                batch_size=batch_size,
                model_version=model_version
            )
            
            # Process in batches
            offset = 0
            while offset < total_count:
                # Fetch batch of entities from the source index table
                async with self.db_pool.acquire() as conn:
                    entities = await conn.fetch("""
                        SELECT entity_id, text, meta, tags
                        FROM search_items
                        WHERE entity_type = $1 AND tenant_id = $2
                        ORDER BY entity_id
                        LIMIT $3 OFFSET $4
                    """, entity_type, "default", batch_size, offset)
                
                if not entities:
                    break
                
                # Prepare items for embedding generation – keep payload small
                items = []
                for entity in entities:
                    items.append({
                        "type": entity_type,
                        "id": entity["entity_id"],
                        "text": entity["text"]
                    })
                
                # Generate embeddings via embedding service then store results
                try:
                    embeddings = await self._generate_embeddings(items, model_version)
                    
                    # Store embeddings in vector store
                    await self._store_embeddings(embeddings, entity_type, model_version)
                    
                    total_processed += len(entities)
                    
                except Exception as e:
                    logger.error(
                        "Batch processing failed",
                        entity_type=entity_type,
                        offset=offset,
                        error=str(e)
                    )
                    total_errors += len(entities)
                
                offset += batch_size
                
                # Log progress for visibility and observability
                progress = (offset / total_count) * 100
                logger.info(
                    "Reindexing progress",
                    entity_type=entity_type,
                    progress=f"{progress:.1f}%",
                    processed=total_processed,
                    errors=total_errors
                )
                
                # Record metrics
                self.metrics_collector.record_vector_store_operation(
                    operation="batch_reindex",
                    entity_type=entity_type
                )
            
            duration = time.time() - start_time
            
            result = {
                "entity_type": entity_type,
                "total_count": total_count,
                "processed": total_processed,
                "errors": total_errors,
                "duration_seconds": duration,
                "model_version": model_version
            }
            
            logger.info(
                "Reindexing completed",
                **result
            )
            
            return result
            
        except Exception as e:
            logger.error("Reindexing failed", entity_type=entity_type, error=str(e))
            raise
    
    async def batch_embed_entities(
        self,
        job_id: str,
        entities: List[Dict[str, Any]],
        model_name: str = "default"
    ) -> Dict[str, Any]:
        """Generate embeddings for a batch of entities.

        Useful for ad‑hoc jobs initiated by operators or other services.
        """
        try:
            start_time = time.time()
            
            # Prepare items for embedding generation
            items = []
            for entity in entities:
                items.append({
                    "type": entity.get("type", "unknown"),
                    "id": entity.get("id", "unknown"),
                    "text": entity.get("text", "")
                })
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(items, model_name)
            
            # Store embeddings
            await self._store_embeddings(embeddings, "mixed", model_name)
            
            duration = time.time() - start_time
            
            result = {
                "job_id": job_id,
                "entity_count": len(entities),
                "embedding_count": len(embeddings),
                "duration_seconds": duration,
                "model_name": model_name
            }
            
            logger.info("Batch embedding completed", **result)
            return result
            
        except Exception as e:
            logger.error("Batch embedding failed", job_id=job_id, error=str(e))
            raise
    
    async def cleanup_old_embeddings(
        self,
        entity_type: str,
        model_version: str,
        days_old: int = 30
    ) -> Dict[str, Any]:
        """Clean up old embeddings.

        Removes records older than ``days_old`` for the given model version.
        """
        try:
            if not self.db_pool:
                raise ValueError("Database pool not initialized")
            
            start_time = time.time()
            
            # Delete old embeddings
            async with self.db_pool.acquire() as conn:
                deleted_count = await conn.fetchval("""
                    DELETE FROM embeddings
                    WHERE entity_type = $1 AND model_version = $2 AND tenant_id = $3
                    AND created_at < NOW() - INTERVAL '%s days'
                """, entity_type, model_version, "default", days_old)
            
            duration = time.time() - start_time
            
            result = {
                "entity_type": entity_type,
                "model_version": model_version,
                "days_old": days_old,
                "deleted_count": deleted_count,
                "duration_seconds": duration
            }
            
            logger.info("Cleanup completed", **result)
            return result
            
        except Exception as e:
            logger.error("Cleanup failed", entity_type=entity_type, error=str(e))
            raise
    
    async def _generate_embeddings(
        self,
        items: List[Dict[str, Any]],
        model_version: str
    ) -> List[Dict[str, Any]]:
        """Generate embeddings using the embedding service.

        Sends a compact payload to the embedding API and returns a normalized
        list of records ready for storage.
        """
        try:
            if not self.http_client:
                raise ValueError("HTTP client not initialized")
            
            # Call embedding service
            response = await self.http_client.post(
                f"{self.embedding_service_url}/api/v1/embed",
                json={
                    "items": items,
                    "model": "default"
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Embedding service returned status {response.status_code}")
            
            data = response.json()
            vectors = data.get("vectors", [])
            model_version_used = data.get("model_version", model_version)
            
            # Prepare embeddings for storage
            embeddings = []
            for i, vector in enumerate(vectors):
                embeddings.append({
                    "entity_type": items[i]["type"],
                    "entity_id": items[i]["id"],
                    "vector": vector,
                    "metadata": {"text": items[i]["text"]},
                    "model_version": model_version_used
                })
            
            return embeddings
            
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise
    
    async def _store_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        entity_type: str,
        model_version: str
    ):
        """Store embeddings in the vector store."""
        try:
            # Convert to numpy arrays for storage
            for embedding in embeddings:
                embedding["vector"] = np.array(embedding["vector"])
            # NOTE: This module references ``np``. Ensure ``numpy`` is imported
            # where this worker is executed.
            
            # Store in batch
            await self.vector_store.batch_store_embeddings(embeddings)
            
            logger.info(
                "Embeddings stored",
                count=len(embeddings),
                entity_type=entity_type,
                model_version=model_version
            )
            
        except Exception as e:
            logger.error("Embedding storage failed", error=str(e))
            raise
    
    async def health_check(self) -> bool:
        """Check if the indexer worker is healthy."""
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
        """Cleanup resources.

        Closes DB connections, HTTP client, and underlying vector store.
        Safe to call multiple times.
        """
        try:
            # Stop event listener task if running
            if self._event_listener_task and not self._event_listener_task.done():
                self._event_listener_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._event_listener_task
            
            # Close event subscriber
            if self.event_subscriber:
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
            
            logger.info("Indexer worker cleanup completed")
            
        except Exception as e:
            logger.error("Indexer worker cleanup failed", error=str(e))
