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
from typing import Any, Dict, List, Optional, Tuple
from contextlib import suppress
import asyncpg
import httpx
import numpy as np
import structlog
from asyncpg import exceptions as pg_exceptions
from service_model_serving.app.adapters.circuit_breaker import (
    get_circuit_breaker,
    CircuitBreakerError,
)

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

        # Resilience configuration for embedding API calls
        self.embedding_circuit_breaker = get_circuit_breaker(
            name="ml-embedding-api",
            failure_threshold=5,
            recovery_timeout=30.0,
            expected_exception=Exception,
        )
        self._embedding_retry_attempts = 3
        self._embedding_retry_base_delay = 1.0
        self._embedding_retry_max_delay = 8.0
        
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

    async def _call_embedding_service(self, payload: Dict[str, Any]) -> httpx.Response:
        """Invoke the embedding service with retry and circuit breaker protection."""
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < self._embedding_retry_attempts:
            attempt += 1
            try:
                async def _request() -> httpx.Response:
                    if not self.http_client:
                        raise ValueError("HTTP client not initialized")
                    return await self.http_client.post(
                        f"{self.embedding_service_url}/api/v1/embed",
                        json=payload,
                    )

                response = await self.embedding_circuit_breaker.call(_request)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status >= 500 or status == 429:
                        raise
                    raise ValueError(f"Embedding service returned status {status}") from exc
                return response

            except ValueError as non_retryable:
                logger.error(
                    "Embedding service returned non-retryable error",
                    attempt=attempt,
                    error=str(non_retryable),
                )
                raise
            except CircuitBreakerError as cb_error:
                logger.error(
                    "Embedding service circuit breaker open",
                    attempt=attempt,
                    error=str(cb_error),
                )
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self._embedding_retry_attempts:
                    logger.error(
                        "Embedding service call failed after retries",
                        attempts=attempt,
                        error=str(exc),
                    )
                    break

                delay = min(
                    self._embedding_retry_base_delay * (2 ** (attempt - 1)),
                    self._embedding_retry_max_delay,
                )
                logger.warning(
                    "Embedding service call failed, retrying",
                    attempt=attempt,
                    max_attempts=self._embedding_retry_attempts,
                    delay_seconds=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)

        if last_error:
            raise last_error
        raise RuntimeError("Embedding service call failed without raising an exception")
    
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

    async def _count_curve_metadata(self, tenant_id: str) -> Tuple[int, int]:
        """Count available curve metadata rows from aggregated sources.

        Returns a tuple ``(count, query_index)`` where ``query_index`` indicates
        which source query produced the count so subsequent pagination can reuse
        the same source when possible.
        """
        if not self.db_pool:
            raise ValueError("Database pool not initialized")
        
        queries = [
            "SELECT COUNT(*) FROM v_curve_metadata WHERE tenant_id = $1",
            "SELECT COUNT(*) FROM curve_metadata WHERE tenant_id = $1 AND active = TRUE",
        ]
        
        last_count: Optional[int] = None
        last_index: Optional[int] = None
        last_error: Optional[Exception] = None
        
        async with self.db_pool.acquire() as conn:
            for idx, query in enumerate(queries):
                try:
                    count = await conn.fetchval(query, tenant_id)
                    count_int = int(count or 0)
                    if count_int > 0:
                        return count_int, idx
                    if last_count is None:
                        last_count = count_int
                        last_index = idx
                except (pg_exceptions.UndefinedTableError, pg_exceptions.UndefinedColumnError) as exc:
                    last_error = exc
                    continue
        
        if last_count is not None and last_index is not None:
            return last_count, last_index
        
        raise ValueError("Curve metadata source not available") from last_error

    async def _fetch_curve_metadata_batch(
        self,
        tenant_id: str,
        limit: int,
        offset: int,
        source_index: int
    ) -> List[asyncpg.Record]:
        """Fetch a batch of curve metadata records."""
        if not self.db_pool:
            raise ValueError("Database pool not initialized")
        
        queries = [
            """
            SELECT
                curve_id,
                curve_name,
                commodity,
                region,
                contract_months,
                interpolation_method,
                metadata,
                commodity_name,
                region_name,
                country,
                timezone
            FROM v_curve_metadata
            WHERE tenant_id = $1
            ORDER BY curve_id
            LIMIT $2 OFFSET $3
            """,
            """
            SELECT
                curve_id,
                curve_name,
                commodity,
                region,
                contract_months,
                interpolation_method,
                metadata,
                NULL::text AS commodity_name,
                NULL::text AS region_name,
                NULL::text AS country,
                NULL::text AS timezone
            FROM curve_metadata
            WHERE tenant_id = $1
              AND active = TRUE
            ORDER BY curve_id
            LIMIT $2 OFFSET $3
            """,
        ]
        
        async with self.db_pool.acquire() as conn:
            query_order = [source_index] + [idx for idx in range(len(queries)) if idx != source_index]
            for idx in query_order:
                query = queries[idx]
                try:
                    rows = await conn.fetch(query, tenant_id, limit, offset)
                    if rows or idx == query_order[-1]:
                        return rows
                except (pg_exceptions.UndefinedTableError, pg_exceptions.UndefinedColumnError):
                    continue
        
        return []

    @staticmethod
    def _prepare_curve_metadata_payload(record: asyncpg.Record) -> Dict[str, Any]:
        """Normalize curve metadata record into text, metadata, and tags."""
        metadata_raw = record.get("metadata")
        if isinstance(metadata_raw, dict):
            metadata = dict(metadata_raw)
        elif metadata_raw is None:
            metadata = {}
        else:
            try:
                metadata = dict(metadata_raw)  # type: ignore[arg-type]
            except Exception:
                metadata = {}
        
        contract_months_raw = record.get("contract_months")
        if isinstance(contract_months_raw, str):
            contract_months = [part.strip() for part in contract_months_raw.split(",") if part.strip()]
        elif isinstance(contract_months_raw, list):
            contract_months = contract_months_raw
        else:
            contract_months = []
        
        description = metadata.get("description") or record.get("curve_name") or record.get("curve_id")
        if description and "description" not in metadata:
            metadata["description"] = description
        if record.get("curve_id") and "curve_id" not in metadata:
            metadata["curve_id"] = record.get("curve_id")
        if record.get("curve_name") and "curve_name" not in metadata:
            metadata["curve_name"] = record.get("curve_name")
        commodity = record.get("commodity")
        commodity_label = record.get("commodity_name") or commodity
        region = record.get("region")
        region_label = record.get("region_name") or region
        interpolation_method = record.get("interpolation_method")
        
        meta_payload: Dict[str, Any] = {
            "curve_id": record.get("curve_id"),
            "curve_name": record.get("curve_name"),
            "description": description,
            "commodity": commodity,
            "commodity_name": commodity_label,
            "region": region,
            "region_name": region_label,
            "country": record.get("country"),
            "timezone": record.get("timezone"),
            "contract_months": contract_months,
            "interpolation_method": interpolation_method,
            "metadata": metadata,
            "source": "aggregated_curve_metadata",
        }
        
        # Build descriptive text for embeddings/search
        text_parts = []
        if meta_payload["curve_name"]:
            text_parts.append(str(meta_payload["curve_name"]))
        if meta_payload["curve_id"]:
            text_parts.append(f"Curve ID {meta_payload['curve_id']}")
        if description and description != meta_payload.get("curve_name"):
            text_parts.append(str(description))
        
        context_parts = []
        if commodity_label:
            context_parts.append(f"Commodity {commodity_label}")
        if region_label:
            context_parts.append(f"Region {region_label}")
        if contract_months:
            context_parts.append(f"Contracts {', '.join(contract_months)}")
        if interpolation_method:
            context_parts.append(f"Interpolation {interpolation_method}")
        if context_parts:
            text_parts.append("; ".join(context_parts))
        
        text = ". ".join(part for part in text_parts if part)
        
        tags: List[str] = ["curve"]
        if commodity:
            tags.append(str(commodity).lower())
        if region:
            tags.append(str(region).lower())
        tags = list(dict.fromkeys(tag for tag in tags if tag))
        
        meta_payload["tags"] = tags
        
        return {
            "text": text,
            "metadata": meta_payload,
            "tags": tags,
        }

    async def _reindex_curve_metadata(
        self,
        batch_size: int,
        model_version: str,
        tenant_id: str = "default"
    ) -> Dict[str, Any]:
        """Reindex aggregated curve metadata into search index and vector store."""
        start_time = time.time()
        total_processed = 0
        total_errors = 0
        
        total_count, source_index = await self._count_curve_metadata(tenant_id)
        source_names = ["v_curve_metadata", "curve_metadata"]
        source_name = source_names[source_index] if source_index < len(source_names) else "unknown"
        logger.info(
            "Starting curve metadata reindex",
            total_count=total_count,
            batch_size=batch_size,
            model_version=model_version,
            source=source_name
        )
        
        offset = 0
        while offset < total_count:
            records = await self._fetch_curve_metadata_batch(tenant_id, batch_size, offset, source_index)
            if not records:
                break
            
            items: List[Dict[str, Any]] = []
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    for record in records:
                        try:
                            prepared = self._prepare_curve_metadata_payload(record)
                            meta_payload = prepared["metadata"]
                            await conn.execute(
                                """
                                INSERT INTO search_items (entity_type, entity_id, text, meta, tags, tenant_id)
                                VALUES ($1, $2, $3, $4, $5, $6)
                                ON CONFLICT (entity_type, entity_id, tenant_id)
                                DO UPDATE SET
                                    text = EXCLUDED.text,
                                    meta = EXCLUDED.meta,
                                    tags = EXCLUDED.tags,
                                    updated_at = CURRENT_TIMESTAMP
                                """,
                                "curve",
                                meta_payload["curve_id"],
                                prepared["text"],
                                meta_payload,
                                prepared["tags"],
                                tenant_id
                            )
                            items.append({
                                "type": "curve",
                                "id": meta_payload["curve_id"],
                                "text": prepared["text"],
                                "metadata": meta_payload,
                                "tags": prepared["tags"],
                                "tenant_id": tenant_id
                            })
                        except Exception as exc:
                            total_errors += 1
                            logger.warning(
                                "Failed to prepare curve metadata",
                                curve_id=record.get("curve_id"),
                                error=str(exc)
                            )
            
            if not items:
                offset += batch_size
                continue
            
            try:
                embeddings = await self._generate_embeddings(items, model_version)
                for idx, embedding in enumerate(embeddings):
                    embedding.setdefault("tenant_id", items[idx].get("tenant_id", "default"))
                await self._store_embeddings(embeddings, "curve", model_version)
                total_processed += len(items)
            except Exception as exc:
                total_errors += len(items)
                logger.error(
                    "Curve metadata embedding generation failed",
                    offset=offset,
                    error=str(exc)
                )
            
            offset += batch_size
            progress = (min(offset, total_count) / max(total_count, 1)) * 100
            logger.info(
                "Curve metadata reindex progress",
                processed=total_processed,
                errors=total_errors,
                progress=f"{progress:.1f}%"
            )
            
            self.metrics_collector.record_vector_store_operation(
                operation="curve_metadata_reindex",
                entity_type="curve"
            )
        
        duration = time.time() - start_time
        result = {
            "entity_type": "curve",
            "total_count": total_count,
            "processed": total_processed,
            "errors": total_errors,
            "duration_seconds": duration,
            "model_version": model_version,
            "source": source_name
        }
        
        logger.info("Curve metadata reindex completed", **result)
        return result
    
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
            
            if entity_type in ("curve", "curve_metadata"):
                return await self._reindex_curve_metadata(batch_size, model_version)
            
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

            payload = {"items": items, "model": "default"}
            response = await self._call_embedding_service(payload)

            data = response.json()
            vectors = data.get("vectors", [])
            model_version_used = data.get("model_version", model_version)
            
            # Prepare embeddings for storage
            embeddings = []
            for i, vector in enumerate(vectors):
                item_metadata = items[i].get("metadata", {})
                if not isinstance(item_metadata, dict):
                    item_metadata = {}
                metadata = dict(item_metadata)
                metadata.setdefault("text", items[i]["text"])
                if "tags" in items[i] and "tags" not in metadata:
                    metadata["tags"] = items[i]["tags"]
                
                embedding = {
                    "entity_type": items[i]["type"],
                    "entity_id": items[i]["id"],
                    "vector": vector,
                    "metadata": metadata,
                    "model_version": model_version_used
                }
                
                tenant_id = items[i].get("tenant_id")
                if tenant_id:
                    embedding["tenant_id"] = tenant_id
                
                embeddings.append(embedding)
            
            return embeddings
            
        except CircuitBreakerError as cb_error:
            logger.error("Embedding circuit breaker open", error=str(cb_error))
            raise
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
            tenant_id = "default"
            if embeddings:
                tenant_id = embeddings[0].get("tenant_id", "default")
            await self.vector_store.batch_store_embeddings(embeddings, tenant_id=tenant_id)
            
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
