"""Indexer worker main application."""

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import redis
from redis.lock import Lock
import structlog
from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from .workers.indexer_worker import IndexerWorker
from .runtime.metrics import get_metrics_collector
from libs.common.config import IndexerConfig
from libs.common.logging import configure_logging
from libs.common.tracing import configure_tracing
from libs.common.metrics import MetricsCollector

logger = structlog.get_logger("indexer_worker")

# Initialize Celery app
celery_app = Celery(
    "indexer_worker",
    broker="redis://localhost:6379",
    backend="redis://localhost:6379"
)

# Reindex idempotency configuration
REINDEX_LOCK_TIMEOUT_SECONDS = 60 * 30  # 30 minutes

# Global runtime state
indexer_config: Optional[IndexerConfig] = None
indexer_tracer: Optional[Any] = None
redis_client: Optional[redis.Redis] = None
metrics_collector: Optional[MetricsCollector] = None


def _ensure_config() -> IndexerConfig:
    """Return the initialized IndexerConfig or raise if missing."""
    if indexer_config is None:
        raise ValueError("Indexer configuration not initialized")
    return indexer_config


def _get_metrics() -> MetricsCollector:
    """Return process-wide metrics collector."""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = get_metrics_collector("indexer-worker")
    return metrics_collector


def _run_with_worker(
    coro_factory: Callable[[IndexerWorker], Awaitable[Dict[str, Any]]],
    *,
    enable_events: bool = False
) -> Dict[str, Any]:
    """Run an async worker operation inside a managed lifecycle."""
    config = _ensure_config()

    async def _runner() -> Dict[str, Any]:
        worker = IndexerWorker(config, enable_events=enable_events)
        if indexer_tracer:
            # Propagate tracer for consistency with legacy behaviour.
            worker.tracer = indexer_tracer
        try:
            await worker.initialize()
            return await coro_factory(worker)
        finally:
            await worker.cleanup()

    return asyncio.run(_runner())


def _acquire_reindex_lock(entity_type: str, model_version: str) -> Tuple[Optional[Lock], str]:
    """Attempt to acquire a distributed lock for reindexing.

    Returns a tuple of (lock, status) where status is one of:
    - "acquired": lock acquired successfully
    - "blocked": lock already held elsewhere
    - "unavailable": Redis client missing
    - "error": Redis error occurred
    """
    if redis_client is None:
        logger.warning(
            "Redis client not available; continuing without reindex idempotency",
            entity_type=entity_type,
            model_version=model_version
        )
        return None, "unavailable"
    
    try:
        lock_name = f"indexer:reindex:{entity_type}:{model_version}"
        lock = redis_client.lock(
            lock_name,
            timeout=REINDEX_LOCK_TIMEOUT_SECONDS,
            blocking=False,
            thread_local=False
        )
        acquired = lock.acquire(blocking=False)
        if not acquired:
            logger.info(
                "Reindex lock already held; task will be skipped",
                entity_type=entity_type,
                model_version=model_version,
                lock_name=lock_name
            )
            return None, "blocked"
        logger.info(
            "Acquired reindex lock",
            entity_type=entity_type,
            model_version=model_version,
            lock_name=lock_name
        )
        return lock, "acquired"
    except Exception as exc:
        logger.warning(
            "Failed to acquire reindex lock; continuing without idempotency guard",
            entity_type=entity_type,
            model_version=model_version,
            error=str(exc)
        )
        return None, "error"


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal."""
    global indexer_config, indexer_tracer, redis_client, metrics_collector
    
    try:
        # Initialize configuration
        indexer_config = IndexerConfig()
        configure_logging("indexer-worker", indexer_config.ml_log_level, indexer_config.ml_log_format)
        
        # Initialize tracing
        if indexer_config.ml_tracing_enabled:
            indexer_tracer = configure_tracing("indexer-worker", indexer_config.ml_otel_exporter)
            if indexer_tracer:
                logger.info("OpenTelemetry tracing enabled", exporter=indexer_config.ml_otel_exporter)
            else:
                logger.warning("Tracing initialization failed")
        else:
            indexer_tracer = None
            logger.info("OpenTelemetry tracing disabled via configuration")
        
        logger.info("Starting indexer worker")
        
        # Initialize metrics collector
        metrics_collector = get_metrics_collector("indexer-worker")
        
        # Initialize Redis client for locks
        try:
            redis_client = redis.from_url(indexer_config.ml_redis_url)
            # Perform a lightweight ping to surface connectivity issues early
            redis_client.ping()
            logger.info("Redis client initialized for reindex locks")
        except Exception as exc:
            redis_client = None
            logger.warning(
                "Failed to initialize Redis client; idempotency locks disabled",
                error=str(exc)
            )
        
        logger.info("Indexer worker started successfully")
        
    except Exception as e:
        logger.error("Failed to start indexer worker", error=str(e))
        raise


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown signal."""
    global redis_client
    
    try:
        if redis_client:
            try:
                redis_client.close()
            except Exception as exc:
                logger.warning("Failed to close Redis client", error=str(exc))
            finally:
                redis_client = None
        
        logger.info("Indexer worker shutdown complete")
        
    except Exception as e:
        logger.error("Indexer worker shutdown failed", error=str(e))


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def reindex_entities(self, entity_type: str, batch_size: int = 100, model_version: str = "default"):
    """Celery task for reindexing entities."""
    task_id = self.request.id
    start_time = time.perf_counter()
    task_status = "success"
    lock: Optional[Lock] = None
    lock_status = "unavailable"
    
    try:
        logger.info(
            "Starting reindexing task",
            task_id=task_id,
            entity_type=entity_type,
            batch_size=batch_size,
            model_version=model_version
        )
        
        lock, lock_status = _acquire_reindex_lock(entity_type, model_version)
        if lock_status == "blocked":
            task_status = "skipped"
            logger.info(
                "Reindexing task skipped due to in-flight job",
                task_id=task_id,
                entity_type=entity_type,
                model_version=model_version
            )
            return {
                "status": "skipped",
                "reason": "already_in_progress",
                "entity_type": entity_type,
                "model_version": model_version
            }
        
        def _reindex(worker: IndexerWorker) -> Awaitable[Dict[str, Any]]:
            return worker.reindex_entities(
                entity_type=entity_type,
                batch_size=batch_size,
                model_version=model_version
            )
        
        result = _run_with_worker(_reindex, enable_events=False)
        
        logger.info(
            "Reindexing task completed",
            task_id=task_id,
            entity_type=entity_type,
            batch_size=batch_size,
            result=result
        )
        
        return result
        
    except Exception as e:
        task_status = "failed"
        logger.error(
            "Reindexing task failed",
            task_id=task_id,
            entity_type=entity_type,
            batch_size=batch_size,
            error=str(e),
            retry_count=self.request.retries
        )
        raise
    finally:
        duration = time.perf_counter() - start_time
        try:
            metrics = _get_metrics()
            metrics.record_celery_task(
                task_name="reindex_entities",
                status=task_status,
                duration=duration,
                labels={"lock_status": lock_status}
            )
        except Exception as exc:
            logger.warning("Failed to record metrics for reindex task", error=str(exc))
        
        if lock:
            try:
                lock.release()
            except Exception as exc:
                logger.warning(
                    "Failed to release reindex lock",
                    entity_type=entity_type,
                    model_version=model_version,
                    error=str(exc)
                )


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 30})
def batch_embed_entities(self, job_id: str, entities: List[Dict[str, Any]], model_name: str = "default"):
    """Celery task for batch embedding generation."""
    task_id = self.request.id
    start_time = time.perf_counter()
    task_status = "success"
    
    try:
        logger.info(
            "Starting batch embedding task",
            task_id=task_id,
            job_id=job_id,
            entity_count=len(entities),
            model_name=model_name
        )
        
        def _batch_embed(worker: IndexerWorker) -> Awaitable[Dict[str, Any]]:
            return worker.batch_embed_entities(
                job_id=job_id,
                entities=entities,
                model_name=model_name
            )
        
        result = _run_with_worker(_batch_embed, enable_events=False)
        
        logger.info(
            "Batch embedding task completed",
            task_id=task_id,
            job_id=job_id,
            entity_count=len(entities),
            result=result
        )
        
        return result
        
    except Exception as e:
        task_status = "failed"
        logger.error(
            "Batch embedding task failed",
            task_id=task_id,
            job_id=job_id,
            entity_count=len(entities),
            error=str(e),
            retry_count=self.request.retries
        )
        raise
    finally:
        duration = time.perf_counter() - start_time
        try:
            metrics = _get_metrics()
            metrics.record_celery_task(
                task_name="batch_embed_entities",
                status=task_status,
                duration=duration
            )
        except Exception as exc:
            logger.warning("Failed to record metrics for batch embed task", error=str(exc))


@celery_app.task(bind=True)
def cleanup_old_embeddings(self, entity_type: str, model_version: str, days_old: int = 30):
    """Celery task for cleaning up old embeddings."""
    start_time = time.perf_counter()
    task_status = "success"
    
    try:
        def _cleanup(worker: IndexerWorker) -> Awaitable[Dict[str, Any]]:
            return worker.cleanup_old_embeddings(
                entity_type=entity_type,
                model_version=model_version,
                days_old=days_old
            )
        
        result = _run_with_worker(_cleanup, enable_events=False)
        
        logger.info(
            "Cleanup task completed",
            entity_type=entity_type,
            model_version=model_version,
            days_old=days_old,
            result=result
        )
        
        return result
        
    except Exception as e:
        task_status = "failed"
        logger.error(
            "Cleanup task failed",
            entity_type=entity_type,
            model_version=model_version,
            days_old=days_old,
            error=str(e)
        )
        raise
    finally:
        duration = time.perf_counter() - start_time
        try:
            metrics = _get_metrics()
            metrics.record_celery_task(
                task_name="cleanup_old_embeddings",
                status=task_status,
                duration=duration
            )
        except Exception as exc:
            logger.warning("Failed to record metrics for cleanup task", error=str(exc))


if __name__ == "__main__":
    # Start Celery worker
    celery_app.worker_main([
        "worker",
        "--loglevel=info",
        "--concurrency=4",
        "--queues=embeddings_rebuild,embeddings_batch,embeddings_cleanup"
    ])
