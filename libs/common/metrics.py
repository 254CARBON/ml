"""Metrics collection for ML platform services.

Provides a thin convenience wrapper around ``prometheus_client`` so services
can consistently record HTTP, model, embedding, search, and cache metrics.

Design notes
- Metrics and labels are predeclared to avoid cardinality explosions
- A single registry is kept per service (can be injected if needed)
- Decorators are provided for quick timing/call‑count instrumentation
"""

import time
from typing import Any, Dict, Optional, Callable
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import structlog

logger = structlog.get_logger("metrics")


class MetricsCollector:
    """Centralized metrics collection for ML services.

    Parameters
    - service_name: Logical name used for scoping/labels if desired
    - registry: Optional custom ``CollectorRegistry`` (e.g. for testing)

    Exposes typed helpers for common events to keep label sets consistent.
    """
    
    def __init__(self, service_name: str, registry: Optional[CollectorRegistry] = None):
        self.service_name = service_name
        self.registry = registry or CollectorRegistry()
        
        # Common metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # ML-specific metrics
        self.inference_requests = Counter(
            'ml_inference_requests_total',
            'Total ML inference requests',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.inference_duration = Histogram(
            'ml_inference_duration_seconds',
            'ML inference duration',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.embedding_requests = Counter(
            'ml_embedding_requests_total',
            'Total embedding generation requests',
            ['model_name', 'entity_type'],
            registry=self.registry
        )
        
        self.embedding_duration = Histogram(
            'ml_embedding_duration_seconds',
            'Embedding generation duration',
            ['model_name', 'entity_type'],
            registry=self.registry
        )
        
        self.search_requests = Counter(
            'ml_search_requests_total',
            'Total search requests',
            ['query_type'],
            registry=self.registry
        )
        
        self.search_duration = Histogram(
            'ml_search_duration_seconds',
            'Search duration',
            ['query_type'],
            registry=self.registry
        )
        
        self.vector_store_operations = Counter(
            'ml_vector_store_operations_total',
            'Total vector store operations',
            ['operation', 'entity_type'],
            registry=self.registry
        )
        
        self.model_versions_loaded = Gauge(
            'ml_model_versions_loaded',
            'Number of loaded model versions',
            ['model_name'],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'ml_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'ml_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Celery worker metrics
        self.celery_task_invocations = Counter(
            'celery_tasks_total',
            'Total Celery task executions partitioned by status.',
            ['task_name', 'status'],
            registry=self.registry
        )
        
        self.celery_task_duration = Histogram(
            'celery_task_duration_seconds',
            'Celery task execution duration seconds.',
            ['task_name'],
            registry=self.registry
        )
        
        self.celery_task_lock_status = Counter(
            'celery_task_lock_status_total',
            'Celery task lock acquisition outcomes.',
            ['task_name', 'lock_status'],
            registry=self.registry
        )
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ) -> None:
        """Record HTTP request metrics.

        duration is expected in seconds to match Prometheus histogram units.
        """
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_inference(
        self,
        model_name: str,
        model_version: str,
        duration: float
    ) -> None:
        """Record ML inference metrics."""
        self.inference_requests.labels(model_name=model_name, model_version=model_version).inc()
        self.inference_duration.labels(model_name=model_name, model_version=model_version).observe(duration)
    
    def record_embedding(
        self,
        model_name: str,
        entity_type: str,
        duration: float
    ) -> None:
        """Record embedding generation metrics."""
        self.embedding_requests.labels(model_name=model_name, entity_type=entity_type).inc()
        self.embedding_duration.labels(model_name=model_name, entity_type=entity_type).observe(duration)
    
    def record_search(
        self,
        query_type: str,
        duration: float
    ) -> None:
        """Record search metrics."""
        self.search_requests.labels(query_type=query_type).inc()
        self.search_duration.labels(query_type=query_type).observe(duration)
    
    def record_vector_store_operation(
        self,
        operation: str,
        entity_type: str
    ) -> None:
        """Record vector store operation metrics."""
        self.vector_store_operations.labels(operation=operation, entity_type=entity_type).inc()
    
    def set_model_versions_loaded(self, model_name: str, count: int) -> None:
        """Set the number of loaded model versions."""
        self.model_versions_loaded.labels(model_name=model_name).set(count)
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus exposition format for scraping."""
        return generate_latest(self.registry).decode('utf-8')
    
    def record_celery_task(
        self,
        task_name: str,
        status: str,
        duration: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record metrics for Celery task execution."""
        self.celery_task_invocations.labels(task_name=task_name, status=status).inc()
        if duration is not None:
            self.celery_task_duration.labels(task_name=task_name).observe(duration)
        
        if labels:
            lock_status = labels.get("lock_status")
            if lock_status is not None:
                self.celery_task_lock_status.labels(
                    task_name=task_name,
                    lock_status=lock_status
                ).inc()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(service_name: str) -> MetricsCollector:
    """Get or create metrics collector for a service.

    Returns a process‑wide singleton to avoid duplicate collectors/labels.
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(service_name)
    return _metrics_collector


def measure_time(operation: str, **labels: Any) -> Callable:
    """Decorator to measure function execution time.

    Example
    >>> @measure_time("inference", model="curve_forecaster")
    ... def predict(x):
    ...     ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                # Emit a structured log as a quick signal; metrics can be
                # recorded by callers using ``MetricsCollector`` if desired.
                logger.info(
                    f"Operation {operation} completed",
                    operation=operation,
                    duration_ms=duration * 1000,
                    **labels
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation {operation} failed",
                    operation=operation,
                    duration_ms=duration * 1000,
                    error=str(e),
                    **labels
                )
                raise
        return wrapper
    return decorator


def count_calls(operation: str, **labels: Any) -> Callable:
    """Decorator to count function calls.

    Lightweight helper for auditing or quick diagnostics. For production
    metrics prefer explicit counters to maintain clarity on label sets.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info(f"Calling {operation}", operation=operation, **labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator
