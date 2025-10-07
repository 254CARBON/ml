"""Distributed tracing configuration for ML platform services.

Wraps OpenTelemetry setup for Jaeger with optional auto‑instrumentation
for FastAPI, SQLAlchemy, Redis, and HTTPX. Also provides small conveniences
for spans and scoped context managers used throughout the services.
"""

import os
from typing import Optional
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
import structlog

logger = structlog.get_logger("tracing")


def configure_tracing(
    service_name: str,
    jaeger_endpoint: str = "http://localhost:14268/api/traces",
    sample_rate: float = 1.0,
    enable_instrumentation: bool = True
) -> Optional[trace.Tracer]:
    """Configure distributed tracing for a service.

    Parameters
    - service_name: Logical service identifier used in trace resources
    - jaeger_endpoint: Collector endpoint for exporting spans
    - sample_rate: Reserved for future sampling configuration
    - enable_instrumentation: Toggle built‑in instrumentation hooks

    Returns
    - A tracer instance for ad‑hoc span creation, or ``None`` on failure
    """
    
    try:
        # Set up tracer provider
        tracer_provider = TracerProvider(
            resource=Resource.create({
                "service.name": service_name,
                "service.version": "0.1.0",
                "deployment.environment": os.getenv("ML_ENV", "local")
            })
        )
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=jaeger_endpoint,
            collector_endpoint=jaeger_endpoint
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Get tracer
        tracer = trace.get_tracer(service_name)
        
        # Enable automatic instrumentation
        if enable_instrumentation:
            try:
                # Instrument FastAPI
                FastAPIInstrumentor.instrument()
                
                # Instrument SQLAlchemy
                SQLAlchemyInstrumentor.instrument()
                
                # Instrument Redis
                RedisInstrumentor.instrument()
                
                # Instrument HTTPX
                HTTPXClientInstrumentor.instrument()
                
                logger.info("Automatic instrumentation enabled")
                
            except Exception as e:
                # Partial failure is acceptable; log but continue.
                logger.warning("Failed to enable some instrumentation", error=str(e))
        
        logger.info(
            "Distributed tracing configured",
            service_name=service_name,
            jaeger_endpoint=jaeger_endpoint,
            sample_rate=sample_rate
        )
        
        return tracer
        
    except Exception as e:
        logger.error("Failed to configure tracing", error=str(e))
        return None


def create_span(
    tracer: trace.Tracer,
    operation_name: str,
    **attributes
) -> trace.Span:
    """Create a new span with attributes.

    Use this for fine‑grained manual spans when a decorator is not suitable.
    """
    span = tracer.start_span(operation_name)
    
    # Add attributes
    for key, value in attributes.items():
        span.set_attribute(key, str(value))
    
    return span


def add_span_event(span: trace.Span, name: str, **attributes):
    """Add an event to a span."""
    span.add_event(name, attributes)


def set_span_status(span: trace.Span, status_code: trace.Status, description: str = ""):
    """Set span status."""
    span.set_status(trace.Status(status_code, description))


class TracingContext:
    """Context manager for tracing operations.

    Starts a span on entry and ensures it ends, recording success or error.
    """
    
    def __init__(self, tracer: trace.Tracer, operation_name: str, **attributes):
        self.tracer = tracer
        self.operation_name = operation_name
        self.attributes = attributes
        self.span: Optional[trace.Span] = None
    
    def __enter__(self):
        self.span = create_span(self.tracer, self.operation_name, **self.attributes)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type is not None:
                set_span_status(
                    self.span,
                    trace.Status.ERROR,
                    f"{exc_type.__name__}: {exc_val}"
                )
            else:
                set_span_status(self.span, trace.Status.OK)
            
            self.span.end()


def trace_function(operation_name: str = None, **span_attributes):
    """Decorator to trace function calls.

    Example
    >>> @trace_function("service.do_work", component="scheduler")
    ... def do_work():
    ...     ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get tracer from global context
            tracer = trace.get_tracer(__name__)
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with TracingContext(tracer, op_name, **span_attributes) as span:
                # Add function parameters as attributes
                if args:
                    span.set_attribute("function.args_count", len(args))
                if kwargs:
                    for key, value in kwargs.items():
                        span.set_attribute(f"function.param.{key}", str(value))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.result_type", type(result).__name__)
                    return result
                except Exception as e:
                    span.set_attribute("function.error", str(e))
                    raise
        
        return wrapper
    return decorator


async def trace_async_function(operation_name: str = None, **span_attributes):
    """Decorator to trace async function calls.

    Mirrors ``trace_function`` but awaits the underlying coroutine.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get tracer from global context
            tracer = trace.get_tracer(__name__)
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with TracingContext(tracer, op_name, **span_attributes) as span:
                # Add function parameters as attributes
                if args:
                    span.set_attribute("function.args_count", len(args))
                if kwargs:
                    for key, value in kwargs.items():
                        span.set_attribute(f"function.param.{key}", str(value))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.result_type", type(result).__name__)
                    return result
                except Exception as e:
                    span.set_attribute("function.error", str(e))
                    raise
        
        return wrapper
    return decorator


class MLTracer:
    """ML-specific tracing utilities.

    Small helpers for common ML operations so span names and attributes remain
    consistent across services and dashboards.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = trace.get_tracer(service_name)
    
    def trace_model_inference(
        self,
        model_name: str,
        model_version: str,
        input_count: int,
        **attributes
    ):
        """Trace model inference operation."""
        return TracingContext(
            self.tracer,
            "model.inference",
            model_name=model_name,
            model_version=model_version,
            input_count=input_count,
            **attributes
        )
    
    def trace_embedding_generation(
        self,
        model_name: str,
        entity_type: str,
        batch_size: int,
        **attributes
    ):
        """Trace embedding generation operation."""
        return TracingContext(
            self.tracer,
            "embedding.generation",
            model_name=model_name,
            entity_type=entity_type,
            batch_size=batch_size,
            **attributes
        )
    
    def trace_search_query(
        self,
        query_type: str,
        semantic: bool,
        limit: int,
        **attributes
    ):
        """Trace search query operation."""
        return TracingContext(
            self.tracer,
            "search.query",
            query_type=query_type,
            semantic=semantic,
            limit=limit,
            **attributes
        )
    
    def trace_vector_operation(
        self,
        operation: str,
        entity_type: str,
        count: int = 1,
        **attributes
    ):
        """Trace vector store operation."""
        return TracingContext(
            self.tracer,
            "vector_store.operation",
            operation=operation,
            entity_type=entity_type,
            count=count,
            **attributes
        )
    
    def trace_event_processing(
        self,
        event_type: str,
        **attributes
    ):
        """Trace event processing operation."""
        return TracingContext(
            self.tracer,
            "event.processing",
            event_type=event_type,
            **attributes
        )


def get_ml_tracer(service_name: str) -> MLTracer:
    """Get ML-specific tracer for a service."""
    return MLTracer(service_name)
