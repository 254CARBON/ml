"""Embedding service main application."""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from .api.routes import router as api_router
from .encoders.embedding_manager import EmbeddingManager
from .runtime.metrics import get_metrics_collector
from libs.common.config import EmbeddingConfig
from libs.common.logging import configure_logging
from libs.common.tracing import configure_tracing
from libs.common.events import create_event_publisher

logger = structlog.get_logger("embedding_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    config = EmbeddingConfig()
    configure_logging("embedding-service", config.ml_log_level, config.ml_log_format)
    app.state.config = config
    app.state.startup_time = time.time()
    
    # Initialize tracing
    if config.ml_tracing_enabled:
        tracer = configure_tracing("embedding-service", config.ml_otel_exporter)
        if tracer:
            logger.info("OpenTelemetry tracing enabled", exporter=config.ml_otel_exporter)
        else:
            logger.warning("Tracing initialization failed")
    else:
        tracer = None
        logger.info("OpenTelemetry tracing disabled via configuration")
    app.state.tracer = tracer
    
    logger.info("Starting embedding service")
    
    # Initialize embedding manager
    app.state.embedding_manager = EmbeddingManager(config)
    await app.state.embedding_manager.initialize()
    
    # Initialize metrics collector
    app.state.metrics_collector = get_metrics_collector("embedding-service")
    
    # Initialize event publisher
    app.state.event_publisher = create_event_publisher(config.ml_redis_url)
    
    logger.info("Embedding service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down embedding service")
    if hasattr(app.state, 'embedding_manager'):
        await app.state.embedding_manager.cleanup()
    logger.info("Embedding service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Embedding Service",
    description="Embedding generation service with batch and on-demand processing",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics for HTTP requests."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        response = JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )
    
    duration = time.time() - start_time
    
    # Record metrics
    if hasattr(app.state, 'metrics_collector'):
        app.state.metrics_collector.record_http_request(
            method=request.method,
            endpoint=request.url.path,
            status=status_code,
            duration=duration
        )
    
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check embedding manager health
        if hasattr(app.state, 'embedding_manager'):
            embedding_health = await app.state.embedding_manager.health_check()
        else:
            embedding_health = False
        
        if embedding_health:
            return {"status": "healthy", "service": "embedding-service"}
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "service": "embedding-service"}
            )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "embedding-service", "error": str(e)}
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if hasattr(app.state, 'metrics_collector'):
        metrics_data = app.state.metrics_collector.get_metrics()
        return Response(content=metrics_data, media_type="text/plain")
    else:
        return Response(content="# No metrics available\n", media_type="text/plain")


@app.get("/live")
async def liveness():
    """Liveness probe. Returns quickly if process is responsive."""
    return {
        "status": "alive",
        "service": "embedding-service",
        "uptime_seconds": time.time() - getattr(app.state, "startup_time", time.time())
    }


@app.get("/ready")
async def readiness():
    """Readiness probe. Validates core dependencies are available."""
    try:
        if not hasattr(app.state, "embedding_manager"):
            raise RuntimeError("Embedding manager not initialized")
        
        is_healthy = await app.state.embedding_manager.health_check()
        if not is_healthy:
            raise RuntimeError("Embedding manager reports unhealthy state")
        
        return {
            "status": "ready",
            "service": "embedding-service",
            "models_loaded": len(app.state.embedding_manager.models)
        }
    except Exception as exc:
        logger.error("Readiness probe failed", error=str(exc))
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "embedding-service",
                "error": str(exc)
            }
        )


@app.get("/resources")
async def resources():
    """Describe service resources and configured backends."""
    config: Optional[EmbeddingConfig] = getattr(app.state, "config", None)
    embedding_manager: Optional[EmbeddingManager] = getattr(app.state, "embedding_manager", None)
    
    models: List[Dict[str, Any]] = []
    if embedding_manager:
        try:
            models = await embedding_manager.list_models()
        except Exception as exc:
            logger.warning("Failed to list models for resources endpoint", error=str(exc))
    
    return {
        "service": "embedding-service",
        "models": models,
        "vector_backend": config.ml_vector_backend if config else None,
        "vector_dimension": config.ml_vector_dimension if config else None,
        "events": {
            "consumes": ["ml.embedding.reindex.request.v1"],
            "produces": ["ml.embedding.generated.v1"]
        }
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "embedding-service",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "embed": "/api/v1/embed",
            "batch": "/api/v1/batch",
            "models": "/api/v1/models"
        },
        "probes": {
            "health": "/health",
            "live": "/live",
            "ready": "/ready"
        },
        "metadata": {
            "resources": "/resources"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=9006,
        reload=True,
        log_level="info"
    )
