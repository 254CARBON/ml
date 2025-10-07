"""Search service main application."""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from .api.routes import router as api_router
from .hybrid.search_manager import SearchManager
from .runtime.metrics import get_metrics_collector
from libs.common.config import SearchConfig
from libs.common.logging import configure_logging
from libs.common.tracing import configure_tracing
from libs.common.events import create_event_publisher

logger = structlog.get_logger("search_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    config = SearchConfig()
    configure_logging("search-service", config.ml_log_level, config.ml_log_format)
    
    # Initialize tracing
    if config.ml_tracing_enabled:
        tracer = configure_tracing("search-service", config.ml_otel_exporter)
        if tracer:
            logger.info("OpenTelemetry tracing enabled", exporter=config.ml_otel_exporter)
        else:
            logger.warning("Tracing initialization failed")
    else:
        tracer = None
        logger.info("OpenTelemetry tracing disabled via configuration")
    app.state.tracer = tracer
    
    logger.info("Starting search service")
    
    # Initialize search manager
    app.state.search_manager = SearchManager(config)
    await app.state.search_manager.initialize()
    
    # Initialize metrics collector
    app.state.metrics_collector = get_metrics_collector("search-service")
    
    # Initialize event publisher
    app.state.event_publisher = create_event_publisher(config.ml_redis_url)
    
    logger.info("Search service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down search service")
    if hasattr(app.state, 'search_manager'):
        await app.state.search_manager.cleanup()
    logger.info("Search service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Search Service",
    description="Hybrid semantic and lexical search service",
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
        # Check search manager health
        if hasattr(app.state, 'search_manager'):
            search_health = await app.state.search_manager.health_check()
        else:
            search_health = False
        
        if search_health:
            return {"status": "healthy", "service": "search-service"}
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "service": "search-service"}
            )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "search-service", "error": str(e)}
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if hasattr(app.state, 'metrics_collector'):
        metrics_data = app.state.metrics_collector.get_metrics()
        return Response(content=metrics_data, media_type="text/plain")
    else:
        return Response(content="# No metrics available\n", media_type="text/plain")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "search-service",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "search": "/api/v1/search",
            "index": "/api/v1/index"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=9007,
        reload=True,
        log_level="info"
    )
