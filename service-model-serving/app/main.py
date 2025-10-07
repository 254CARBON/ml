"""Model serving service main application."""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from .api.routes import router as api_router
from .runtime.model_manager import ModelManager
from .runtime.ensemble_manager import create_ensemble_manager
from .runtime.ensemble_manager import create_shadow_deployment_manager
from .runtime.metrics import get_metrics_collector
from libs.common.config import ModelServingConfig
from libs.common.logging import configure_logging
from libs.common.tracing import configure_tracing
from libs.common.events import create_event_publisher, ModelPromotedEvent

logger = structlog.get_logger("model_serving")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    config = ModelServingConfig()
    configure_logging("model-serving", config.ml_log_level, config.ml_log_format)
    
    # Initialize tracing
    if config.ml_tracing_enabled:
        tracer = configure_tracing("model-serving", config.ml_otel_exporter)
        if tracer:
            logger.info("OpenTelemetry tracing enabled", exporter=config.ml_otel_exporter)
        else:
            logger.warning("Tracing initialization failed")
    else:
        tracer = None
        logger.info("OpenTelemetry tracing disabled via configuration")
    app.state.tracer = tracer
    
    logger.info("Starting model serving service")
    
    # Initialize model manager
    app.state.model_manager = ModelManager(config)
    await app.state.model_manager.initialize()
    
    # Initialize shadow deployment manager
    app.state.shadow_deployment_manager = create_shadow_deployment_manager(
        config=config,
        model_manager=app.state.model_manager
    )
    
    # Initialize ensemble manager with dependencies
    app.state.ensemble_manager = create_ensemble_manager(
        config=config,
        model_manager=app.state.model_manager,
        shadow_manager=app.state.shadow_deployment_manager
    )
    
    # Initialize metrics collector
    app.state.metrics_collector = get_metrics_collector("model-serving")
    
    # Initialize event publisher
    app.state.event_publisher = create_event_publisher(config.ml_redis_url)
    
    logger.info("Model serving service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down model serving service")
    if hasattr(app.state, 'model_manager'):
        await app.state.model_manager.cleanup()
    logger.info("Model serving service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Model Serving Service",
    description="Production model serving with REST API endpoints",
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
        # Check model manager health
        if hasattr(app.state, 'model_manager'):
            model_health = await app.state.model_manager.health_check()
        else:
            model_health = False
        
        if model_health:
            return {"status": "healthy", "service": "model-serving"}
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "service": "model-serving"}
            )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "model-serving", "error": str(e)}
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if hasattr(app.state, 'metrics_collector'):
        metrics_data = app.state.metrics_collector.get_metrics()
        return Response(content=metrics_data, media_type="text/plain")
    else:
        return Response(content="# No metrics available\n", media_type="text/plain")


@app.post("/reload")
async def reload_models():
    """Force reload all models."""
    try:
        if hasattr(app.state, 'model_manager'):
            await app.state.model_manager.reload_all_models()
            return {"status": "success", "message": "Models reloaded"}
        else:
            raise HTTPException(status_code=503, detail="Model manager not available")
    except Exception as e:
        logger.error("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "model-serving",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "predict": "/api/v1/predict",
            "batch": "/api/v1/batch",
            "models": "/api/v1/models",
            "reload": "/reload"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=9005,
        reload=True,
        log_level="info"
    )
