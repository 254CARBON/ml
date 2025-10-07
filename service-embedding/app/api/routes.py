"""API routes for embedding service."""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from pydantic import BaseModel, Field
import structlog

from ..encoders.embedding_manager import EmbeddingManager
from ..runtime.metrics import get_metrics_collector
from libs.common.auth import TenantContext, create_optional_auth_dependency
from libs.common.config import EmbeddingConfig
from libs.common.security import InputSanitizer

logger = structlog.get_logger("embedding_service.api")

router = APIRouter()

_config = EmbeddingConfig()
_optional_auth_dependency = create_optional_auth_dependency(_config)
_sanitizer = InputSanitizer()


class EmbedRequest(BaseModel):
    """Request model for embedding endpoint."""
    items: List[Dict[str, Any]] = Field(..., description="Items to embed")
    model: Optional[str] = Field(None, description="Specific model to use")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")
    tenant_id: Optional[str] = Field(None, description="Tenant scope for the embedding operation")


class EmbedResponse(BaseModel):
    """Response model for embedding endpoint."""
    model_version: str = Field(..., description="Model version used")
    vectors: List[List[float]] = Field(..., description="Generated embeddings")
    count: int = Field(..., description="Number of embeddings generated")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class BatchEmbedRequest(BaseModel):
    """Request model for batch embedding endpoint."""
    items: List[Dict[str, Any]] = Field(..., description="Items to embed")
    model: Optional[str] = Field(None, description="Specific model to use")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")
    tenant_id: Optional[str] = Field(None, description="Tenant scope for the batch operation")


class BatchEmbedResponse(BaseModel):
    """Response model for batch embedding endpoint."""
    job_id: str = Field(..., description="Batch job ID")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


class ModelInfo(BaseModel):
    """Model information model."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    dimension: int = Field(..., description="Embedding dimension")
    max_length: int = Field(..., description="Maximum input length")


def resolve_tenant_scope(
    explicit_tenant_id: Optional[str],
    auth_context: Optional[Dict[str, Any]]
) -> str:
    """Resolve tenant scope from request payload and optional auth context."""
    sanitized_explicit: Optional[str] = None
    if explicit_tenant_id:
        try:
            sanitized_explicit = _sanitizer.validate_tenant_id(explicit_tenant_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    
    token_tenant: Optional[str] = None
    if auth_context:
        token_tenant = auth_context.get("tenant_id")
    
    if token_tenant:
        try:
            sanitized_token = _sanitizer.validate_tenant_id(token_tenant)
        except ValueError:
            logger.warning("Invalid tenant in auth token", tenant_id=token_tenant)
            raise HTTPException(status_code=400, detail="Invalid tenant in auth token")
        
        if sanitized_explicit and sanitized_explicit != sanitized_token:
            raise HTTPException(
                status_code=403,
                detail="Tenant mismatch between token and request"
            )
        return sanitized_token
    
    if sanitized_explicit:
        return sanitized_explicit
    
    return "default"


def ensure_items_match_tenant(items: List[Dict[str, Any]], tenant_id: str) -> None:
    """Ensure all request items align with resolved tenant scope."""
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_tenant = item.get("tenant_id")
        if item_tenant and item_tenant != tenant_id:
            raise HTTPException(
                status_code=403,
                detail=f"Item at position {index} has mismatched tenant"
            )


def get_embedding_manager(request: Request) -> EmbeddingManager:
    """Get embedding manager from application state."""
    return request.app.state.embedding_manager


def get_metrics(request: Request):
    """Get metrics collector from application state."""
    return request.app.state.metrics_collector


def get_event_publisher(request: Request):
    """Get event publisher from application state."""
    return request.app.state.event_publisher


@router.post("/embed", response_model=EmbedResponse)
async def embed(
    request: EmbedRequest,
    auth_context: Optional[Dict[str, Any]] = Depends(_optional_auth_dependency),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    metrics_collector = Depends(get_metrics),
    event_publisher = Depends(get_event_publisher)
):
    """Generate embeddings for input items."""
    start_time = time.time()
    
    try:
        tenant_id = resolve_tenant_scope(request.tenant_id, auth_context)
        ensure_items_match_tenant(request.items, tenant_id)
        
        # Get model name
        model_name = request.model or "default"
        
        # Generate embeddings under tenant scope
        with TenantContext(tenant_id):
            vectors, model_version = await embedding_manager.generate_embeddings(
                items=request.items,
                model_name=model_name,
                batch_size=request.batch_size,
                tenant_id=tenant_id
            )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        metrics_collector.record_embedding(
            model_name=model_name,
            entity_type="mixed",  # Could be determined from items
            duration=time.time() - start_time
        )
        
        # Publish events for each generated embedding
        for i, item in enumerate(request.items):
            entity_type = item.get("type", "unknown")
            entity_id = item.get("id", f"item_{i}")
            event_publisher.publish_embedding_generated(
                entity_type=entity_type,
                entity_id=entity_id,
                model_version=model_version,
                tenant_id=tenant_id
            )
        
        logger.info(
            "Embeddings generated",
            model_name=model_name,
            model_version=model_version,
            count=len(vectors),
            latency_ms=latency_ms,
            tenant_id=tenant_id
        )
        
        return EmbedResponse(
            model_version=model_version,
            vectors=vectors,
            count=len(vectors),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error("Embedding generation failed", error=str(e), tenant_id=locals().get("tenant_id"))
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@router.post("/batch", response_model=BatchEmbedResponse)
async def batch_embed(
    request: BatchEmbedRequest,
    auth_context: Optional[Dict[str, Any]] = Depends(_optional_auth_dependency),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """Start a batch embedding job."""
    try:
        tenant_id = resolve_tenant_scope(request.tenant_id, auth_context)
        ensure_items_match_tenant(request.items, tenant_id)
        
        # Generate job ID
        job_hash = hash(str(request.items)) % 10000
        job_id = f"embed_batch_{tenant_id}_{int(time.time())}_{job_hash}"
        
        # Start batch job (simplified implementation)
        # In a real implementation, this would queue the job
        await embedding_manager.batch_generate_embeddings(
            job_id=job_id,
            items=request.items,
            model_name=request.model or "default",
            batch_size=request.batch_size,
            tenant_id=tenant_id
        )
        
        logger.info(
            "Batch embedding job started",
            job_id=job_id,
            item_count=len(request.items),
            tenant_id=tenant_id
        )
        
        return BatchEmbedResponse(
            job_id=job_id,
            status="queued",
            message="Batch embedding job queued successfully"
        )
        
    except Exception as e:
        logger.error("Batch embedding failed", error=str(e), tenant_id=locals().get("tenant_id"))
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")


@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """List available embedding models."""
    try:
        models = await embedding_manager.list_models()
        
        model_info = []
        for model in models:
            model_info.append(ModelInfo(
                name=model["name"],
                version=model["version"],
                dimension=model["dimension"],
                max_length=model["max_length"]
            ))
        
        logger.info("Models listed", count=len(model_info))
        return model_info
        
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_name}")
async def get_model_info(
    model_name: str,
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """Get information about a specific model."""
    try:
        model_info = await embedding_manager.get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        logger.info("Model info retrieved", model_name=model_name)
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model info", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_batch_job_status(
    job_id: str,
    tenant_id: Optional[str] = Query(None, description="Tenant scope for job lookup"),
    auth_context: Optional[Dict[str, Any]] = Depends(_optional_auth_dependency),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """Get status of a batch embedding job."""
    try:
        resolved_tenant = resolve_tenant_scope(tenant_id, auth_context)
        job_status = await embedding_manager.get_batch_job_status(job_id, tenant_id=resolved_tenant)
        
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        logger.info("Batch job status retrieved", job_id=job_id, tenant_id=resolved_tenant)
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get batch job status",
            job_id=job_id,
            tenant_id=locals().get("resolved_tenant"),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.post("/reindex")
async def trigger_reindex(
    entity_type: str,
    batch_size: int = 100,
    tenant_id: Optional[str] = Query(None, description="Tenant scope for reindex request"),
    auth_context: Optional[Dict[str, Any]] = Depends(_optional_auth_dependency),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """Trigger reindexing for a specific entity type."""
    try:
        resolved_tenant = resolve_tenant_scope(tenant_id, auth_context)
        await embedding_manager.trigger_reindex(entity_type, batch_size, tenant_id=resolved_tenant)
        
        logger.info(
            "Reindex triggered",
            entity_type=entity_type,
            batch_size=batch_size,
            tenant_id=resolved_tenant
        )
        return {"status": "success", "message": f"Reindex triggered for {entity_type}"}
        
    except Exception as e:
        logger.error(
            "Reindex trigger failed",
            entity_type=entity_type,
            tenant_id=locals().get("resolved_tenant"),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Reindex trigger failed: {str(e)}")
