"""API routes for search service."""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from pydantic import BaseModel, Field
import structlog

from ..hybrid.search_manager import SearchManager
from ..runtime.metrics import get_metrics_collector
from libs.common.events import SearchIndexUpdatedEvent

logger = structlog.get_logger("search_service.api")

router = APIRouter()


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query")
    semantic: bool = Field(True, description="Enable semantic search")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    limit: int = Field(10, description="Maximum number of results")
    similarity_threshold: float = Field(0.0, description="Minimum similarity threshold")


class SearchResult(BaseModel):
    """Search result model."""
    entity_type: str = Field(..., description="Entity type")
    entity_id: str = Field(..., description="Entity ID")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(..., description="Entity metadata")
    text: str = Field(..., description="Entity text")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: List[SearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original query")
    semantic: bool = Field(..., description="Semantic search enabled")
    latency_ms: float = Field(..., description="Search latency in milliseconds")


class IndexRequest(BaseModel):
    """Request model for index endpoint."""
    entity_type: str = Field(..., description="Entity type")
    entity_id: str = Field(..., description="Entity ID")
    text: str = Field(..., description="Entity text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Entity metadata")
    tags: Optional[List[str]] = Field(None, description="Entity tags")


class IndexResponse(BaseModel):
    """Response model for index endpoint."""
    status: str = Field(..., description="Indexing status")
    message: str = Field(..., description="Status message")
    entity_type: str = Field(..., description="Entity type")
    entity_id: str = Field(..., description="Entity ID")


def get_search_manager(request: Request) -> SearchManager:
    """Get search manager from application state."""
    return request.app.state.search_manager


def get_metrics(request: Request):
    """Get metrics collector from application state."""
    return request.app.state.metrics_collector


def get_event_publisher(request: Request):
    """Get event publisher from application state."""
    return request.app.state.event_publisher


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    tenant_id: Optional[str] = Query(None, description="Optional tenant ID for multi-tenant search"),
    search_manager: SearchManager = Depends(get_search_manager),
    metrics_collector = Depends(get_metrics),
    event_publisher = Depends(get_event_publisher)
):
    """Perform hybrid search."""
    start_time = time.time()
    
    try:
        # Perform search
        results = await search_manager.search(
            query=request.query,
            semantic=request.semantic,
            filters=request.filters,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            tenant_id=tenant_id
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        query_type = "semantic" if request.semantic else "lexical"
        metrics_collector.record_search(
            query_type=query_type,
            duration=time.time() - start_time
        )
        
        # Convert results to response format
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                entity_type=result[0],
                entity_id=result[1],
                score=result[2],
                metadata=result[3],
                text=result[3].get("text", "")
            ))
        
        logger.info(
            "Search completed",
            query=request.query,
            semantic=request.semantic,
            results_count=len(search_results),
            latency_ms=latency_ms
        )
        
        return SearchResponse(
            results=search_results,
            total=len(search_results),
            query=request.query,
            semantic=request.semantic,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error("Search failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/index", response_model=IndexResponse)
async def index_entity(
    request: IndexRequest,
    tenant_id: Optional[str] = Query(None, description="Optional tenant ID for multi-tenant indexing"),
    search_manager: SearchManager = Depends(get_search_manager),
    event_publisher = Depends(get_event_publisher)
):
    """Index an entity for search."""
    try:
        # Index the entity
        await search_manager.index_entity(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            text=request.text,
            metadata=request.metadata,
            tags=request.tags,
            tenant_id=tenant_id
        )
        
        # Publish index updated event
        event_publisher.publish_search_index_updated(
            entity_type=request.entity_type,
            count=1
        )
        
        logger.info(
            "Entity indexed",
            entity_type=request.entity_type,
            entity_id=request.entity_id
        )
        
        return IndexResponse(
            status="success",
            message="Entity indexed successfully",
            entity_type=request.entity_type,
            entity_id=request.entity_id
        )
        
    except Exception as e:
        logger.error("Indexing failed", entity_type=request.entity_type, entity_id=request.entity_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.delete("/index/{entity_type}/{entity_id}")
async def remove_entity(
    entity_type: str,
    entity_id: str,
    search_manager: SearchManager = Depends(get_search_manager)
):
    """Remove an entity from the search index."""
    try:
        await search_manager.remove_entity(entity_type, entity_id)
        
        logger.info("Entity removed from index", entity_type=entity_type, entity_id=entity_id)
        
        return {"status": "success", "message": "Entity removed from index"}
        
    except Exception as e:
        logger.error("Failed to remove entity", entity_type=entity_type, entity_id=entity_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to remove entity: {str(e)}")


@router.get("/index/stats")
async def get_index_stats(
    search_manager: SearchManager = Depends(get_search_manager)
):
    """Get search index statistics."""
    try:
        stats = await search_manager.get_index_stats()
        
        logger.info("Index stats retrieved")
        return stats
        
    except Exception as e:
        logger.error("Failed to get index stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get index stats: {str(e)}")


@router.post("/index/rebuild")
async def rebuild_index(
    entity_type: Optional[str] = None,
    search_manager: SearchManager = Depends(get_search_manager)
):
    """Rebuild the search index."""
    try:
        await search_manager.rebuild_index(entity_type)
        
        logger.info("Index rebuild triggered", entity_type=entity_type)
        return {"status": "success", "message": "Index rebuild triggered"}
        
    except Exception as e:
        logger.error("Index rebuild failed", entity_type=entity_type, error=str(e))
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")
