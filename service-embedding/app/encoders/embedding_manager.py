"""Embedding manager for handling model loading and embedding generation.

Loads and caches SentenceTransformer models, generates embeddings for inputs,
handles simple batch jobs, and participates in reindex workflows initiated
via events.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from contextlib import suppress
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import structlog

from libs.common.config import EmbeddingConfig
from libs.common.events import EventSubscriber, EventType
from libs.vector_store.factory import create_vector_store_from_env

logger = structlog.get_logger("embedding_service.embedding_manager")


class EmbeddingManager:
    """Manages embedding model loading and generation.

    Notes
    - Models are cached by name; metadata (dim, max length) kept in ``model_info``
    - Batch jobs are tracked in memory (simple first cut for UX/testing)
    - Vector store dependency is created from env to avoid tight coupling
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Create an embedding manager.

        Parameters
        - config: ``EmbeddingConfig`` containing model name and backends
        """
        self.config = config
        self.models: Dict[str, SentenceTransformer] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.batch_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize vector store
        env_config = {
            "ML_VECTOR_BACKEND": config.ml_vector_backend,
            "ML_VECTOR_DB_DSN": config.ml_vector_db_dsn,
            "ML_VECTOR_POOL_SIZE": "10",
            "ML_VECTOR_MAX_QUERIES": "50000",
            "ML_VECTOR_COMMAND_TIMEOUT": "60",
            "ML_VECTOR_DIMENSION": str(config.ml_vector_dimension),
            "ML_VECTOR_DUAL_WRITE": "true" if config.ml_vector_dual_write else "false",
            "ML_VECTOR_READ_FROM_SECONDARY": "true" if config.ml_vector_read_from_secondary else "false",
            "ML_OPENSEARCH_HOSTS": config.ml_opensearch_hosts,
            "ML_OPENSEARCH_INDEX": config.ml_opensearch_index,
            "ML_OPENSEARCH_USERNAME": config.ml_opensearch_username,
            "ML_OPENSEARCH_PASSWORD": config.ml_opensearch_password,
            "ML_OPENSEARCH_VERIFY_CERTS": "true" if config.ml_opensearch_verify_certs else "false",
            "ML_OPENSEARCH_SSL_ASSERT_HOSTNAME": "true" if config.ml_opensearch_ssl_assert_hostname else "false",
            "ML_OPENSEARCH_SSL_SHOW_WARN": "true" if config.ml_opensearch_ssl_show_warn else "false",
        }
        self.vector_store = create_vector_store_from_env(env_config)
        
        # Initialize event subscriber for reindex requests
        self.event_subscriber = EventSubscriber(config.ml_redis_url)
        self._setup_event_handlers()
        self._event_listener_task: Optional[asyncio.Task] = None
    
    def _setup_event_handlers(self):
        """Set up event handlers for reindex requests."""
        def handle_reindex_request(event_data: Dict[str, Any]):
            """Background handler to kick off reindex from event payload."""
            asyncio.create_task(self._handle_reindex_request(event_data))
        
        self.event_subscriber.subscribe(EventType.EMBEDDING_REINDEX_REQUEST, handle_reindex_request)
    
    async def initialize(self):
        """Initialize the embedding manager.

        Loads the default model on startup and starts the background event
        listener to react to reindex requests.
        """
        try:
            # Load default model
            await self._load_default_model()
            
            # Start event listener in background
            asyncio.create_task(self._start_event_listener())
            
            logger.info("Embedding manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize embedding manager", error=str(e))
            raise
    
    async def _load_default_model(self):
        """Load the default embedding model."""
        try:
            model_name = self.config.ml_embedding_model
            
            # Load the model
            model = SentenceTransformer(model_name)
            
            # Cache the model
            self.models[model_name] = model
            
            # Store model info
            self.model_info[model_name] = {
                "name": model_name,
                "version": "1.0.0",  # Could be determined from model
                "dimension": model.get_sentence_embedding_dimension(),
                "max_length": model.max_seq_length,
                "loaded_at": time.time()
            }
            
            logger.info("Loaded default embedding model", model_name=model_name)
            
        except Exception as e:
            logger.error("Failed to load default model", model_name=model_name, error=str(e))
            raise
    
    async def _start_event_listener(self):
        """Start the event listener in background."""
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
        """Handle embedding reindex request event."""
        try:
            entity_type = event_data.get("entity_type")
            batch_size = event_data.get("batch_size", 100)
            model_version = event_data.get("model_version", "default")
            tenant_id = event_data.get("tenant_id", "default")
            
            await self.trigger_reindex(entity_type, batch_size, tenant_id=tenant_id)
            
            logger.info(
                "Handled reindex request event",
                entity_type=entity_type,
                batch_size=batch_size,
                model_version=model_version,
                tenant_id=tenant_id
            )
            
        except Exception as e:
            logger.error("Failed to handle reindex request event", error=str(e))
    
    async def generate_embeddings(
        self,
        items: List[Dict[str, Any]],
        model_name: str = "default",
        batch_size: Optional[int] = None,
        tenant_id: str = "default"
    ) -> Tuple[List[List[float]], str]:
        """Generate embeddings for input items.

        Parameters
        - items: List of objects with a ``text`` field (dicts or strings)
        - model_name: Model identifier (defaults to configured default)
        - batch_size: Optional batch size; falls back to config value

        Returns
        - Tuple of ``(vectors, model_version)`` where ``vectors`` is a list of
          float lists (JSON‑serializable)
        """
        try:
            # Get the model
            model = self._get_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
            
            # Extract texts from items
            texts = []
            for item in items:
                if isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = str(item)
                texts.append(text)
            
            # Generate embeddings
            if batch_size is None:
                batch_size = self.config.ml_max_batch_size
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = model.encode(batch_texts)
                all_embeddings.extend(batch_embeddings.tolist())
            
            # Get model version
            model_version = self.model_info[model_name]["version"]
            
            logger.info(
                "Embeddings generated",
                model_name=model_name,
                count=len(all_embeddings),
                batch_size=batch_size,
                tenant_id=tenant_id
            )
            
            return all_embeddings, model_version
            
        except Exception as e:
            logger.error(
                "Failed to generate embeddings",
                model_name=model_name,
                tenant_id=tenant_id,
                error=str(e)
            )
            raise
    
    async def batch_generate_embeddings(
        self,
        job_id: str,
        items: List[Dict[str, Any]],
        model_name: str = "default",
        batch_size: Optional[int] = None,
        tenant_id: str = "default"
    ):
        """Start a batch embedding generation job.

        Simple in‑memory job execution for demo/PoC purposes. A production
        variant would enqueue work to a durable queue and write status to a DB.
        """
        try:
            # Initialize job
            self.batch_jobs[job_id] = {
                "status": "processing",
                "started_at": time.time(),
                "model_name": model_name,
                "item_count": len(items),
                "progress": 0,
                "tenant_id": tenant_id
            }
            
            # Process batch (simplified implementation)
            # In a real implementation, this would be queued and processed asynchronously
            vectors, model_version = await self.generate_embeddings(
                items,
                model_name,
                batch_size,
                tenant_id=tenant_id
            )
            
            # Update job status
            self.batch_jobs[job_id].update({
                "status": "completed",
                "completed_at": time.time(),
                "vectors": vectors,
                "model_version": model_version,
                "progress": 100
            })
            
            logger.info(
                "Batch embedding generation completed",
                job_id=job_id,
                tenant_id=tenant_id
            )
            
        except Exception as e:
            # Update job status to failed
            self.batch_jobs[job_id].update({
                "status": "failed",
                "failed_at": time.time(),
                "error": str(e),
                "progress": 0
            })
            
            logger.error(
                "Batch embedding generation failed",
                job_id=job_id,
                tenant_id=tenant_id,
                error=str(e)
            )
            raise
    
    async def trigger_reindex(
        self,
        entity_type: str,
        batch_size: int = 100,
        tenant_id: str = "default"
    ):
        """Trigger reindexing for a specific entity type.

        Placeholder implementation that logs the intent. The end‑to‑end flow
        is owned by the indexer worker which queries entities and persists
        results to the vector store.
        """
        try:
            # This would typically:
            # 1. Query the database for entities of the specified type
            # 2. Generate embeddings for each entity
            # 3. Store embeddings in the vector store
            # 4. Update search metadata
            
            logger.info(
                "Reindex triggered",
                entity_type=entity_type,
                batch_size=batch_size,
                tenant_id=tenant_id
            )
            
            # For now, just log the request
            # In a real implementation, this would start a background job
            
        except Exception as e:
            logger.error(
                "Reindex trigger failed",
                entity_type=entity_type,
                tenant_id=tenant_id,
                error=str(e)
            )
            raise
    
    def _get_model(self, model_name: str) -> Optional[SentenceTransformer]:
        """Get a model from cache.

        Loads the model on demand if not present and updates ``model_info``.
        """
        if model_name in self.models:
            return self.models[model_name]
        
        # Try to load the model if not cached
        try:
            model = SentenceTransformer(model_name)
            self.models[model_name] = model
            
            # Store model info
            self.model_info[model_name] = {
                "name": model_name,
                "version": "1.0.0",
                "dimension": model.get_sentence_embedding_dimension(),
                "max_length": model.max_seq_length,
                "loaded_at": time.time()
            }
            
            logger.info("Loaded embedding model", model_name=model_name)
            return model
            
        except Exception as e:
            logger.error("Failed to load model", model_name=model_name, error=str(e))
            return None
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        try:
            models = []
            for model_name, info in self.model_info.items():
                models.append({
                    "name": model_name,
                    "version": info["version"],
                    "dimension": info["dimension"],
                    "max_length": info["max_length"]
                })
            
            return models
            
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        try:
            if model_name not in self.model_info:
                return None
            
            info = self.model_info[model_name]
            return {
                "name": model_name,
                "version": info["version"],
                "dimension": info["dimension"],
                "max_length": info["max_length"],
                "loaded_at": info["loaded_at"]
            }
            
        except Exception as e:
            logger.error("Failed to get model info", model_name=model_name, error=str(e))
            raise
    
    async def get_batch_job_status(
        self,
        job_id: str,
        tenant_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Get status of a batch embedding job."""
        try:
            if job_id not in self.batch_jobs:
                return None
            
            job_metadata = self.batch_jobs[job_id]
            if job_metadata.get("tenant_id", "default") != tenant_id:
                logger.warning(
                    "Attempt to access batch job with mismatched tenant",
                    job_id=job_id,
                    expected_tenant=tenant_id,
                    job_tenant=job_metadata.get("tenant_id", "default")
                )
                return None
            
            job = job_metadata.copy()
            
            # Calculate duration
            if job["status"] == "completed":
                duration = job["completed_at"] - job["started_at"]
            elif job["status"] == "failed":
                duration = job["failed_at"] - job["started_at"]
            else:
                duration = time.time() - job["started_at"]
            
            job["duration_seconds"] = duration
            
            return job
            
        except Exception as e:
            logger.error("Failed to get batch job status", job_id=job_id, error=str(e))
            raise
    
    async def health_check(self) -> bool:
        """Check if the embedding manager is healthy."""
        try:
            # Check if we have at least one model loaded
            return len(self.models) > 0
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Cleanup resources.

        Clears model caches and closes the vector store if present.
        """
        try:
            # Clear model cache
            self.models.clear()
            self.model_info.clear()
            self.batch_jobs.clear()
            
            # Stop event listener task if running
            if self._event_listener_task and not self._event_listener_task.done():
                self._event_listener_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._event_listener_task
            
            # Close event subscriber connection
            with suppress(Exception):
                await self.event_subscriber.close()
            
            # Close vector store connection
            if hasattr(self.vector_store, 'close'):
                await self.vector_store.close()
            
            logger.info("Embedding manager cleanup completed")
            
        except Exception as e:
            logger.error("Embedding manager cleanup failed", error=str(e))
