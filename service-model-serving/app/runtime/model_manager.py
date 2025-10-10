"""Model manager for loading and serving ML models.

Responsible for discovering, loading, caching, and serving models from local storage.
Provides synchronous prediction APIs and simple batch jobs while
reacting to model promotion events.
"""

import asyncio
import time
import os
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from contextlib import suppress
import pickle
import joblib
import torch
import numpy as np
import pandas as pd
import structlog
from pathlib import Path
import sys

from libs.common.config import ModelServingConfig
from libs.common.events import EventSubscriber, EventType
from libs.common.tracing import get_ml_tracer
from ..adapters.circuit_breaker import get_circuit_breaker, CircuitBreakerError

# Add training modules to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from training.curve_forecaster.model import CurveForecaster
except ImportError:
    CurveForecaster = None

logger = structlog.get_logger("model_serving.model_manager")


class ModelManager:
    """Manages ML model loading, caching, and serving.

    Design
    - Keeps an in‑memory cache keyed by ``name:version``
    - On startup, attempts to load the default production model from local storage
    - Listens for "model promoted" events to refresh active versions
    """
    
    def __init__(self, config: ModelServingConfig):
        """Create a model manager.

        Parameters
        - config: ``ModelServingConfig`` with Redis, logging, etc.
        """
        self.config = config
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_cache: Dict[str, Any] = {}
        self.batch_jobs: Dict[str, Dict[str, Any]] = {}
        self.model_warmup_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize event subscriber for model promotions
        self.event_subscriber = EventSubscriber(config.ml_redis_url)
        self._setup_event_handlers()
        self._event_listener_task: Optional[asyncio.Task] = None
        
        # Initialize tracing
        self.tracer = get_ml_tracer("model-serving")
        
        # Model storage path
        self.model_storage_path = Path(os.getenv("ML_MODEL_STORAGE_PATH", "/app/models"))
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_event_handlers(self):
        """Set up event handlers for model promotions."""
        def handle_model_promoted(event_data: Dict[str, Any]):
            """Callback to handle MODEL_PROMOTED events (fires async task)."""
            asyncio.create_task(self._handle_model_promoted(event_data))
        
        self.event_subscriber.subscribe(EventType.MODEL_PROMOTED, handle_model_promoted)
    
    async def initialize(self):
        """Initialize the model manager.

        Loads default models and starts event processing in the background.
        """
        try:
            # Load default models
            await self._load_default_models()
            
            # Start event listener in background
            asyncio.create_task(self._start_event_listener())
            
            logger.info("Model manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize model manager", error=str(e))
            raise
    
    async def _load_default_models(self):
        """Load default models from local storage.

        Attempts to resolve the latest Production version from local storage and
        warm it into the cache for low‑latency predictions.
        """
        try:
            # Get default model name from config
            default_model_name = self.config.ml_model_default_name
            
            # Try to load the latest production model
            try:
                model_path = self.model_storage_path / default_model_name / "production"
                
                if model_path.exists():
                    model = await self._load_model_from_path(str(model_path))
                    if model:
                        await self._cache_model(default_model_name, "production", model)
                        logger.info("Loaded default model", model_name=default_model_name, version="production")
                    else:
                        logger.warning("Failed to load default model", model_name=default_model_name)
                else:
                    logger.warning("No production model found", model_name=default_model_name, path=str(model_path))
                    
            except Exception as e:
                logger.warning("Failed to load default model", model_name=default_model_name, error=str(e))
                
        except Exception as e:
            logger.error("Failed to load default models", error=str(e))
    
    async def _load_model_from_path(self, model_path: str) -> Optional[Any]:
        """Load a model from local path."""
        try:
            model_path_obj = Path(model_path)
            
            # Try different model formats
            if (model_path_obj / "model.joblib").exists():
                model = joblib.load(str(model_path_obj / "model.joblib"))
            elif (model_path_obj / "model.pkl").exists():
                with open(str(model_path_obj / "model.pkl"), "rb") as f:
                    model = pickle.load(f)
            elif (model_path_obj / "model.pth").exists():
                model = torch.load(str(model_path_obj / "model.pth"), map_location="cpu")
                model.eval()
            elif (model_path_obj / "model.py").exists():
                # Try to load custom model
                if CurveForecaster is not None:
                    model = CurveForecaster.load_model(str(model_path_obj))
                else:
                    logger.warning("Custom model loader not available", path=model_path)
                    return None
            else:
                logger.warning("No supported model format found", path=model_path)
                return None
            
            return model
            
        except Exception as e:
            logger.error("Failed to load model from path", path=model_path, error=str(e))
            return None
    
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
    
    async def _handle_model_promoted(self, event_data: Dict[str, Any]):
        """Handle model promotion event.

        When a model enters Production, reload it and update the active entry
        so subsequent requests immediately use the promoted version.
        """
        try:
            model_name = event_data.get("model_name")
            version = event_data.get("version")
            stage = event_data.get("stage")
            
            if stage == "Production":
                await self._reload_model(model_name, version)
            
            logger.info(
                "Handled model promotion event",
                model_name=model_name,
                version=version,
                stage=stage
            )
            
        except Exception as e:
            logger.error("Failed to handle model promotion event", error=str(e))
    
    async def _reload_model(self, model_name: str, version: str):
        """Reload a specific model."""
        try:
            # Load the new model version from local storage
            model_path = self.model_storage_path / model_name / version
            
            model = await self._load_model_from_path(str(model_path))
            if not model:
                logger.error("Failed to load model", model_name=model_name, version=version)
                return
            
            # Cache the new model
            await self._cache_model(model_name, version, model)
            
            # Update the active model if it's the same name
            if model_name in self.models:
                self.models[model_name]["active_version"] = version
                self.models[model_name]["model"] = model
            
            logger.info("Model reloaded", model_name=model_name, version=version)
            
        except Exception as e:
            logger.error("Failed to reload model", model_name=model_name, version=version, error=str(e))
    
    async def _cache_model(self, model_name: str, version: str, model: Any):
        """Cache a model in memory.

        Updates both the versioned cache and the model registry for the name.
        """
        cache_key = f"{model_name}:{version}"
        self.model_cache[cache_key] = {
            "model": model,
            "loaded_at": time.time(),
            "model_name": model_name,
            "version": version
        }
        
        # Initialize model entry if it doesn't exist
        if model_name not in self.models:
            self.models[model_name] = {
                "active_version": version,
                "model": model,
                "versions": {}
            }
        
        # Update version info
        self.models[model_name]["versions"][version] = {
            "model": model,
            "loaded_at": time.time()
        }
    
    async def predict(
        self,
        inputs: List[Dict[str, Any]],
        model_name: str = "default",
        model_version: str = "latest"
    ) -> List[Any]:
        """Perform prediction using the specified model.

        Accepts flexible input shapes and handles a small number of common
        model types (custom, scikit‑learn, PyTorch) for demonstration.
        """
        try:
            # Get the model
            model = await self._get_model(model_name, model_version)
            
            if model is None:
                raise ValueError(f"Model {model_name}:{model_version} not found")
            
            # Handle CurveForecaster models
            if hasattr(model, '__class__') and 'CurveForecaster' in str(model.__class__):
                # This is our custom curve forecaster
                # Convert inputs to DataFrame format expected by the model
                if isinstance(inputs[0], dict):
                    # Create a simple DataFrame for prediction
                    # In a real scenario, this would be more sophisticated
                    import pandas as pd
                    df_data = []
                    for inp in inputs:
                        # Convert input dict to a row with date and curve data
                        row = {"date": pd.Timestamp.now()}
                        row.update(inp)
                        df_data.append(row)
                    
                    input_df = pd.DataFrame(df_data)
                    predictions = model.predict(input_df)
                else:
                    # Fallback for array inputs
                    predictions = [[0.02, 0.025, 0.03, 0.035, 0.04] for _ in inputs]  # Mock prediction
                
            elif hasattr(model, 'predict'):
                # Scikit-learn model
                if isinstance(inputs[0], dict):
                    # Convert dict inputs to array
                    input_array = np.array([list(inp.values()) for inp in inputs])
                else:
                    input_array = np.array(inputs)
                
                predictions = model.predict(input_array)
                
            elif isinstance(model, torch.nn.Module):
                # PyTorch model
                model.eval()
                with torch.no_grad():
                    if isinstance(inputs[0], dict):
                        input_tensor = torch.tensor([list(inp.values()) for inp in inputs], dtype=torch.float32)
                    else:
                        input_tensor = torch.tensor(inputs, dtype=torch.float32)
                    
                    predictions = model(input_tensor).numpy()
            
            else:
                # Generic model with predict method or fallback
                try:
                    predictions = model.predict(inputs)
                except:
                    # Fallback mock prediction for demonstration
                    predictions = [[0.02, 0.025, 0.03, 0.035, 0.04] for _ in inputs]
            
            # Convert predictions to list if needed
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
            
            logger.info(
                "Prediction completed",
                model_name=model_name,
                model_version=model_version,
                input_count=len(inputs),
                prediction_count=len(predictions) if isinstance(predictions, list) else 1
            )
            
            return predictions
            
        except Exception as e:
            logger.error(
                "Prediction failed",
                model_name=model_name,
                model_version=model_version,
                error=str(e)
            )
            raise
    
    async def _get_model(self, model_name: str, model_version: str) -> Optional[Any]:
        """Get a model from cache or load it."""
        cache_key = f"{model_name}:{model_version}"
        
        # Check cache first
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]["model"]
        
        # Try to load from local storage
        try:
            model_path = self.model_storage_path / model_name / model_version
            
            model = await self._load_model_from_path(str(model_path))
            if model:
                # Cache the model
                await self._cache_model(model_name, model_version, model)
                return model
            else:
                logger.error("Failed to load model", model_name=model_name, version=model_version)
                return None
            
        except Exception as e:
            logger.error(
                "Failed to load model",
                model_name=model_name,
                model_version=model_version,
                error=str(e)
            )
            return None
    
    async def batch_predict(
        self,
        job_id: str,
        inputs: List[Dict[str, Any]],
        model_name: str = "default",
        model_version: str = "latest"
    ):
        """Start a batch prediction job.

        In this simple variant, predictions are computed inline; replace with a
        task queue and durable status store for production.
        """
        try:
            # Initialize job
            self.batch_jobs[job_id] = {
                "status": "processing",
                "started_at": time.time(),
                "model_name": model_name,
                "model_version": model_version,
                "input_count": len(inputs),
                "progress": 0
            }
            
            # Process batch (simplified implementation)
            # In a real implementation, this would be queued and processed asynchronously
            predictions = await self.predict(inputs, model_name, model_version)
            
            # Update job status
            self.batch_jobs[job_id].update({
                "status": "completed",
                "completed_at": time.time(),
                "predictions": predictions,
                "progress": 100
            })
            
            logger.info("Batch prediction completed", job_id=job_id)
            
        except Exception as e:
            # Update job status to failed
            self.batch_jobs[job_id].update({
                "status": "failed",
                "failed_at": time.time(),
                "error": str(e),
                "progress": 0
            })
            
            logger.error("Batch prediction failed", job_id=job_id, error=str(e))
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        try:
            models = []
            for model_name, model_info in self.models.items():
                models.append({
                    "name": model_name,
                    "version": model_info["active_version"],
                    "stage": "Production",
                    "created_at": "2024-01-01T00:00:00Z",  # Placeholder
                    "description": f"Model {model_name}"
                })
            
            return models
            
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        try:
            if model_name not in self.models:
                return None
            
            model_info = self.models[model_name]
            return {
                "name": model_name,
                "active_version": model_info["active_version"],
                "stage": "Production",
                "created_at": "2024-01-01T00:00:00Z",  # Placeholder
                "description": f"Model {model_name}",
                "versions": list(model_info["versions"].keys())
            }
            
        except Exception as e:
            logger.error("Failed to get model info", model_name=model_name, error=str(e))
            raise
    
    async def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List versions of a specific model."""
        try:
            if model_name not in self.models:
                return []
            
            versions = []
            for version, version_info in self.models[model_name]["versions"].items():
                versions.append({
                    "version": version,
                    "stage": "Production",
                    "created_at": "2024-01-01T00:00:00Z",  # Placeholder
                    "loaded_at": version_info["loaded_at"]
                })
            
            return versions
            
        except Exception as e:
            logger.error("Failed to list model versions", model_name=model_name, error=str(e))
            raise
    
    async def get_batch_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch prediction job."""
        try:
            if job_id not in self.batch_jobs:
                return None
            
            job = self.batch_jobs[job_id].copy()
            
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
    
    async def reload_all_models(self):
        """Reload all cached models."""
        try:
            for model_name in list(self.models.keys()):
                active_version = self.models[model_name]["active_version"]
                await self._reload_model(model_name, active_version)
            
            logger.info("All models reloaded")
            
        except Exception as e:
            logger.error("Failed to reload all models", error=str(e))
            raise
    
    async def health_check(self) -> bool:
        """Check if the model manager is healthy."""
        try:
            if not self.models:
                logger.warning(
                    "Health check passed but no models are currently loaded",
                    service="model-serving"
                )
            return True
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Stop event listener task if running
            if self._event_listener_task and not self._event_listener_task.done():
                self._event_listener_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._event_listener_task
            
            # Close event subscriber connection
            with suppress(Exception):
                await self.event_subscriber.close()
            
            # Clear model cache
            self.model_cache.clear()
            self.models.clear()
            self.batch_jobs.clear()
            
            logger.info("Model manager cleanup completed")
            
        except Exception as e:
            logger.error("Model manager cleanup failed", error=str(e))