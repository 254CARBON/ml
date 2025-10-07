"""Streaming prediction processor for real-time ML inference."""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
import redis.asyncio as redis
from dataclasses import dataclass, asdict
import numpy as np
import structlog
from contextlib import asynccontextmanager

from libs.common.config import ModelServingConfig
from libs.common.events import create_event_publisher
from libs.common.tracing import get_ml_tracer

logger = structlog.get_logger("stream_processor")


@dataclass
class StreamPredictionRequest:
    """Streaming prediction request."""
    request_id: str
    timestamp: float
    model_name: str
    model_version: str
    inputs: List[Dict[str, Any]]
    callback_url: Optional[str] = None
    priority: int = 1  # 1 = high, 2 = medium, 3 = low


@dataclass
class StreamPredictionResponse:
    """Streaming prediction response."""
    request_id: str
    timestamp: float
    model_name: str
    model_version: str
    predictions: List[Any]
    latency_ms: float
    success: bool
    error: Optional[str] = None


class StreamingPredictor:
    """Handles streaming predictions with real-time processing."""
    
    def __init__(self, config: ModelServingConfig):
        """Initialize streaming predictor internals.

        Parameters
        - config: ``ModelServingConfig`` for Redis URL and tracing
        """
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.model_manager = None  # Will be injected
        self.active_streams: Dict[str, asyncio.Task] = {}
        self.prediction_queue = asyncio.Queue(maxsize=10000)
        self.response_callbacks: Dict[str, Callable] = {}
        
        # Stream configuration
        self.batch_size = 32
        self.batch_timeout_ms = 100  # 100ms batching window
        self.max_concurrent_predictions = 100
        
        # Initialize tracing and events
        self.tracer = get_ml_tracer("streaming-predictor")
        self.event_publisher = create_event_publisher(config.ml_redis_url)
    
    async def initialize(self, model_manager):
        """Initialize streaming predictor."""
        
        self.redis_client = redis.from_url(self.config.ml_redis_url)
        self.model_manager = model_manager
        
        # Start processing tasks
        self.active_streams["batch_processor"] = asyncio.create_task(self._batch_processor())
        self.active_streams["response_handler"] = asyncio.create_task(self._response_handler())
        
        logger.info("Streaming predictor initialized")
    
    async def submit_prediction(
        self,
        request: StreamPredictionRequest,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit a prediction request for streaming processing."""
        
        try:
            # Store callback if provided
            if callback:
                self.response_callbacks[request.request_id] = callback
            
            # Add to processing queue
            await self.prediction_queue.put(request)
            
            # Store request in Redis for tracking
            await self.redis_client.setex(
                f"stream_request:{request.request_id}",
                300,  # 5 minute TTL
                json.dumps(asdict(request), default=str)
            )
            
            logger.info("Streaming prediction submitted", 
                       request_id=request.request_id,
                       model_name=request.model_name)
            
            return request.request_id
            
        except Exception as e:
            logger.error("Failed to submit streaming prediction", 
                        request_id=request.request_id, 
                        error=str(e))
            raise
    
    async def _batch_processor(self):
        """Process predictions in batches for efficiency."""
        
        logger.info("Starting batch processor")
        
        while True:
            try:
                batch_requests = []
                batch_start_time = time.time()
                
                # Collect requests for batching
                while (len(batch_requests) < self.batch_size and 
                       (time.time() - batch_start_time) * 1000 < self.batch_timeout_ms):
                    
                    try:
                        request = await asyncio.wait_for(
                            self.prediction_queue.get(),
                            timeout=0.01  # 10ms timeout
                        )
                        batch_requests.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if batch_requests:
                    # Process batch
                    await self._process_batch(batch_requests)
                else:
                    # No requests, brief sleep
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                logger.error("Batch processor error", error=str(e))
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, requests: List[StreamPredictionRequest]):
        """Process a batch of prediction requests."""
        
        batch_start_time = time.perf_counter()
        
        # Group requests by model
        model_groups = {}
        for request in requests:
            model_key = f"{request.model_name}:{request.model_version}"
            if model_key not in model_groups:
                model_groups[model_key] = []
            model_groups[model_key].append(request)
        
        # Process each model group
        for model_key, model_requests in model_groups.items():
            model_name, model_version = model_key.split(":", 1)
            
            try:
                # Prepare batch inputs
                batch_inputs = []
                for request in model_requests:
                    batch_inputs.extend(request.inputs)
                
                # Make batch prediction
                with self.tracer.trace_model_inference(
                    model_name=model_name,
                    model_version=model_version,
                    input_count=len(batch_inputs),
                    operation="streaming_batch"
                ):
                    predictions = await self.model_manager.predict(
                        inputs=batch_inputs,
                        model_name=model_name,
                        model_version=model_version
                    )
                
                # Split predictions back to individual requests
                prediction_idx = 0
                for request in model_requests:
                    request_predictions = predictions[prediction_idx:prediction_idx + len(request.inputs)]
                    prediction_idx += len(request.inputs)
                    
                    # Create response
                    response = StreamPredictionResponse(
                        request_id=request.request_id,
                        timestamp=time.time(),
                        model_name=model_name,
                        model_version=model_version,
                        predictions=request_predictions,
                        latency_ms=(time.perf_counter() - batch_start_time) * 1000,
                        success=True
                    )
                    
                    # Send response
                    await self._send_response(response)
                    
            except Exception as e:
                # Handle batch failure
                for request in model_requests:
                    error_response = StreamPredictionResponse(
                        request_id=request.request_id,
                        timestamp=time.time(),
                        model_name=model_name,
                        model_version=model_version,
                        predictions=[],
                        latency_ms=(time.perf_counter() - batch_start_time) * 1000,
                        success=False,
                        error=str(e)
                    )
                    
                    await self._send_response(error_response)
                
                logger.error("Batch prediction failed", 
                           model_key=model_key,
                           batch_size=len(model_requests),
                           error=str(e))
    
    async def _send_response(self, response: StreamPredictionResponse):
        """Send prediction response."""
        
        try:
            # Store response in Redis
            await self.redis_client.setex(
                f"stream_response:{response.request_id}",
                300,  # 5 minute TTL
                json.dumps(asdict(response), default=str)
            )
            
            # Publish response event
            await self.redis_client.publish(
                f"stream_responses:{response.request_id}",
                json.dumps(asdict(response), default=str)
            )
            
            # Call callback if registered
            if response.request_id in self.response_callbacks:
                callback = self.response_callbacks.pop(response.request_id)
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(response)
                    else:
                        callback(response)
                except Exception as e:
                    logger.error("Callback failed", 
                               request_id=response.request_id, 
                               error=str(e))
            
            logger.debug("Response sent", 
                        request_id=response.request_id,
                        success=response.success)
            
        except Exception as e:
            logger.error("Failed to send response", 
                        request_id=response.request_id, 
                        error=str(e))
    
    async def _response_handler(self):
        """Handle response delivery and cleanup."""
        
        logger.info("Starting response handler")
        
        while True:
            try:
                # Clean up old requests and responses
                current_time = time.time()
                
                # Get all stream request keys
                request_keys = await self.redis_client.keys("stream_request:*")
                
                for key in request_keys:
                    request_data = await self.redis_client.get(key)
                    if request_data:
                        request_info = json.loads(request_data)
                        request_age = current_time - request_info.get("timestamp", current_time)
                        
                        # Clean up old requests (older than 10 minutes)
                        if request_age > 600:
                            await self.redis_client.delete(key)
                            # Also clean up corresponding response
                            response_key = key.replace("stream_request:", "stream_response:")
                            await self.redis_client.delete(response_key)
                
                await asyncio.sleep(30)  # Clean up every 30 seconds
                
            except Exception as e:
                logger.error("Response handler error", error=str(e))
                await asyncio.sleep(10)
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a streaming prediction request."""
        
        try:
            # Check for response first
            response_data = await self.redis_client.get(f"stream_response:{request_id}")
            if response_data:
                response = json.loads(response_data)
                return {
                    "status": "completed" if response["success"] else "failed",
                    "response": response
                }
            
            # Check if request exists
            request_data = await self.redis_client.get(f"stream_request:{request_id}")
            if request_data:
                return {
                    "status": "processing",
                    "request": json.loads(request_data)
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to get request status", 
                        request_id=request_id, 
                        error=str(e))
            return None
    
    async def stream_predictions(
        self,
        model_name: str,
        input_stream: AsyncGenerator[Dict[str, Any], None],
        model_version: str = "latest"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream predictions for continuous input."""
        
        logger.info("Starting prediction streaming", 
                   model_name=model_name,
                   model_version=model_version)
        
        request_counter = 0
        
        async for input_data in input_stream:
            try:
                request_counter += 1
                request_id = f"stream_{int(time.time() * 1000)}_{request_counter}"
                
                # Create streaming request
                stream_request = StreamPredictionRequest(
                    request_id=request_id,
                    timestamp=time.time(),
                    model_name=model_name,
                    model_version=model_version,
                    inputs=[input_data]
                )
                
                # Submit for processing
                await self.submit_prediction(stream_request)
                
                # Wait for response (with timeout)
                response = await self._wait_for_response(request_id, timeout=5.0)
                
                if response:
                    yield {
                        "request_id": request_id,
                        "predictions": response.predictions,
                        "latency_ms": response.latency_ms,
                        "success": response.success,
                        "error": response.error
                    }
                else:
                    yield {
                        "request_id": request_id,
                        "predictions": [],
                        "latency_ms": 5000,
                        "success": False,
                        "error": "Timeout waiting for prediction"
                    }
                    
            except Exception as e:
                logger.error("Streaming prediction error", error=str(e))
                yield {
                    "request_id": f"error_{request_counter}",
                    "predictions": [],
                    "latency_ms": 0,
                    "success": False,
                    "error": str(e)
                }
    
    async def _wait_for_response(
        self,
        request_id: str,
        timeout: float = 5.0
    ) -> Optional[StreamPredictionResponse]:
        """Wait for streaming prediction response."""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response_data = await self.redis_client.get(f"stream_response:{request_id}")
            if response_data:
                response_dict = json.loads(response_data)
                return StreamPredictionResponse(**response_dict)
            
            await asyncio.sleep(0.01)  # 10ms polling interval
        
        return None
    
    async def cleanup(self):
        """Cleanup streaming predictor resources."""
        
        logger.info("Cleaning up streaming predictor")
        
        # Cancel active tasks
        for task_name, task in self.active_streams.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Cancelled {task_name} task")
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Streaming predictor cleanup completed")


class OnlineLearningManager:
    """Manages online learning and model updates."""
    
    def __init__(self, config: ModelServingConfig):
        """Configure buffers and thresholds for online learning.

        Parameters
        - config: ``ModelServingConfig`` providing Redis URL and env details
        """
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.learning_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.model_updates: Dict[str, Dict[str, Any]] = {}
        self.update_thresholds = {
            "min_samples": 1000,
            "max_age_hours": 24,
            "performance_degradation_threshold": 0.1
        }
    
    async def initialize(self):
        """Initialize online learning manager."""
        
        self.redis_client = redis.from_url(self.config.ml_redis_url)
        
        # Start background tasks
        asyncio.create_task(self._learning_buffer_processor())
        asyncio.create_task(self._model_update_scheduler())
        
        logger.info("Online learning manager initialized")
    
    async def record_prediction_feedback(
        self,
        model_name: str,
        model_version: str,
        input_data: Dict[str, Any],
        prediction: Any,
        actual_outcome: Optional[Any] = None,
        feedback_score: Optional[float] = None
    ):
        """Record prediction feedback for online learning."""
        
        try:
            feedback_record = {
                "timestamp": time.time(),
                "model_name": model_name,
                "model_version": model_version,
                "input_data": input_data,
                "prediction": prediction,
                "actual_outcome": actual_outcome,
                "feedback_score": feedback_score,
                "prediction_error": self._calculate_prediction_error(prediction, actual_outcome)
            }
            
            # Add to learning buffer
            model_key = f"{model_name}:{model_version}"
            if model_key not in self.learning_buffer:
                self.learning_buffer[model_key] = []
            
            self.learning_buffer[model_key].append(feedback_record)
            
            # Store in Redis for persistence
            await self.redis_client.lpush(
                f"learning_buffer:{model_key}",
                json.dumps(feedback_record, default=str)
            )
            
            # Limit buffer size
            await self.redis_client.ltrim(f"learning_buffer:{model_key}", 0, 10000)
            
            logger.debug("Prediction feedback recorded", 
                        model_name=model_name,
                        feedback_score=feedback_score)
            
        except Exception as e:
            logger.error("Failed to record prediction feedback", error=str(e))
    
    def _calculate_prediction_error(self, prediction: Any, actual: Any) -> Optional[float]:
        """Calculate prediction error if actual outcome is available."""
        
        if actual is None:
            return None
        
        try:
            if isinstance(prediction, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
                pred_array = np.array(prediction)
                actual_array = np.array(actual)
                
                if pred_array.shape == actual_array.shape:
                    # Mean squared error
                    return float(np.mean((pred_array - actual_array) ** 2))
            
            elif isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
                return float((prediction - actual) ** 2)
            
            return None
            
        except Exception:
            return None
    
    async def _learning_buffer_processor(self):
        """Process learning buffer and trigger model updates."""
        
        logger.info("Starting learning buffer processor")
        
        while True:
            try:
                for model_key, buffer in self.learning_buffer.items():
                    if len(buffer) >= self.update_thresholds["min_samples"]:
                        # Check if we should trigger an update
                        should_update = await self._should_trigger_update(model_key, buffer)
                        
                        if should_update:
                            await self._trigger_model_update(model_key, buffer)
                            # Clear processed buffer
                            self.learning_buffer[model_key] = []
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error("Learning buffer processor error", error=str(e))
                await asyncio.sleep(60)
    
    async def _should_trigger_update(self, model_key: str, buffer: List[Dict[str, Any]]) -> bool:
        """Determine if model update should be triggered."""
        
        try:
            # Check buffer age
            oldest_record = min(buffer, key=lambda x: x["timestamp"])
            age_hours = (time.time() - oldest_record["timestamp"]) / 3600
            
            if age_hours > self.update_thresholds["max_age_hours"]:
                logger.info("Triggering update due to buffer age", 
                           model_key=model_key, 
                           age_hours=age_hours)
                return True
            
            # Check performance degradation
            recent_errors = [r["prediction_error"] for r in buffer[-100:] 
                           if r["prediction_error"] is not None]
            
            if len(recent_errors) > 10:
                avg_error = np.mean(recent_errors)
                # Compare with historical performance (simplified)
                historical_error = 0.01  # Placeholder
                
                if avg_error > historical_error * (1 + self.update_thresholds["performance_degradation_threshold"]):
                    logger.info("Triggering update due to performance degradation",
                               model_key=model_key,
                               current_error=avg_error,
                               historical_error=historical_error)
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Error checking update trigger", error=str(e))
            return False
    
    async def _trigger_model_update(self, model_key: str, buffer: List[Dict[str, Any]]):
        """Trigger model update based on feedback."""
        
        try:
            model_name, model_version = model_key.split(":", 1)
            
            # Prepare update data
            update_data = {
                "model_name": model_name,
                "current_version": model_version,
                "feedback_samples": len(buffer),
                "update_triggered_at": time.time(),
                "trigger_reason": "online_learning",
                "buffer_summary": self._summarize_buffer(buffer)
            }
            
            # Store update request
            self.model_updates[model_key] = update_data
            
            # Publish update event (would trigger retraining pipeline)
            await self.redis_client.publish(
                "model_update_requests",
                json.dumps(update_data, default=str)
            )
            
            logger.info("Model update triggered", 
                       model_key=model_key,
                       samples=len(buffer))
            
        except Exception as e:
            logger.error("Failed to trigger model update", error=str(e))
    
    def _summarize_buffer(self, buffer: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize feedback buffer for analysis."""
        
        errors = [r["prediction_error"] for r in buffer if r["prediction_error"] is not None]
        feedback_scores = [r["feedback_score"] for r in buffer if r["feedback_score"] is not None]
        
        return {
            "total_samples": len(buffer),
            "samples_with_errors": len(errors),
            "samples_with_feedback": len(feedback_scores),
            "avg_error": np.mean(errors) if errors else None,
            "avg_feedback_score": np.mean(feedback_scores) if feedback_scores else None,
            "time_span_hours": (buffer[-1]["timestamp"] - buffer[0]["timestamp"]) / 3600 if buffer else 0
        }
    
    async def _model_update_scheduler(self):
        """Schedule and coordinate model updates."""
        
        logger.info("Starting model update scheduler")
        
        while True:
            try:
                # Check for pending updates
                for model_key, update_data in list(self.model_updates.items()):
                    update_age = time.time() - update_data["update_triggered_at"]
                    
                    # Check if update is complete (simplified check)
                    if update_age > 3600:  # 1 hour timeout
                        logger.warning("Model update timeout", 
                                     model_key=model_key,
                                     age_hours=update_age / 3600)
                        del self.model_updates[model_key]
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error("Model update scheduler error", error=str(e))
                await asyncio.sleep(300)


class RealTimeFeatureStore:
    """Real-time feature store for streaming predictions."""
    
    def __init__(self, config: ModelServingConfig):
        """Create a real-time feature store client.

        Parameters
        - config: ``ModelServingConfig`` for Redis connectivity
        """
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
        self.feature_ttl = 300  # 5 minutes default TTL
    
    async def initialize(self):
        """Initialize real-time feature store."""
        
        self.redis_client = redis.from_url(self.config.ml_redis_url)
        
        # Start feature refresh task
        asyncio.create_task(self._feature_refresh_loop())
        
        logger.info("Real-time feature store initialized")
    
    async def get_features(
        self,
        entity_id: str,
        feature_names: List[str],
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get real-time features for an entity."""
        
        try:
            features = {}
            
            for feature_name in feature_names:
                cache_key = f"feature:{entity_id}:{feature_name}"
                
                # Check cache first
                if cache_key in self.feature_cache:
                    cached_feature = self.feature_cache[cache_key]
                    if time.time() - cached_feature["timestamp"] < self.feature_ttl:
                        features[feature_name] = cached_feature["value"]
                        continue
                
                # Check Redis
                feature_data = await self.redis_client.get(cache_key)
                if feature_data:
                    feature_info = json.loads(feature_data)
                    if time.time() - feature_info["timestamp"] < self.feature_ttl:
                        features[feature_name] = feature_info["value"]
                        # Cache locally
                        self.feature_cache[cache_key] = feature_info
                        continue
                
                # Feature not available or expired
                logger.warning("Feature not available", 
                             entity_id=entity_id, 
                             feature_name=feature_name)
                features[feature_name] = None
            
            return features
            
        except Exception as e:
            logger.error("Failed to get features", 
                        entity_id=entity_id, 
                        error=str(e))
            return {}
    
    async def update_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        timestamp: Optional[float] = None
    ):
        """Update real-time features for an entity."""
        
        if timestamp is None:
            timestamp = time.time()
        
        try:
            for feature_name, feature_value in features.items():
                cache_key = f"feature:{entity_id}:{feature_name}"
                
                feature_data = {
                    "value": feature_value,
                    "timestamp": timestamp,
                    "entity_id": entity_id,
                    "feature_name": feature_name
                }
                
                # Store in Redis
                await self.redis_client.setex(
                    cache_key,
                    self.feature_ttl,
                    json.dumps(feature_data, default=str)
                )
                
                # Update local cache
                self.feature_cache[cache_key] = feature_data
            
            logger.debug("Features updated", 
                        entity_id=entity_id,
                        feature_count=len(features))
            
        except Exception as e:
            logger.error("Failed to update features", 
                        entity_id=entity_id, 
                        error=str(e))
    
    async def _feature_refresh_loop(self):
        """Background task to refresh features from external sources."""
        
        logger.info("Starting feature refresh loop")
        
        while True:
            try:
                # This would typically:
                # 1. Query external data sources for latest features
                # 2. Update feature cache
                # 3. Notify dependent services
                
                # For now, just clean up expired features
                current_time = time.time()
                expired_keys = []
                
                for cache_key, feature_data in self.feature_cache.items():
                    if current_time - feature_data["timestamp"] > self.feature_ttl:
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del self.feature_cache[key]
                
                if expired_keys:
                    logger.debug("Cleaned up expired features", count=len(expired_keys))
                
                await asyncio.sleep(60)  # Refresh every minute
                
            except Exception as e:
                logger.error("Feature refresh loop error", error=str(e))
                await asyncio.sleep(30)


def create_streaming_predictor(config: ModelServingConfig) -> StreamingPredictor:
    """Create streaming predictor."""
    return StreamingPredictor(config)


def create_online_learning_manager(config: ModelServingConfig) -> OnlineLearningManager:
    """Create online learning manager."""
    return OnlineLearningManager(config)


def create_realtime_feature_store(config: ModelServingConfig) -> RealTimeFeatureStore:
    """Create real-time feature store."""
    return RealTimeFeatureStore(config)
