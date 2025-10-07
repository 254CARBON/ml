"""API routes for model serving service."""

import os
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
import structlog

from ..runtime.model_manager import ModelManager
from ..runtime.ensemble_manager import EnsembleManager, ExperimentConfig, ModelVariant, TrafficSplitStrategy, ShadowDeploymentManager
from ..runtime.metrics import get_metrics_collector
from libs.common.events import InferenceUsageEvent

logger = structlog.get_logger("model_serving.api")

router = APIRouter()


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    inputs: List[Dict[str, Any]] = Field(..., description="Input data for prediction")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    model_version: Optional[str] = Field(None, description="Specific model version")
    experiment_id: Optional[str] = Field(None, description="Experiment identifier for traffic routing")
    user_id: Optional[str] = Field(None, description="User identifier for sticky routing strategies")
    session_id: Optional[str] = Field(None, description="Session identifier for sticky routing strategies")


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    predictions: List[Any] = Field(..., description="Model predictions")
    model_name: str = Field(..., description="Model name used")
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    experiment_id: Optional[str] = Field(None, description="Experiment identifier used for routing")
    routing_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata about routing decisions")


class BatchRequest(BaseModel):
    """Request model for batch prediction endpoint."""
    inputs: List[Dict[str, Any]] = Field(..., description="Input data for batch prediction")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    model_version: Optional[str] = Field(None, description="Specific model version")


class BatchResponse(BaseModel):
    """Response model for batch prediction endpoint."""
    job_id: str = Field(..., description="Batch job ID")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


class ModelInfo(BaseModel):
    """Model information model."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Model stage")
    created_at: str = Field(..., description="Model creation timestamp")
    description: Optional[str] = Field(None, description="Model description")


class ExperimentRequest(BaseModel):
    """Request model for creating A/B testing experiments."""
    name: str = Field(..., description="Experiment name")
    model_a: str = Field(..., description="Model A name")
    model_b: str = Field(..., description="Model B name")
    traffic_split: float = Field(0.5, description="Traffic split ratio (0.0-1.0)")
    description: Optional[str] = Field(None, description="Experiment description")


class ExperimentResponse(BaseModel):
    """Response model for experiment operations."""
    experiment_id: str = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    status: str = Field(..., description="Experiment status")
    traffic_split: float = Field(..., description="Traffic split ratio")
    created_at: str = Field(..., description="Creation timestamp")


class ShadowRequest(BaseModel):
    """Request model for shadow deployment."""
    model_name: str = Field(..., description="Model name")
    shadow_model: str = Field(..., description="Shadow model name")
    enabled: bool = Field(True, description="Enable shadow deployment")


class ShadowResponse(BaseModel):
    """Response model for shadow deployment."""
    model_name: str = Field(..., description="Model name")
    shadow_model: str = Field(..., description="Shadow model name")
    enabled: bool = Field(..., description="Shadow deployment status")
    comparison_metrics: Dict[str, Any] = Field(..., description="Comparison metrics")


def get_model_manager(request: Request) -> ModelManager:
    """Get model manager from application state."""
    return request.app.state.model_manager


def get_ensemble_manager(request: Request) -> EnsembleManager:
    """Get ensemble manager from application state."""
    return request.app.state.ensemble_manager


def get_shadow_deployment_manager(request: Request) -> ShadowDeploymentManager:
    """Get shadow deployment manager from application state."""
    return request.app.state.shadow_deployment_manager


def get_metrics(request: Request):
    """Get metrics collector from application state."""
    return request.app.state.metrics_collector


def get_event_publisher(request: Request):
    """Get event publisher from application state."""
    return request.app.state.event_publisher


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    ensemble_manager: EnsembleManager = Depends(get_ensemble_manager),
    metrics_collector = Depends(get_metrics),
    event_publisher = Depends(get_event_publisher)
):
    """Synchronous prediction endpoint."""
    
    try:
        # Get model name and version
        model_name = request.model_name or "default"
        model_version = request.model_version or "latest"
        
        # Check if A/B testing is enabled
        ab_testing_enabled = os.getenv("ML_AB_TESTING_ENABLED", "false").lower() == "true"
        
        # Use ensemble manager to perform prediction (with optional experiments)
        inference_result = await ensemble_manager.predict(
            inputs=request.inputs,
            model_name=model_name,
            model_version=model_version,
            experiment_id=request.experiment_id,
            request_context=request.dict(exclude_none=True),
            user_id=request.user_id,
            session_id=request.session_id,
            use_experiments=ab_testing_enabled
        )
        
        predictions = inference_result["predictions"]
        resolved_model_name = inference_result["model_name"]
        resolved_model_version = inference_result["model_version"]
        latency_ms = inference_result["latency_ms"]
        experiment_id = inference_result.get("experiment_id")
        routing_metadata = inference_result.get("routing_metadata")
        
        # Record metrics using resolved model identifiers
        metrics_collector.record_inference(
            model_name=resolved_model_name,
            model_version=resolved_model_version,
            duration=latency_ms / 1000.0
        )
        
        # Publish usage event
        event_publisher.publish_inference_usage(
            model_name=resolved_model_name,
            version=resolved_model_version,
            latency_ms=int(latency_ms),
            request_size=len(request.inputs)
        )
        
        logger.info(
            "Prediction completed",
            requested_model=model_name,
            resolved_model=resolved_model_name,
            resolved_version=resolved_model_version,
            latency_ms=latency_ms,
            input_count=len(request.inputs),
            experiment_id=experiment_id
        )
        
        return PredictResponse(
            predictions=predictions,
            model_name=resolved_model_name,
            model_version=resolved_model_version,
            latency_ms=latency_ms,
            experiment_id=experiment_id,
            routing_metadata=routing_metadata
        )
        
    except RuntimeError as config_error:
        logger.error(
            "Prediction failed due to ensemble manager configuration",
            error=str(config_error)
        )
        raise HTTPException(status_code=503, detail=str(config_error)) from config_error
    except Exception as e:
        logger.error(
            "Prediction failed",
            model_name=request.model_name,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch", response_model=BatchResponse)
async def batch_predict(
    request: BatchRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Asynchronous batch prediction endpoint."""
    try:
        # Generate job ID
        job_id = f"batch_{int(time.time())}_{hash(str(request.inputs)) % 10000}"
        
        # Start batch job (simplified implementation)
        # In a real implementation, this would queue the job
        await model_manager.batch_predict(
            job_id=job_id,
            inputs=request.inputs,
            model_name=request.model_name or "default",
            model_version=request.model_version or "latest"
        )
        
        logger.info(
            "Batch prediction job started",
            job_id=job_id,
            input_count=len(request.inputs)
        )
        
        return BatchResponse(
            job_id=job_id,
            status="queued",
            message="Batch prediction job queued successfully"
        )
        
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """List available models."""
    try:
        models = await model_manager.list_models()
        
        model_info = []
        for model in models:
            model_info.append(ModelInfo(
                name=model["name"],
                version=model["version"],
                stage=model["stage"],
                created_at=model["created_at"],
                description=model.get("description")
            ))
        
        logger.info("Models listed", count=len(model_info))
        return model_info
        
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_name}")
async def get_model_info(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get information about a specific model."""
    try:
        model_info = await model_manager.get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        logger.info("Model info retrieved", model_name=model_name)
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model info", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.get("/models/{model_name}/versions")
async def list_model_versions(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """List versions of a specific model."""
    try:
        versions = await model_manager.list_model_versions(model_name)
        
        logger.info("Model versions listed", model_name=model_name, count=len(versions))
        return versions
        
    except Exception as e:
        logger.error("Failed to list model versions", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list model versions: {str(e)}")


# A/B Testing Experiment Endpoints
@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentRequest,
    ensemble_manager: EnsembleManager = Depends(get_ensemble_manager)
):
    """Create a new A/B testing experiment."""
    # Check if A/B testing is enabled
    ab_testing_enabled = os.getenv("ML_AB_TESTING_ENABLED", "false").lower() == "true"
    if not ab_testing_enabled:
        raise HTTPException(status_code=501, detail="A/B testing is not enabled")
    
    try:
        # Create experiment configuration
        experiment_id = f"exp_{int(time.time())}_{hash(request.name) % 10000}"
        
        variants = [
            ModelVariant(
                model_name=request.model_a,
                model_version="latest",
                traffic_percentage=request.traffic_split * 100,
                is_champion=True
            ),
            ModelVariant(
                model_name=request.model_b,
                model_version="latest", 
                traffic_percentage=(1 - request.traffic_split) * 100,
                is_challenger=True
            )
        ]
        
        experiment_config = ExperimentConfig(
            experiment_id=experiment_id,
            name=request.name,
            description=request.description or "",
            variants=variants,
            traffic_split_strategy=TrafficSplitStrategy.RANDOM,
            start_time=time.time()
        )
        
        success = await ensemble_manager.create_experiment(experiment_config)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create experiment")
        
        logger.info("Experiment created", experiment_id=experiment_id, name=request.name)
        
        return ExperimentResponse(
            experiment_id=experiment_id,
            name=request.name,
            status="active",
            traffic_split=request.traffic_split,
            created_at=str(int(time.time()))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create experiment", name=request.name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    ensemble_manager: EnsembleManager = Depends(get_ensemble_manager)
):
    """List all A/B testing experiments."""
    # Check if A/B testing is enabled
    ab_testing_enabled = os.getenv("ML_AB_TESTING_ENABLED", "false").lower() == "true"
    if not ab_testing_enabled:
        raise HTTPException(status_code=501, detail="A/B testing is not enabled")
    
    try:
        experiments = []
        for exp_id, exp_config in ensemble_manager.active_experiments.items():
            experiments.append(ExperimentResponse(
                experiment_id=exp_id,
                name=exp_config.name,
                status="running" if exp_config.end_time is None else "completed",
                traffic_split=exp_config.variants[0].traffic_percentage / 100.0 if exp_config.variants else 0.5,
                created_at=str(int(exp_config.start_time))
            ))
        
        logger.info("Experiments listed", count=len(experiments))
        return experiments
        
    except Exception as e:
        logger.error("Failed to list experiments", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    ensemble_manager: EnsembleManager = Depends(get_ensemble_manager)
):
    """Stop an A/B testing experiment."""
    # Check if A/B testing is enabled
    ab_testing_enabled = os.getenv("ML_AB_TESTING_ENABLED", "false").lower() == "true"
    if not ab_testing_enabled:
        raise HTTPException(status_code=501, detail="A/B testing is not enabled")
    
    try:
        success = await ensemble_manager.stop_experiment(experiment_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
        logger.info("Experiment stopped", experiment_id=experiment_id)
        return {"status": "success", "message": f"Experiment {experiment_id} stopped"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to stop experiment", experiment_id=experiment_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to stop experiment: {str(e)}")


# Shadow Deployment Endpoints
@router.post("/shadow", response_model=ShadowResponse)
async def configure_shadow_deployment(
    request: ShadowRequest,
    shadow_manager: ShadowDeploymentManager = Depends(get_shadow_deployment_manager)
):
    """Configure shadow deployment for a model."""
    try:
        success = await shadow_manager.enable_shadow_deployment(
            production_model=request.model_name,
            shadow_model=request.shadow_model,
            shadow_model_version="latest",
            traffic_percentage=100.0 if request.enabled else 0.0
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to configure shadow deployment")
        
        # Get comparison metrics
        shadow_key = f"{request.shadow_model}:latest"
        metrics = await shadow_manager.get_shadow_deployment_report(shadow_key)
        
        logger.info("Shadow deployment configured", 
                   model_name=request.model_name, 
                   shadow_model=request.shadow_model,
                   enabled=request.enabled)
        
        return ShadowResponse(
            model_name=request.model_name,
            shadow_model=request.shadow_model,
            enabled=request.enabled,
            comparison_metrics=metrics or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to configure shadow deployment", 
                    model_name=request.model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to configure shadow deployment: {str(e)}")


@router.get("/shadow/{model_name}", response_model=ShadowResponse)
async def get_shadow_deployment(
    model_name: str,
    shadow_manager: ShadowDeploymentManager = Depends(get_shadow_deployment_manager)
):
    """Get shadow deployment status and metrics."""
    try:
        # Find shadow deployment for this model
        shadow_key = None
        for key, config in shadow_manager.shadow_models.items():
            if config["production_model"] == model_name:
                shadow_key = key
                break
        
        if not shadow_key:
            raise HTTPException(status_code=404, detail=f"No shadow deployment found for model {model_name}")
        
        # Get comparison metrics
        metrics = await shadow_manager.get_shadow_deployment_report(shadow_key)
        config = shadow_manager.shadow_models[shadow_key]
        
        logger.info("Shadow deployment status retrieved", model_name=model_name)
        
        return ShadowResponse(
            model_name=model_name,
            shadow_model=config["shadow_model"],
            enabled=config["traffic_percentage"] > 0,
            comparison_metrics=metrics or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get shadow deployment status", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get shadow deployment status: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_batch_job_status(
    job_id: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get status of a batch prediction job."""
    try:
        job_status = await model_manager.get_batch_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        logger.info("Batch job status retrieved", job_id=job_id)
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get batch job status", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")
