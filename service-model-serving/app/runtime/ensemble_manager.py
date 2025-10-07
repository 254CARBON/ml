"""Ensemble model serving and A/B testing framework."""

import time
import random
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
import structlog
from dataclasses import dataclass
from enum import Enum

from libs.common.config import ModelServingConfig
from libs.common.events import create_event_publisher
from libs.common.tracing import get_ml_tracer

logger = structlog.get_logger("ensemble_manager")

if TYPE_CHECKING:
    from .model_manager import ModelManager


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies for A/B testing."""
    RANDOM = "random"
    WEIGHTED = "weighted"
    STICKY_SESSION = "sticky_session"
    FEATURE_BASED = "feature_based"


@dataclass
class ModelVariant:
    """Represents a model variant in A/B testing."""
    model_name: str
    model_version: str
    traffic_percentage: float
    is_champion: bool = False
    is_challenger: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiments."""
    experiment_id: str
    name: str
    description: str
    variants: List[ModelVariant]
    traffic_split_strategy: TrafficSplitStrategy
    start_time: float
    end_time: Optional[float] = None
    success_metrics: List[str] = None
    minimum_sample_size: int = 1000
    statistical_significance_threshold: float = 0.05


class EnsembleManager:
    """Manages ensemble serving and A/B testing."""
    
    def __init__(
        self,
        config: ModelServingConfig,
        model_manager: Optional["ModelManager"] = None,
        shadow_manager: Optional["ShadowDeploymentManager"] = None
    ):
        """Create an ensemble/A-B testing manager.

        Parameters
        - config: ``ModelServingConfig`` for eventing, tracing, and Redis
        - model_manager: Optional reference to the primary ``ModelManager`` used for predictions
        - shadow_manager: Optional ``ShadowDeploymentManager`` for mirroring production traffic
        """
        self.config = config
        self.model_manager = model_manager
        self.shadow_manager = shadow_manager
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.model_variants: Dict[str, ModelVariant] = {}
        self.experiment_results: Dict[str, Dict[str, Any]] = {}
        self.traffic_router = TrafficRouter()
        self.metrics_collector = ExperimentMetricsCollector()
        
        # Initialize tracing and events
        self.tracer = get_ml_tracer("ensemble-manager")
        self.event_publisher = create_event_publisher(config.ml_redis_url)
    
    def set_model_manager(self, model_manager: "ModelManager"):
        """Attach the primary model manager after creation."""
        self.model_manager = model_manager
    
    def set_shadow_manager(self, shadow_manager: "ShadowDeploymentManager"):
        """Attach a shadow deployment manager after creation."""
        self.shadow_manager = shadow_manager
    
    async def create_experiment(self, experiment_config: ExperimentConfig) -> bool:
        """Create a new A/B testing experiment."""
        
        try:
            # Validate experiment configuration
            if not self._validate_experiment_config(experiment_config):
                return False
            
            # Check traffic allocation
            total_traffic = sum(v.traffic_percentage for v in experiment_config.variants)
            if abs(total_traffic - 100.0) > 0.1:
                logger.error("Traffic allocation must sum to 100%", total=total_traffic)
                return False
            
            # Store experiment
            self.active_experiments[experiment_config.experiment_id] = experiment_config
            
            # Register model variants
            for variant in experiment_config.variants:
                variant_key = f"{experiment_config.experiment_id}:{variant.model_name}:{variant.model_version}"
                self.model_variants[variant_key] = variant
            
            # Initialize metrics collection
            self.metrics_collector.initialize_experiment(experiment_config)
            
            logger.info(
                "A/B testing experiment created",
                experiment_id=experiment_config.experiment_id,
                variants=len(experiment_config.variants),
                strategy=experiment_config.traffic_split_strategy.value
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to create experiment", error=str(e))
            return False
    
    async def predict(
        self,
        inputs: List[Dict[str, Any]],
        model_name: str = "default",
        model_version: str = "latest",
        experiment_id: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        use_experiments: bool = True
    ) -> Dict[str, Any]:
        """Generate predictions with optional experiment routing and shadow comparison."""
        
        if self.model_manager is None:
            raise RuntimeError("Model manager is not configured for the ensemble manager")
        
        request_payload: Dict[str, Any] = request_context.copy() if request_context else {"inputs": inputs}
        request_payload.setdefault("inputs", inputs)
        
        routed_model_name = model_name
        routed_model_version = model_version
        routing_metadata: Dict[str, Any] = {}
        selected_experiment_id: Optional[str] = None
        
        if use_experiments and self.active_experiments:
            routed_model_name, routed_model_version, routing_metadata = await self.route_prediction_request(
                request_data=request_payload,
                user_id=user_id,
                session_id=session_id,
                experiment_id=experiment_id
            )
            selected_experiment_id = routing_metadata.get("experiment_id")
        else:
            routing_metadata = {"experiments_enabled": False}
        
        start_time = time.perf_counter()
        latency_ms = 0.0
        predictions: Optional[List[Any]] = None
        error_message: Optional[str] = None
        success = False
        
        try:
            predictions = await self.model_manager.predict(
                inputs=inputs,
                model_name=routed_model_name,
                model_version=routed_model_version
            )
            success = True
        except Exception as exc:
            error_message = str(exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if selected_experiment_id:
                variant_key = f"{routed_model_name}:{routed_model_version}"
                await self.record_prediction_result(
                    experiment_id=selected_experiment_id,
                    variant_key=variant_key,
                    request_data=request_payload,
                    prediction_result=predictions,
                    latency_ms=latency_ms,
                    success=success,
                    error=error_message
                )
            
            if success and predictions is not None:
                await self._process_shadow_deployments(
                    production_model=routed_model_name,
                    production_version=routed_model_version,
                    request_context=request_payload,
                    predictions=predictions,
                    latency_ms=latency_ms
                )
        
        return {
            "predictions": predictions if predictions is not None else [],
            "model_name": routed_model_name,
            "model_version": routed_model_version,
            "latency_ms": latency_ms,
            "experiment_id": selected_experiment_id,
            "routing_metadata": routing_metadata
        }
    
    async def route_prediction_request(
        self,
        request_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Route prediction request to appropriate model variant."""
        
        # Check if there are active experiments
        if not self.active_experiments:
            # No experiments, use default model
            return "default", "latest", {}
        
        experiment: Optional[ExperimentConfig] = None
        
        if experiment_id:
            experiment = self.active_experiments.get(experiment_id)
            if experiment is None:
                logger.warning(
                    "Requested experiment not active, falling back to first active experiment",
                    experiment_id=experiment_id
                )
        
        # Select experiment (for now, use the first active one)
        if experiment is None:
            experiment = next(iter(self.active_experiments.values()))
        
        # Route traffic based on strategy
        selected_variant = await self.traffic_router.route_request(
            experiment=experiment,
            request_data=request_data,
            user_id=user_id,
            session_id=session_id
        )
        
        # Record routing decision
        routing_metadata = {
            "experiment_id": experiment.experiment_id,
            "variant_selected": f"{selected_variant.model_name}:{selected_variant.model_version}",
            "traffic_percentage": selected_variant.traffic_percentage,
            "routing_strategy": experiment.traffic_split_strategy.value,
            "is_champion": selected_variant.is_champion,
            "is_challenger": selected_variant.is_challenger,
            "experiment_id": experiment.experiment_id
        }
        
        return selected_variant.model_name, selected_variant.model_version, routing_metadata
    
    async def record_prediction_result(
        self,
        experiment_id: str,
        variant_key: str,
        request_data: Dict[str, Any],
        prediction_result: Any,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record prediction result for experiment analysis."""
        
        await self.metrics_collector.record_prediction(
            experiment_id=experiment_id,
            variant_key=variant_key,
            request_data=request_data,
            prediction_result=prediction_result,
            latency_ms=latency_ms,
            success=success,
            error=error
        )
    
    async def _process_shadow_deployments(
        self,
        production_model: str,
        production_version: str,
        request_context: Dict[str, Any],
        predictions: Any,
        latency_ms: float
    ):
        """Mirror production traffic to configured shadow models."""
        
        if self.shadow_manager is None:
            return
        
        matching_shadows = [
            (shadow_key, config)
            for shadow_key, config in self.shadow_manager.shadow_models.items()
            if config.get("production_model") == production_model
        ]
        
        for shadow_key, _ in matching_shadows:
            try:
                await self.shadow_manager.process_shadow_request(
                    shadow_key=shadow_key,
                    request_data=request_context,
                    production_result=predictions,
                    production_latency_ms=latency_ms
                )
            except Exception as exc:
                logger.warning(
                    "Shadow request forwarding failed",
                    shadow_key=shadow_key,
                    production_model=production_model,
                    production_version=production_version,
                    error=str(exc)
                )
    
    async def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an experiment."""
        
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        metrics = await self.metrics_collector.get_experiment_metrics(experiment_id)
        
        # Calculate statistical significance
        significance_results = await self._calculate_statistical_significance(experiment_id, metrics)
        
        status = {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": "running" if experiment.end_time is None else "completed",
            "start_time": experiment.start_time,
            "end_time": experiment.end_time,
            "duration_hours": (time.time() - experiment.start_time) / 3600,
            "variants": [
                {
                    "model_name": v.model_name,
                    "model_version": v.model_version,
                    "traffic_percentage": v.traffic_percentage,
                    "is_champion": v.is_champion,
                    "is_challenger": v.is_challenger
                }
                for v in experiment.variants
            ],
            "metrics": metrics,
            "statistical_significance": significance_results,
            "recommendation": self._generate_experiment_recommendation(metrics, significance_results)
        }
        
        return status
    
    async def stop_experiment(self, experiment_id: str, winner_variant: Optional[str] = None) -> bool:
        """Stop an A/B testing experiment."""
        
        try:
            if experiment_id not in self.active_experiments:
                logger.error("Experiment not found", experiment_id=experiment_id)
                return False
            
            experiment = self.active_experiments[experiment_id]
            experiment.end_time = time.time()
            
            # Get final results
            final_metrics = await self.metrics_collector.get_experiment_metrics(experiment_id)
            significance_results = await self._calculate_statistical_significance(experiment_id, final_metrics)
            
            # Store final results
            self.experiment_results[experiment_id] = {
                "experiment_config": experiment,
                "final_metrics": final_metrics,
                "statistical_significance": significance_results,
                "winner_variant": winner_variant,
                "stopped_at": time.time()
            }
            
            # Clean up active experiment
            del self.active_experiments[experiment_id]
            
            # Remove variant registrations
            for variant in experiment.variants:
                variant_key = f"{experiment_id}:{variant.model_name}:{variant.model_version}"
                self.model_variants.pop(variant_key, None)
            
            logger.info(
                "Experiment stopped",
                experiment_id=experiment_id,
                winner=winner_variant,
                duration_hours=(experiment.end_time - experiment.start_time) / 3600
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to stop experiment", experiment_id=experiment_id, error=str(e))
            return False
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration."""
        
        if not config.variants:
            logger.error("Experiment must have at least one variant")
            return False
        
        if len(config.variants) > 10:
            logger.error("Too many variants", count=len(config.variants))
            return False
        
        # Check for duplicate model variants
        variant_keys = [(v.model_name, v.model_version) for v in config.variants]
        if len(variant_keys) != len(set(variant_keys)):
            logger.error("Duplicate model variants found")
            return False
        
        # Validate traffic percentages
        for variant in config.variants:
            if variant.traffic_percentage < 0 or variant.traffic_percentage > 100:
                logger.error("Invalid traffic percentage", percentage=variant.traffic_percentage)
                return False
        
        return True
    
    async def _calculate_statistical_significance(
        self,
        experiment_id: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate statistical significance of experiment results."""
        
        try:
            # This is a simplified implementation
            # In production, you'd use proper statistical tests
            
            variant_metrics = metrics.get("by_variant", {})
            
            if len(variant_metrics) < 2:
                return {"significant": False, "reason": "insufficient_variants"}
            
            # Compare champion vs challenger(s)
            champion_metrics = None
            challenger_metrics = []
            
            for variant_key, variant_data in variant_metrics.items():
                if variant_data.get("is_champion"):
                    champion_metrics = variant_data
                elif variant_data.get("is_challenger"):
                    challenger_metrics.append(variant_data)
            
            if not champion_metrics or not challenger_metrics:
                return {"significant": False, "reason": "no_champion_challenger_setup"}
            
            # Simple significance test based on sample size and performance difference
            champion_latency = champion_metrics.get("avg_latency_ms", 0)
            champion_samples = champion_metrics.get("total_requests", 0)
            
            significance_results = {}
            
            for challenger in challenger_metrics:
                challenger_latency = challenger.get("avg_latency_ms", 0)
                challenger_samples = challenger.get("total_requests", 0)
                
                # Check minimum sample size
                min_samples = 1000
                if champion_samples < min_samples or challenger_samples < min_samples:
                    significance_results[challenger["variant_key"]] = {
                        "significant": False,
                        "reason": "insufficient_samples",
                        "champion_samples": champion_samples,
                        "challenger_samples": challenger_samples
                    }
                    continue
                
                # Simple difference test (in production, use proper t-test)
                latency_diff_percent = abs(champion_latency - challenger_latency) / champion_latency * 100
                
                significance_results[challenger["variant_key"]] = {
                    "significant": latency_diff_percent > 5,  # 5% difference threshold
                    "latency_difference_percent": latency_diff_percent,
                    "champion_latency_ms": champion_latency,
                    "challenger_latency_ms": challenger_latency,
                    "sample_sizes": {
                        "champion": champion_samples,
                        "challenger": challenger_samples
                    }
                }
            
            return significance_results
            
        except Exception as e:
            logger.error("Statistical significance calculation failed", error=str(e))
            return {"error": str(e)}
    
    def _generate_experiment_recommendation(
        self,
        metrics: Dict[str, Any],
        significance_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendation based on experiment results."""
        
        try:
            recommendations = []
            
            # Analyze significance results
            for variant_key, sig_result in significance_results.items():
                if isinstance(sig_result, dict) and sig_result.get("significant"):
                    champion_latency = sig_result.get("champion_latency_ms", 0)
                    challenger_latency = sig_result.get("challenger_latency_ms", 0)
                    
                    if challenger_latency < champion_latency:
                        recommendations.append({
                            "action": "promote_challenger",
                            "variant": variant_key,
                            "reason": f"Challenger shows {((champion_latency - challenger_latency) / champion_latency * 100):.1f}% latency improvement",
                            "confidence": "high" if sig_result.get("latency_difference_percent", 0) > 10 else "medium"
                        })
                    else:
                        recommendations.append({
                            "action": "keep_champion",
                            "reason": f"Champion performs {((challenger_latency - champion_latency) / champion_latency * 100):.1f}% better than challenger",
                            "confidence": "high"
                        })
            
            if not recommendations:
                recommendations.append({
                    "action": "continue_experiment",
                    "reason": "No statistically significant differences found",
                    "confidence": "low"
                })
            
            return {
                "recommendations": recommendations,
                "generated_at": time.time()
            }
            
        except Exception as e:
            logger.error("Failed to generate recommendation", error=str(e))
            return {"error": str(e)}


class TrafficRouter:
    """Routes traffic between model variants."""
    
    def __init__(self):
        """Prepare routing caches for sticky sessions."""
        self.routing_cache: Dict[str, str] = {}  # For sticky sessions
    
    async def route_request(
        self,
        experiment: ExperimentConfig,
        request_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ModelVariant:
        """Route request to appropriate model variant."""
        
        if experiment.traffic_split_strategy == TrafficSplitStrategy.RANDOM:
            return self._route_random(experiment)
        
        elif experiment.traffic_split_strategy == TrafficSplitStrategy.WEIGHTED:
            return self._route_weighted(experiment)
        
        elif experiment.traffic_split_strategy == TrafficSplitStrategy.STICKY_SESSION:
            return self._route_sticky_session(experiment, session_id or user_id)
        
        elif experiment.traffic_split_strategy == TrafficSplitStrategy.FEATURE_BASED:
            return self._route_feature_based(experiment, request_data)
        
        else:
            # Default to random
            return self._route_random(experiment)
    
    def _route_random(self, experiment: ExperimentConfig) -> ModelVariant:
        """Route using random selection based on traffic percentages."""
        
        rand_value = random.uniform(0, 100)
        cumulative_percentage = 0
        
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage
            if rand_value <= cumulative_percentage:
                return variant
        
        # Fallback to first variant
        return experiment.variants[0]
    
    def _route_weighted(self, experiment: ExperimentConfig) -> ModelVariant:
        """Route using weighted selection."""
        
        weights = [v.traffic_percentage for v in experiment.variants]
        selected_idx = np.random.choice(len(experiment.variants), p=np.array(weights) / 100)
        return experiment.variants[selected_idx]
    
    def _route_sticky_session(self, experiment: ExperimentConfig, session_key: Optional[str]) -> ModelVariant:
        """Route using sticky session strategy."""
        
        if not session_key:
            return self._route_random(experiment)
        
        cache_key = f"{experiment.experiment_id}:{session_key}"
        
        # Check if we already routed this session
        if cache_key in self.routing_cache:
            variant_key = self.routing_cache[cache_key]
            # Find the variant
            for variant in experiment.variants:
                if f"{variant.model_name}:{variant.model_version}" == variant_key:
                    return variant
        
        # First time routing for this session
        selected_variant = self._route_weighted(experiment)
        self.routing_cache[cache_key] = f"{selected_variant.model_name}:{selected_variant.model_version}"
        
        return selected_variant
    
    def _route_feature_based(self, experiment: ExperimentConfig, request_data: Dict[str, Any]) -> ModelVariant:
        """Route based on request features."""
        
        # Example: route based on input characteristics
        inputs = request_data.get("inputs", [])
        if inputs:
            first_input = inputs[0]
            
            # Route based on VIX level (example)
            vix = first_input.get("vix", 20)
            if vix > 30:  # High volatility
                # Prefer challenger for high volatility scenarios
                challengers = [v for v in experiment.variants if v.is_challenger]
                if challengers:
                    return challengers[0]
        
        # Default routing
        return self._route_weighted(experiment)


class ExperimentMetricsCollector:
    """Collects and analyzes experiment metrics."""
    
    def __init__(self):
        """Initialize in-memory structures for experiment telemetry."""
        self.experiment_data: Dict[str, Dict[str, List[Any]]] = {}
    
    def initialize_experiment(self, experiment: ExperimentConfig):
        """Initialize metrics collection for an experiment."""
        
        self.experiment_data[experiment.experiment_id] = {
            "requests": [],
            "predictions": [],
            "latencies": [],
            "errors": [],
            "by_variant": {}
        }
        
        # Initialize variant-specific tracking
        for variant in experiment.variants:
            variant_key = f"{variant.model_name}:{variant.model_version}"
            self.experiment_data[experiment.experiment_id]["by_variant"][variant_key] = {
                "requests": [],
                "predictions": [],
                "latencies": [],
                "errors": []
            }
    
    async def record_prediction(
        self,
        experiment_id: str,
        variant_key: str,
        request_data: Dict[str, Any],
        prediction_result: Any,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record a prediction result."""
        
        if experiment_id not in self.experiment_data:
            return
        
        timestamp = time.time()
        
        record = {
            "timestamp": timestamp,
            "variant_key": variant_key,
            "request_data": request_data,
            "prediction_result": prediction_result,
            "latency_ms": latency_ms,
            "success": success,
            "error": error
        }
        
        # Store in overall experiment data
        exp_data = self.experiment_data[experiment_id]
        exp_data["requests"].append(record)
        exp_data["latencies"].append(latency_ms)
        
        if not success:
            exp_data["errors"].append(record)
        else:
            exp_data["predictions"].append(record)
        
        # Store in variant-specific data
        if variant_key in exp_data["by_variant"]:
            variant_data = exp_data["by_variant"][variant_key]
            variant_data["requests"].append(record)
            variant_data["latencies"].append(latency_ms)
            
            if not success:
                variant_data["errors"].append(record)
            else:
                variant_data["predictions"].append(record)
    
    async def get_experiment_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for an experiment."""
        
        if experiment_id not in self.experiment_data:
            return {}
        
        exp_data = self.experiment_data[experiment_id]
        
        # Overall metrics
        total_requests = len(exp_data["requests"])
        total_errors = len(exp_data["errors"])
        latencies = exp_data["latencies"]
        
        overall_metrics = {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests) * 100 if total_requests > 0 else 0,
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0
        }
        
        # Variant-specific metrics
        variant_metrics = {}
        for variant_key, variant_data in exp_data["by_variant"].items():
            variant_requests = len(variant_data["requests"])
            variant_errors = len(variant_data["errors"])
            variant_latencies = variant_data["latencies"]
            
            variant_metrics[variant_key] = {
                "variant_key": variant_key,
                "total_requests": variant_requests,
                "total_errors": variant_errors,
                "error_rate": (variant_errors / variant_requests) * 100 if variant_requests > 0 else 0,
                "avg_latency_ms": np.mean(variant_latencies) if variant_latencies else 0,
                "p95_latency_ms": np.percentile(variant_latencies, 95) if variant_latencies else 0,
                "traffic_share": (variant_requests / total_requests) * 100 if total_requests > 0 else 0
            }
        
        return {
            "overall": overall_metrics,
            "by_variant": variant_metrics,
            "collected_at": time.time()
        }


class ShadowDeploymentManager:
    """Manages shadow deployments for model validation."""
    
    def __init__(
        self,
        config: ModelServingConfig,
        model_manager: Optional["ModelManager"] = None
    ):
        self.config = config
        self.model_manager = model_manager
        self.shadow_models: Dict[str, Dict[str, Any]] = {}
        self.comparison_results: Dict[str, List[Dict[str, Any]]] = {}
    
    def set_model_manager(self, model_manager: "ModelManager"):
        """Attach the primary model manager after creation."""
        self.model_manager = model_manager
    
    async def enable_shadow_deployment(
        self,
        production_model: str,
        shadow_model: str,
        shadow_model_version: str,
        traffic_percentage: float = 100.0
    ) -> bool:
        """Enable shadow deployment for a model."""
        
        try:
            shadow_key = f"{shadow_model}:{shadow_model_version}"
            
            self.shadow_models[shadow_key] = {
                "production_model": production_model,
                "shadow_model": shadow_model,
                "shadow_model_version": shadow_model_version,
                "traffic_percentage": traffic_percentage,
                "enabled_at": time.time(),
                "request_count": 0,
                "comparison_count": 0
            }
            
            self.comparison_results[shadow_key] = []
            
            logger.info(
                "Shadow deployment enabled",
                production_model=production_model,
                shadow_model=shadow_model,
                shadow_version=shadow_model_version,
                traffic_percentage=traffic_percentage
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to enable shadow deployment", error=str(e))
            return False
    
    async def process_shadow_request(
        self,
        shadow_key: str,
        request_data: Dict[str, Any],
        production_result: Any,
        production_latency_ms: float
    ):
        """Process request through shadow model and compare results."""
        
        if shadow_key not in self.shadow_models:
            return
        
        shadow_config = self.shadow_models[shadow_key]
        shadow_config["request_count"] += 1
        
        inputs = request_data.get("inputs")
        if not inputs:
            logger.warning(
                "Shadow deployment request missing inputs",
                shadow_key=shadow_key
            )
            return
        
        if self.model_manager is None:
            logger.warning(
                "Shadow deployment manager is not attached to a model manager",
                shadow_key=shadow_key
            )
            return
        
        # Check if we should process this request
        if random.uniform(0, 100) > shadow_config["traffic_percentage"]:
            return
        
        try:
            # Make shadow prediction (this would call the actual shadow model)
            shadow_start = time.perf_counter()
            shadow_result = await self.model_manager.predict(
                inputs=inputs,
                model_name=shadow_config["shadow_model"],
                model_version=shadow_config["shadow_model_version"]
            )
            shadow_latency_ms = (time.perf_counter() - shadow_start) * 1000
            
            # Compare results
            comparison = {
                "timestamp": time.time(),
                "request_data": request_data,
                "production_result": production_result,
                "shadow_result": shadow_result,
                "production_latency_ms": production_latency_ms,
                "shadow_latency_ms": shadow_latency_ms,
                "latency_difference_ms": shadow_latency_ms - production_latency_ms,
                "results_match": self._compare_predictions(production_result, shadow_result)
            }
            
            self.comparison_results[shadow_key].append(comparison)
            shadow_config["comparison_count"] += 1
            
            # Limit stored comparisons
            if len(self.comparison_results[shadow_key]) > 10000:
                self.comparison_results[shadow_key] = self.comparison_results[shadow_key][-5000:]
            
        except Exception as e:
            logger.error(
                "Shadow request processing failed",
                shadow_key=shadow_key,
                error=str(e)
            )
    
    def _compare_predictions(self, prod_result: Any, shadow_result: Any) -> bool:
        """Compare production and shadow predictions."""
        
        try:
            # Simple comparison - in production, this would be more sophisticated
            if isinstance(prod_result, list) and isinstance(shadow_result, list):
                if len(prod_result) != len(shadow_result):
                    return False
                
                # Compare numerical predictions with tolerance
                for p_pred, s_pred in zip(prod_result, shadow_result):
                    if isinstance(p_pred, (int, float)) and isinstance(s_pred, (int, float)):
                        if abs(p_pred - s_pred) / abs(p_pred) > 0.05:  # 5% tolerance
                            return False
                
                return True
            
            return prod_result == shadow_result
            
        except Exception:
            return False
    
    async def get_shadow_deployment_report(self, shadow_key: str) -> Optional[Dict[str, Any]]:
        """Get shadow deployment comparison report."""
        
        if shadow_key not in self.shadow_models:
            return None
        
        shadow_config = self.shadow_models[shadow_key]
        comparisons = self.comparison_results.get(shadow_key, [])
        
        if not comparisons:
            return {
                "shadow_key": shadow_key,
                "status": "no_data",
                "message": "No comparison data available"
            }
        
        # Analyze comparisons
        latency_diffs = [c["latency_difference_ms"] for c in comparisons]
        match_rate = sum(1 for c in comparisons if c["results_match"]) / len(comparisons) * 100
        
        report = {
            "shadow_key": shadow_key,
            "production_model": shadow_config["production_model"],
            "shadow_model": shadow_config["shadow_model"],
            "shadow_version": shadow_config["shadow_model_version"],
            "enabled_duration_hours": (time.time() - shadow_config["enabled_at"]) / 3600,
            "total_comparisons": len(comparisons),
            "results_match_rate": match_rate,
            "latency_comparison": {
                "mean_difference_ms": np.mean(latency_diffs),
                "median_difference_ms": np.median(latency_diffs),
                "p95_difference_ms": np.percentile(latency_diffs, 95),
                "shadow_faster_percentage": sum(1 for d in latency_diffs if d < 0) / len(latency_diffs) * 100
            },
            "recommendation": self._generate_shadow_recommendation(match_rate, latency_diffs)
        }
        
        return report
    
    def _generate_shadow_recommendation(self, match_rate: float, latency_diffs: List[float]) -> Dict[str, str]:
        """Generate recommendation for shadow deployment."""
        
        avg_latency_diff = np.mean(latency_diffs)
        
        if match_rate < 95:
            return {
                "action": "investigate",
                "reason": f"Low result match rate: {match_rate:.1f}%",
                "priority": "high"
            }
        elif avg_latency_diff < -10:  # Shadow is 10ms faster on average
            return {
                "action": "promote",
                "reason": f"Shadow model is {abs(avg_latency_diff):.1f}ms faster with {match_rate:.1f}% match rate",
                "priority": "medium"
            }
        elif avg_latency_diff > 50:  # Shadow is 50ms slower
            return {
                "action": "reject",
                "reason": f"Shadow model is {avg_latency_diff:.1f}ms slower",
                "priority": "low"
            }
        else:
            return {
                "action": "continue_monitoring",
                "reason": f"Similar performance: {avg_latency_diff:.1f}ms difference, {match_rate:.1f}% match rate",
                "priority": "low"
            }


def create_ensemble_manager(
    config: ModelServingConfig,
    model_manager: Optional["ModelManager"] = None,
    shadow_manager: Optional[ShadowDeploymentManager] = None
) -> EnsembleManager:
    """Create ensemble manager."""
    manager = EnsembleManager(
        config=config,
        model_manager=model_manager,
        shadow_manager=shadow_manager
    )
    return manager


def create_shadow_deployment_manager(
    config: ModelServingConfig,
    model_manager: Optional["ModelManager"] = None
) -> ShadowDeploymentManager:
    """Create shadow deployment manager."""
    manager = ShadowDeploymentManager(config=config, model_manager=model_manager)
    return manager
