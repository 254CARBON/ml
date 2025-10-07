"""Statistical drift detection and automated retraining system."""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import structlog

from libs.common.config import BaseConfig
from libs.common.events import create_event_publisher
from libs.common.logging import configure_logging

logger = structlog.get_logger("drift_detection")


class DriftType(Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"           # Input data distribution changes
    CONCEPT_DRIFT = "concept_drift"     # Input-output relationship changes
    PERFORMANCE_DRIFT = "performance_drift"  # Model performance degradation


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    drift_type: DriftType
    detected: bool
    severity: str  # low, medium, high, critical
    drift_score: float
    threshold: float
    timestamp: float
    model_name: str
    model_version: str
    details: Dict[str, Any]
    recommendation: str


class DataDriftDetector:
    """Detects drift in input data distributions."""
    
    def __init__(self, reference_window_size: int = 1000, detection_window_size: int = 100):
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.reference_distributions: Dict[str, Dict[str, Any]] = {}
    
    def set_reference_distribution(
        self,
        model_name: str,
        feature_name: str,
        reference_data: np.ndarray
    ):
        """Set reference distribution for a feature."""
        
        if model_name not in self.reference_distributions:
            self.reference_distributions[model_name] = {}
        
        # Calculate reference statistics
        self.reference_distributions[model_name][feature_name] = {
            "mean": np.mean(reference_data),
            "std": np.std(reference_data),
            "min": np.min(reference_data),
            "max": np.max(reference_data),
            "percentiles": np.percentile(reference_data, [25, 50, 75, 95, 99]),
            "sample_size": len(reference_data),
            "timestamp": time.time()
        }
        
        logger.info("Reference distribution set", 
                   model_name=model_name,
                   feature_name=feature_name,
                   sample_size=len(reference_data))
    
    def detect_drift(
        self,
        model_name: str,
        feature_name: str,
        current_data: np.ndarray,
        threshold: float = 0.05
    ) -> DriftDetectionResult:
        """Detect drift using statistical tests."""
        
        if (model_name not in self.reference_distributions or 
            feature_name not in self.reference_distributions[model_name]):
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                detected=False,
                severity="unknown",
                drift_score=0.0,
                threshold=threshold,
                timestamp=time.time(),
                model_name=model_name,
                model_version="unknown",
                details={"error": "No reference distribution available"},
                recommendation="Set reference distribution first"
            )
        
        reference_stats = self.reference_distributions[model_name][feature_name]
        
        # Kolmogorov-Smirnov test
        reference_sample = np.random.normal(
            reference_stats["mean"],
            reference_stats["std"],
            size=min(1000, len(current_data))
        )
        
        ks_statistic, ks_p_value = stats.ks_2samp(reference_sample, current_data)
        
        # Population Stability Index (PSI)
        psi_score = self._calculate_psi(reference_stats, current_data)
        
        # Jensen-Shannon Divergence
        js_divergence = self._calculate_js_divergence(reference_sample, current_data)
        
        # Determine drift based on multiple tests
        drift_detected = (
            ks_p_value < threshold or
            psi_score > 0.2 or  # PSI threshold
            js_divergence > 0.1  # JS divergence threshold
        )
        
        # Calculate overall drift score
        drift_score = max(1 - ks_p_value, psi_score, js_divergence)
        
        # Determine severity
        if drift_score > 0.8:
            severity = "critical"
        elif drift_score > 0.5:
            severity = "high"
        elif drift_score > 0.3:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate recommendation
        if drift_detected and severity in ["high", "critical"]:
            recommendation = "Immediate model retraining recommended"
        elif drift_detected and severity == "medium":
            recommendation = "Schedule model retraining within 7 days"
        elif drift_detected and severity == "low":
            recommendation = "Monitor closely, consider retraining if drift persists"
        else:
            recommendation = "No action required"
        
        result = DriftDetectionResult(
            drift_type=DriftType.DATA_DRIFT,
            detected=drift_detected,
            severity=severity,
            drift_score=drift_score,
            threshold=threshold,
            timestamp=time.time(),
            model_name=model_name,
            model_version="current",
            details={
                "feature_name": feature_name,
                "ks_statistic": ks_statistic,
                "ks_p_value": ks_p_value,
                "psi_score": psi_score,
                "js_divergence": js_divergence,
                "current_sample_size": len(current_data),
                "reference_sample_size": reference_stats["sample_size"],
                "current_mean": np.mean(current_data),
                "reference_mean": reference_stats["mean"],
                "current_std": np.std(current_data),
                "reference_std": reference_stats["std"]
            },
            recommendation=recommendation
        )
        
        logger.info("Data drift detection completed",
                   model_name=model_name,
                   feature_name=feature_name,
                   drift_detected=drift_detected,
                   severity=severity,
                   drift_score=drift_score)
        
        return result
    
    def _calculate_psi(self, reference_stats: Dict[str, Any], current_data: np.ndarray) -> float:
        """Calculate Population Stability Index."""
        
        try:
            # Create bins based on reference percentiles
            bins = [-np.inf] + list(reference_stats["percentiles"]) + [np.inf]
            
            # Calculate expected frequencies (from reference)
            expected_freq = np.array([0.25, 0.25, 0.25, 0.20, 0.04, 0.01])  # Approximate
            
            # Calculate actual frequencies
            actual_counts, _ = np.histogram(current_data, bins=bins)
            actual_freq = actual_counts / len(current_data)
            
            # Avoid division by zero
            expected_freq = np.maximum(expected_freq, 1e-10)
            actual_freq = np.maximum(actual_freq, 1e-10)
            
            # Calculate PSI
            psi = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
            
            return psi
            
        except Exception as e:
            logger.error("PSI calculation failed", error=str(e))
            return 0.0
    
    def _calculate_js_divergence(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence."""
        
        try:
            # Create histograms
            bins = np.linspace(
                min(np.min(reference_data), np.min(current_data)),
                max(np.max(reference_data), np.max(current_data)),
                50
            )
            
            ref_hist, _ = np.histogram(reference_data, bins=bins, density=True)
            cur_hist, _ = np.histogram(current_data, bins=bins, density=True)
            
            # Normalize to probabilities
            ref_prob = ref_hist / np.sum(ref_hist)
            cur_prob = cur_hist / np.sum(cur_hist)
            
            # Avoid zeros
            ref_prob = np.maximum(ref_prob, 1e-10)
            cur_prob = np.maximum(cur_prob, 1e-10)
            
            # Calculate JS divergence
            m = 0.5 * (ref_prob + cur_prob)
            js_div = 0.5 * stats.entropy(ref_prob, m) + 0.5 * stats.entropy(cur_prob, m)
            
            return js_div
            
        except Exception as e:
            logger.error("JS divergence calculation failed", error=str(e))
            return 0.0


class ConceptDriftDetector:
    """Detects drift in the relationship between inputs and outputs."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.reference_performance: Dict[str, Dict[str, float]] = {}
    
    def set_reference_performance(
        self,
        model_name: str,
        reference_predictions: np.ndarray,
        reference_actuals: np.ndarray
    ):
        """Set reference performance metrics."""
        
        mse = mean_squared_error(reference_actuals, reference_predictions)
        mae = mean_absolute_error(reference_actuals, reference_predictions)
        
        # Calculate correlation
        correlation = np.corrcoef(reference_predictions.flatten(), reference_actuals.flatten())[0, 1]
        
        self.reference_performance[model_name] = {
            "mse": mse,
            "mae": mae,
            "correlation": correlation,
            "sample_size": len(reference_predictions),
            "timestamp": time.time()
        }
        
        logger.info("Reference performance set",
                   model_name=model_name,
                   mse=mse,
                   mae=mae,
                   correlation=correlation)
    
    def detect_concept_drift(
        self,
        model_name: str,
        current_predictions: np.ndarray,
        current_actuals: np.ndarray,
        threshold: float = 0.1
    ) -> DriftDetectionResult:
        """Detect concept drift by comparing performance metrics."""
        
        if model_name not in self.reference_performance:
            return DriftDetectionResult(
                drift_type=DriftType.CONCEPT_DRIFT,
                detected=False,
                severity="unknown",
                drift_score=0.0,
                threshold=threshold,
                timestamp=time.time(),
                model_name=model_name,
                model_version="unknown",
                details={"error": "No reference performance available"},
                recommendation="Set reference performance first"
            )
        
        reference = self.reference_performance[model_name]
        
        # Calculate current performance
        current_mse = mean_squared_error(current_actuals, current_predictions)
        current_mae = mean_absolute_error(current_actuals, current_predictions)
        current_correlation = np.corrcoef(current_predictions.flatten(), current_actuals.flatten())[0, 1]
        
        # Calculate performance degradation
        mse_degradation = (current_mse - reference["mse"]) / reference["mse"]
        mae_degradation = (current_mae - reference["mae"]) / reference["mae"]
        correlation_degradation = (reference["correlation"] - current_correlation) / reference["correlation"]
        
        # Overall drift score
        drift_score = max(mse_degradation, mae_degradation, correlation_degradation)
        
        # Detect drift
        drift_detected = drift_score > threshold
        
        # Determine severity
        if drift_score > 0.5:
            severity = "critical"
        elif drift_score > 0.3:
            severity = "high"
        elif drift_score > 0.15:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate recommendation
        if drift_detected and severity in ["high", "critical"]:
            recommendation = "Immediate model retraining required"
        elif drift_detected and severity == "medium":
            recommendation = "Schedule model retraining within 3 days"
        else:
            recommendation = "Continue monitoring"
        
        result = DriftDetectionResult(
            drift_type=DriftType.CONCEPT_DRIFT,
            detected=drift_detected,
            severity=severity,
            drift_score=drift_score,
            threshold=threshold,
            timestamp=time.time(),
            model_name=model_name,
            model_version="current",
            details={
                "current_mse": current_mse,
                "reference_mse": reference["mse"],
                "mse_degradation": mse_degradation,
                "current_mae": current_mae,
                "reference_mae": reference["mae"],
                "mae_degradation": mae_degradation,
                "current_correlation": current_correlation,
                "reference_correlation": reference["correlation"],
                "correlation_degradation": correlation_degradation,
                "sample_size": len(current_predictions)
            },
            recommendation=recommendation
        )
        
        logger.info("Concept drift detection completed",
                   model_name=model_name,
                   drift_detected=drift_detected,
                   severity=severity,
                   drift_score=drift_score)
        
        return result


class PerformanceDriftDetector:
    """Detects drift in model performance over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_performance(
        self,
        model_name: str,
        model_version: str,
        latency_ms: float,
        accuracy: Optional[float] = None,
        error_rate: float = 0.0,
        throughput_rps: float = 0.0
    ):
        """Record performance metrics."""
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        performance_record = {
            "timestamp": time.time(),
            "model_version": model_version,
            "latency_ms": latency_ms,
            "accuracy": accuracy,
            "error_rate": error_rate,
            "throughput_rps": throughput_rps
        }
        
        self.performance_history[model_name].append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history[model_name]) > self.window_size * 2:
            self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]
    
    def detect_performance_drift(
        self,
        model_name: str,
        threshold: float = 0.2
    ) -> DriftDetectionResult:
        """Detect performance drift using trend analysis."""
        
        if (model_name not in self.performance_history or 
            len(self.performance_history[model_name]) < self.window_size):
            return DriftDetectionResult(
                drift_type=DriftType.PERFORMANCE_DRIFT,
                detected=False,
                severity="unknown",
                drift_score=0.0,
                threshold=threshold,
                timestamp=time.time(),
                model_name=model_name,
                model_version="unknown",
                details={"error": "Insufficient performance history"},
                recommendation="Collect more performance data"
            )
        
        history = self.performance_history[model_name]
        
        # Split into reference and current windows
        mid_point = len(history) // 2
        reference_window = history[:mid_point]
        current_window = history[mid_point:]
        
        # Calculate performance metrics for each window
        ref_metrics = self._calculate_window_metrics(reference_window)
        cur_metrics = self._calculate_window_metrics(current_window)
        
        # Calculate drift scores
        latency_drift = (cur_metrics["avg_latency"] - ref_metrics["avg_latency"]) / ref_metrics["avg_latency"]
        error_rate_drift = cur_metrics["avg_error_rate"] - ref_metrics["avg_error_rate"]
        throughput_drift = (ref_metrics["avg_throughput"] - cur_metrics["avg_throughput"]) / ref_metrics["avg_throughput"]
        
        # Overall drift score
        drift_score = max(abs(latency_drift), abs(error_rate_drift), abs(throughput_drift))
        
        # Detect drift
        drift_detected = drift_score > threshold
        
        # Determine severity
        if drift_score > 0.8:
            severity = "critical"
        elif drift_score > 0.5:
            severity = "high"
        elif drift_score > 0.3:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate recommendation
        if drift_detected and latency_drift > 0.3:
            recommendation = "Performance degradation detected - investigate and optimize"
        elif drift_detected and error_rate_drift > 0.05:
            recommendation = "Error rate increase detected - check model and data quality"
        elif drift_detected and throughput_drift > 0.3:
            recommendation = "Throughput decrease detected - check resource allocation"
        else:
            recommendation = "Continue monitoring performance"
        
        result = DriftDetectionResult(
            drift_type=DriftType.PERFORMANCE_DRIFT,
            detected=drift_detected,
            severity=severity,
            drift_score=drift_score,
            threshold=threshold,
            timestamp=time.time(),
            model_name=model_name,
            model_version=cur_metrics.get("latest_version", "unknown"),
            details={
                "latency_drift": latency_drift,
                "error_rate_drift": error_rate_drift,
                "throughput_drift": throughput_drift,
                "reference_metrics": ref_metrics,
                "current_metrics": cur_metrics,
                "sample_sizes": {
                    "reference": len(reference_window),
                    "current": len(current_window)
                }
            },
            recommendation=recommendation
        )
        
        logger.info("Performance drift detection completed",
                   model_name=model_name,
                   drift_detected=drift_detected,
                   severity=severity,
                   drift_score=drift_score)
        
        return result
    
    def _calculate_window_metrics(self, window: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for a performance window."""
        
        if not window:
            return {}
        
        latencies = [r["latency_ms"] for r in window if r["latency_ms"] is not None]
        error_rates = [r["error_rate"] for r in window if r["error_rate"] is not None]
        throughputs = [r["throughput_rps"] for r in window if r["throughput_rps"] is not None]
        
        return {
            "avg_latency": np.mean(latencies) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            "avg_error_rate": np.mean(error_rates) if error_rates else 0,
            "avg_throughput": np.mean(throughputs) if throughputs else 0,
            "latest_version": window[-1].get("model_version", "unknown")
        }


class DriftMonitor:
    """Comprehensive drift monitoring system."""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.data_drift_detector = DataDriftDetector()
        self.concept_drift_detector = ConceptDriftDetector()
        self.performance_drift_detector = PerformanceDriftDetector()
        
        self.monitoring_active = False
        self.drift_history: List[DriftDetectionResult] = []
        self.alert_thresholds = {
            "data_drift": 0.05,
            "concept_drift": 0.1,
            "performance_drift": 0.2
        }
        
        # Initialize event publisher
        self.event_publisher = create_event_publisher(config.ml_redis_url)
    
    async def start_monitoring(self, models: List[str]):
        """Start drift monitoring for specified models."""
        
        if self.monitoring_active:
            logger.warning("Drift monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks for each model
        monitoring_tasks = []
        for model_name in models:
            task = asyncio.create_task(self._monitor_model_drift(model_name))
            monitoring_tasks.append(task)
        
        logger.info("Drift monitoring started", models=models)
        
        try:
            await asyncio.gather(*monitoring_tasks)
        except Exception as e:
            logger.error("Drift monitoring error", error=str(e))
        finally:
            self.monitoring_active = False
    
    async def _monitor_model_drift(self, model_name: str):
        """Monitor drift for a specific model."""
        
        logger.info("Starting drift monitoring for model", model_name=model_name)
        
        while self.monitoring_active:
            try:
                # Collect recent data and performance metrics
                # This would typically query databases and metrics stores
                
                # Simulate data collection
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Run drift detection
                await self._run_drift_detection(model_name)
                
            except Exception as e:
                logger.error("Model drift monitoring error", 
                           model_name=model_name, 
                           error=str(e))
                await asyncio.sleep(60)  # Wait before retry
    
    async def _run_drift_detection(self, model_name: str):
        """Run comprehensive drift detection for a model."""
        
        try:
            drift_results = []
            
            # Data drift detection
            # This would get actual input data from recent predictions
            current_data = np.random.normal(0, 1, 200)  # Simulate current data
            
            data_drift_result = self.data_drift_detector.detect_drift(
                model_name=model_name,
                feature_name="primary_features",
                current_data=current_data,
                threshold=self.alert_thresholds["data_drift"]
            )
            drift_results.append(data_drift_result)
            
            # Performance drift detection
            perf_drift_result = self.performance_drift_detector.detect_performance_drift(
                model_name=model_name,
                threshold=self.alert_thresholds["performance_drift"]
            )
            drift_results.append(perf_drift_result)
            
            # Store results
            self.drift_history.extend(drift_results)
            
            # Keep only recent history
            if len(self.drift_history) > 10000:
                self.drift_history = self.drift_history[-5000:]
            
            # Check for alerts
            for result in drift_results:
                if result.detected and result.severity in ["high", "critical"]:
                    await self._send_drift_alert(result)
                    
                    # Trigger automated retraining if critical
                    if result.severity == "critical":
                        await self._trigger_automated_retraining(result)
            
        except Exception as e:
            logger.error("Drift detection failed", model_name=model_name, error=str(e))
    
    async def _send_drift_alert(self, drift_result: DriftDetectionResult):
        """Send drift detection alert."""
        
        try:
            alert_data = {
                "alert_type": "model_drift_detected",
                "model_name": drift_result.model_name,
                "drift_type": drift_result.drift_type.value,
                "severity": drift_result.severity,
                "drift_score": drift_result.drift_score,
                "recommendation": drift_result.recommendation,
                "timestamp": drift_result.timestamp,
                "details": drift_result.details
            }
            
            # Publish alert event
            await self.event_publisher.redis_client.publish(
                "ml_alerts",
                json.dumps(alert_data, default=str)
            )
            
            logger.warning("Drift alert sent",
                          model_name=drift_result.model_name,
                          drift_type=drift_result.drift_type.value,
                          severity=drift_result.severity)
            
        except Exception as e:
            logger.error("Failed to send drift alert", error=str(e))
    
    async def _trigger_automated_retraining(self, drift_result: DriftDetectionResult):
        """Trigger automated model retraining."""
        
        try:
            retraining_request = {
                "model_name": drift_result.model_name,
                "trigger_reason": f"{drift_result.drift_type.value}_detected",
                "drift_severity": drift_result.severity,
                "drift_score": drift_result.drift_score,
                "triggered_at": time.time(),
                "priority": "high" if drift_result.severity == "critical" else "medium"
            }
            
            # Publish retraining request
            await self.event_publisher.redis_client.publish(
                "model_retraining_requests",
                json.dumps(retraining_request, default=str)
            )
            
            logger.info("Automated retraining triggered",
                       model_name=drift_result.model_name,
                       trigger_reason=retraining_request["trigger_reason"])
            
        except Exception as e:
            logger.error("Failed to trigger automated retraining", error=str(e))
    
    def get_drift_summary(self, model_name: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get drift detection summary."""
        
        cutoff_time = time.time() - (hours * 3600)
        
        # Filter recent drift results
        recent_results = [
            r for r in self.drift_history
            if r.timestamp > cutoff_time and (model_name is None or r.model_name == model_name)
        ]
        
        if not recent_results:
            return {"no_data": True, "message": "No recent drift detection results"}
        
        # Aggregate statistics
        drift_counts = {}
        severity_counts = {}
        
        for result in recent_results:
            # Count by drift type
            drift_type = result.drift_type.value
            if drift_type not in drift_counts:
                drift_counts[drift_type] = {"detected": 0, "total": 0}
            
            drift_counts[drift_type]["total"] += 1
            if result.detected:
                drift_counts[drift_type]["detected"] += 1
            
            # Count by severity
            if result.detected:
                severity = result.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Calculate drift rates
        drift_rates = {}
        for drift_type, counts in drift_counts.items():
            drift_rates[drift_type] = counts["detected"] / counts["total"] * 100 if counts["total"] > 0 else 0
        
        summary = {
            "summary_period_hours": hours,
            "total_checks": len(recent_results),
            "drift_detected_count": sum(1 for r in recent_results if r.detected),
            "drift_detection_rate": sum(1 for r in recent_results if r.detected) / len(recent_results) * 100,
            "drift_rates_by_type": drift_rates,
            "severity_distribution": severity_counts,
            "models_monitored": list(set(r.model_name for r in recent_results)),
            "latest_check": max(r.timestamp for r in recent_results),
            "recommendations": self._generate_summary_recommendations(recent_results)
        }
        
        return summary
    
    def _generate_summary_recommendations(self, results: List[DriftDetectionResult]) -> List[str]:
        """Generate summary recommendations based on drift results."""
        
        recommendations = []
        
        # Count critical and high severity drifts
        critical_drifts = [r for r in results if r.detected and r.severity == "critical"]
        high_drifts = [r for r in results if r.detected and r.severity == "high"]
        
        if critical_drifts:
            models_with_critical = set(r.model_name for r in critical_drifts)
            recommendations.append(f"URGENT: {len(models_with_critical)} models have critical drift - immediate retraining required")
        
        if high_drifts:
            models_with_high = set(r.model_name for r in high_drifts)
            recommendations.append(f"WARNING: {len(models_with_high)} models have high drift - schedule retraining within 24 hours")
        
        # Performance-specific recommendations
        perf_drifts = [r for r in results if r.drift_type == DriftType.PERFORMANCE_DRIFT and r.detected]
        if len(perf_drifts) > len(results) * 0.5:  # More than 50% performance drift
            recommendations.append("High rate of performance drift detected - review infrastructure and model deployment")
        
        if not recommendations:
            recommendations.append("No immediate action required - continue monitoring")
        
        return recommendations


async def main():
    """Main drift monitoring function."""
    
    # Configure logging
    configure_logging("drift_monitor", "INFO", "json")
    
    # Create drift monitor
    config = BaseConfig()
    monitor = DriftMonitor(config)
    
    # Set up reference distributions (this would use real data)
    reference_data = np.random.normal(0, 1, 1000)
    monitor.data_drift_detector.set_reference_distribution(
        model_name="curve_forecaster",
        feature_name="rate_features",
        reference_data=reference_data
    )
    
    # Start monitoring
    models_to_monitor = ["curve_forecaster", "embedding_model"]
    await monitor.start_monitoring(models_to_monitor)


if __name__ == "__main__":
    asyncio.run(main())
