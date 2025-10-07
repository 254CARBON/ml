"""Metrics collection facade for the model serving service.

Re-exports the shared metrics helpers under a local path to simplify
imports across the service codebase.
"""

from libs.common.metrics import MetricsCollector, get_metrics_collector
