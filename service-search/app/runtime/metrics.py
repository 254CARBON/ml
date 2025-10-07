"""Metrics collection facade for the search service.

Re-exports shared metrics helpers so callers can import from a consistent
local path within the service.
"""

from libs.common.metrics import MetricsCollector, get_metrics_collector
