"""Metrics collection facade for the indexer worker.

Re-exports the shared metrics helpers. The worker records counts and durations
for background tasks such as reindexing and batch embedding.
"""

from libs.common.metrics import MetricsCollector, get_metrics_collector
