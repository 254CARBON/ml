"""Metrics collection facade for the embedding service.

This module re-exports shared metrics utilities so the rest of the service
can import from a stable local path (``app.runtime.metrics``) without knowing
about the underlying shared library layout.

Key APIs:
- ``get_metrics_collector(service_name)``: return the process-wide collector.
- ``MetricsCollector``: record request, inference, and cache metrics.
"""

from libs.common.metrics import MetricsCollector, get_metrics_collector
