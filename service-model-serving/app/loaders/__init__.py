"""Model loaders for different frameworks and runtimes.

Loaders encapsulate how models are materialized (from disk, MLflow, object
storage) and prepared for inference (device placement, warmup, versioning).

Goals
- Provide a consistent interface regardless of the framework/backend
- Support multiple model versions and hot reloads
"""
