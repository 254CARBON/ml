"""Configuration management for ML platform services.

This module centralizes environment-driven configuration for all services in
the platform (model serving, embeddings, search, indexer, MLflow). It builds
on ``pydantic.BaseSettings`` so configuration can be provided via environment
variables, ``.env`` files, or defaults.

Highlights
- Strongly‑typed settings with sensible defaults
- One place to discover commonly used environment variables
- Small service‑specific subclasses to keep concerns clear

Usage
- Inject the appropriate config in your service entrypoint:
  ``config = ModelServingConfig()``
- Or select dynamically: ``config = get_config("search")``
"""

import os
from typing import Any, Dict, Optional

try:  # Prefer pydantic v2 style imports with backwards compatibility.
    from pydantic import Field  # type: ignore
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for slim test envs
    class BaseSettings:  # type: ignore
        """Minimal BaseSettings fallback when pydantic is unavailable."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        class Config:  # noqa: D401
            env_file = ".env"
            case_sensitive = False

    def Field(default=None, **kwargs):  # type: ignore
        return default


class BaseConfig(BaseSettings):
    """Base configuration class for all services.

    Parameters are read from the process environment with the given names.
    Defaults keep local development convenient while still being explicit.

    Notes
    - Add new shared settings here so downstream services inherit them.
    - Prefer ``Field(..., env="NAME")`` over reading ``os.environ`` directly.
    """
    
    if "SettingsConfigDict" in globals():  # pragma: no branch
        model_config = SettingsConfigDict(
            env_file=".env",
            case_sensitive=False,
        )
    
    # Environment
    ml_env: str = Field(default="local", env="ML_ENV")
    
    # Database
    ml_vector_db_dsn: str = Field(default="postgresql://mlflow:mlflow_password@localhost:5432/mlflow", env="ML_VECTOR_DB_DSN")
    ml_redis_url: str = Field(default="redis://localhost:6379", env="ML_REDIS_URL")
    
    # MLflow
    ml_mlflow_tracking_uri: str = Field(default="file:///tmp/254carbon/mlruns", env="ML_MLFLOW_TRACKING_URI")
    ml_mlflow_backend_dsn: str = Field(default="postgresql://mlflow:mlflow_password@localhost:5432/mlflow", env="ML_MLFLOW_BACKEND_DSN")
    ml_mlflow_artifact_uri: str = Field(default="file:///tmp/254carbon/mlruns/artifacts", env="ML_MLFLOW_ARTIFACT_URI")
    
    # MinIO
    ml_minio_endpoint: str = Field(default="http://localhost:9000", env="ML_MINIO_ENDPOINT")
    ml_minio_access_key: str = Field(default="minioadmin", env="ML_MINIO_ACCESS_KEY")
    ml_minio_secret_key: str = Field(default="minioadmin123", env="ML_MINIO_SECRET_KEY")
    
    # Observability
    ml_tracing_enabled: bool = Field(default=True, env="ML_TRACING_ENABLED")
    ml_otel_exporter: str = Field(default="http://localhost:14268/api/traces", env="ML_OTEL_EXPORTER")
    ml_otel_service_name: str = Field(default="ml-platform", env="ML_OTEL_SERVICE_NAME")
    
    # Logging
    ml_log_level: str = Field(default="INFO", env="ML_LOG_LEVEL")
    ml_log_format: str = Field(default="json", env="ML_LOG_FORMAT")
    
    # Performance
    ml_gpu_preference: str = Field(default="auto", env="ML_GPU_PREFERENCE")
    ml_max_batch_size: int = Field(default=256, env="ML_MAX_BATCH_SIZE")
    
    # Vector store
    ml_vector_dimension: int = Field(default=384, env="ML_VECTOR_DIMENSION")
    ml_vector_backend: str = Field(default="pgvector", env="ML_VECTOR_BACKEND")
    ml_vector_dual_write: bool = Field(default=False, env="ML_VECTOR_DUAL_WRITE")
    ml_vector_read_from_secondary: bool = Field(default=False, env="ML_VECTOR_READ_FROM_SECONDARY")
    
    # OpenSearch
    ml_opensearch_hosts: str = Field(default="http://localhost:9200", env="ML_OPENSEARCH_HOSTS")
    ml_opensearch_index: str = Field(default="ml_embeddings", env="ML_OPENSEARCH_INDEX")
    ml_opensearch_username: Optional[str] = Field(default=None, env="ML_OPENSEARCH_USERNAME")
    ml_opensearch_password: Optional[str] = Field(default=None, env="ML_OPENSEARCH_PASSWORD")
    ml_opensearch_verify_certs: bool = Field(default=False, env="ML_OPENSEARCH_VERIFY_CERTS")
    ml_opensearch_ssl_assert_hostname: bool = Field(default=False, env="ML_OPENSEARCH_SSL_ASSERT_HOSTNAME")
    ml_opensearch_ssl_show_warn: bool = Field(default=False, env="ML_OPENSEARCH_SSL_SHOW_WARN")
    
    # Security
    ml_jwt_secret_key: str = Field(default="dev-secret-key-change-in-production", env="ML_JWT_SECRET_KEY")
    ml_jwt_algorithm: str = Field(default="HS256", env="ML_JWT_ALGORITHM")
    ml_jwt_access_token_expire_minutes: int = Field(default=30, env="ML_JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    
    if "SettingsConfigDict" not in globals():  # pragma: no cover - pydantic v1 path
        class Config:
            env_file = ".env"
            case_sensitive = False


class ModelServingConfig(BaseConfig):
    """Configuration for model serving service.

    Extends ``BaseConfig`` with runtime ports and model defaults used by the
    model‑serving API.
    """
    
    ml_model_serving_port: int = Field(default=9005, env="ML_MODEL_SERVING_PORT")
    ml_model_default_name: str = Field(default="curve_forecaster", env="ML_MODEL_DEFAULT_NAME")


class EmbeddingConfig(BaseConfig):
    """Configuration for embedding service.

    Includes embedding model selection and the service’s public URL used by
    other services (e.g., search manager, indexer worker).
    """
    
    ml_embedding_port: int = Field(default=9006, env="ML_EMBEDDING_PORT")
    ml_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="ML_EMBEDDING_MODEL")
    ml_embedding_service_url: str = Field(default="http://localhost:9006", env="ML_EMBEDDING_SERVICE_URL")


class SearchConfig(BaseConfig):
    """Configuration for search service.

    Adds search API port and the service URL consumed by clients.
    """
    
    ml_search_port: int = Field(default=9007, env="ML_SEARCH_PORT")
    ml_search_service_url: str = Field(default="http://localhost:9007", env="ML_SEARCH_SERVICE_URL")


class MLflowConfig(BaseConfig):
    """Configuration for MLflow server.

    Captures only the parameters specific to the MLflow deployment.
    """
    
    ml_mlflow_port: int = Field(default=5000, env="ML_MLFLOW_PORT")


class IndexerConfig(BaseConfig):
    """Configuration for indexer worker.

    Keeps queue and ingestion‑related knobs together.
    """
    
    ml_ingest_queue: str = Field(default="embeddings_rebuild", env="ML_INGEST_QUEUE")


def get_config(service_name: str) -> BaseConfig:
    """Get configuration for a specific service.

    Parameters
    - service_name: Literal name: ``model-serving``, ``embedding``, ``search``,
      ``mlflow``, or ``indexer``.

    Returns
    - A concrete ``BaseConfig`` subclass pre‑wired to read the right env vars.
    """
    config_map = {
        "model-serving": ModelServingConfig,
        "embedding": EmbeddingConfig,
        "search": SearchConfig,
        "mlflow": MLflowConfig,
        "indexer": IndexerConfig,
    }
    
    # Default to ``BaseConfig`` to avoid surprising crashes for unknown names.
    config_class = config_map.get(service_name, BaseConfig)
    return config_class()


def load_env_file(env_file: str = ".env") -> Dict[str, Any]:
    """Load environment variables from a file.

    This helper parses a simple ``KEY=VALUE`` file, ignoring blank lines and
    comments. It does not modify the process environment; callers can decide
    whether to merge or just inspect values.

    Parameters
    - env_file: Path to a dotenv‑style file (default: ``.env``)

    Returns
    - Dict of parsed key/value pairs in file order.
    """
    env_vars = {}
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
    return env_vars
