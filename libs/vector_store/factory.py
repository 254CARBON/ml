"""Vector store factory for creating different implementations.

Centralizes creation of concrete ``VectorStore`` backends so callers don't
depend on implementation details. New stores can be added without changing
call sites.
"""

from typing import Any, Dict, Optional
from enum import Enum
import structlog

from .base import VectorStore
from .pgvector import PgVectorStore
from .opensearch import OpenSearchVectorStore
from .dual_write import create_dual_write_store

logger = structlog.get_logger("vector_store.factory")


class VectorStoreType(Enum):
    """Supported vector store types."""
    PGVECTOR = "pgvector"
    OPENSEARCH = "opensearch"
    PINEONE = "pinecone"  # Future implementation


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create(
        store_type: VectorStoreType,
        config: Dict[str, Any],
        **kwargs: Any
    ) -> VectorStore:
        """Create a vector store instance.

        Parameters
        - store_type: A ``VectorStoreType`` enum value
        - config: Backend‑specific parameters (e.g., DSN for pgvector)
        - kwargs: Additional optional overrides forwarded to implementation
        """
        
        if store_type == VectorStoreType.PGVECTOR:
            kwargs = dict(kwargs)
            dsn = config.get("dsn")
            if not dsn:
                raise ValueError("PgVector requires 'dsn' in config")
            
            vector_dimension = kwargs.pop(
                "vector_dimension",
                config.get("vector_dimension")
            )
            
            return PgVectorStore(
                dsn=dsn,
                pool_size=config.get("pool_size", 10),
                max_queries=config.get("max_queries", 50000),
                command_timeout=config.get("command_timeout", 60),
                vector_dimension=vector_dimension,
                **kwargs
            )
        
        elif store_type == VectorStoreType.OPENSEARCH:
            hosts = config.get("hosts", ["http://localhost:9200"])
            if not hosts:
                raise ValueError("OpenSearch requires 'hosts' in config")
            
            return OpenSearchVectorStore(
                hosts=hosts,
                index_name=config.get("index_name", "ml_embeddings"),
                vector_dimension=config.get("vector_dimension", 384),
                username=config.get("username"),
                password=config.get("password"),
                verify_certs=config.get("verify_certs", False),
                ssl_assert_hostname=config.get("ssl_assert_hostname", False),
                ssl_show_warn=config.get("ssl_show_warn", False)
            )
        
        elif store_type == VectorStoreType.PINEONE:
            # Future implementation
            raise NotImplementedError("Pinecone vector store not yet implemented")
        
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> VectorStore:
        """Create vector store from configuration dictionary.

        Expects a ``type`` key and any implementation‑specific fields.
        """
        store_type_str = config.get("type", "pgvector")
        
        try:
            store_type = VectorStoreType(store_type_str)
        except ValueError:
            raise ValueError(f"Unsupported vector store type: {store_type_str}")
        
        return VectorStoreFactory.create(store_type, config)


def create_vector_store(
    store_type: str,
    config: Dict[str, Any],
    **kwargs: Any
) -> VectorStore:
    """Convenience function to create a vector store."""
    try:
        store_type_enum = VectorStoreType(store_type)
        return VectorStoreFactory.create(store_type_enum, config, **kwargs)
    except ValueError:
        raise ValueError(f"Unsupported vector store type: {store_type}")


def create_vector_store_from_env(env_config: Dict[str, str]) -> VectorStore:
    """Create vector store from environment configuration.

    Parameters
    - env_config: A flat mapping of environment variable names to values

    Returns
    - A ``VectorStore`` configured to talk to the backing datastore
    """
    backend = env_config.get("ML_VECTOR_BACKEND", "pgvector")
    dual_write_enabled = env_config.get("ML_VECTOR_DUAL_WRITE", "false").lower() == "true"
    
    if backend == "pgvector":
        primary_config = {
            "type": "pgvector",
            "dsn": env_config.get("ML_VECTOR_DB_DSN"),
            "pool_size": int(env_config.get("ML_VECTOR_POOL_SIZE", "10")),
            "max_queries": int(env_config.get("ML_VECTOR_MAX_QUERIES", "50000")),
            "command_timeout": int(env_config.get("ML_VECTOR_COMMAND_TIMEOUT", "60")),
            "vector_dimension": int(env_config.get("ML_VECTOR_DIMENSION", "384")),
        }
        
        if not primary_config["dsn"]:
            raise ValueError("ML_VECTOR_DB_DSN environment variable is required")
        
        primary_store = VectorStoreFactory.create_from_config(primary_config)
        
        # Check if dual-write to OpenSearch is enabled
        if dual_write_enabled:
            try:
                secondary_config = {
                    "type": "opensearch",
                    "hosts": env_config.get("ML_OPENSEARCH_HOSTS", "http://localhost:9200").split(","),
                    "username": env_config.get("ML_OPENSEARCH_USERNAME"),
                    "password": env_config.get("ML_OPENSEARCH_PASSWORD"),
                    "verify_certs": env_config.get("ML_OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
                    "index_name": env_config.get("ML_OPENSEARCH_INDEX", "ml_embeddings"),
                    "vector_dimension": int(env_config.get("ML_VECTOR_DIMENSION", "384")),
                    "ssl_assert_hostname": env_config.get("ML_OPENSEARCH_SSL_ASSERT_HOSTNAME", "false").lower() == "true",
                    "ssl_show_warn": env_config.get("ML_OPENSEARCH_SSL_SHOW_WARN", "false").lower() == "true",
                }
                
                secondary_store = VectorStoreFactory.create_from_config(secondary_config)
                
                # Create dual-write store
                read_from_secondary = env_config.get("ML_VECTOR_READ_FROM_SECONDARY", "false").lower() == "true"
                return create_dual_write_store(
                    primary_store=primary_store,
                    secondary_store=secondary_store,
                    read_from_secondary=read_from_secondary,
                    write_to_secondary=True
                )
                
            except Exception as e:
                logger.warning("Failed to initialize secondary store, using primary only", error=str(e))
                return primary_store
        
        return primary_store
    
    elif backend == "opensearch":
        config = {
            "type": "opensearch",
            "hosts": env_config.get("ML_OPENSEARCH_HOSTS", "http://localhost:9200").split(","),
            "username": env_config.get("ML_OPENSEARCH_USERNAME"),
            "password": env_config.get("ML_OPENSEARCH_PASSWORD"),
            "verify_certs": env_config.get("ML_OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
            "index_name": env_config.get("ML_OPENSEARCH_INDEX", "ml_embeddings"),
            "vector_dimension": int(env_config.get("ML_VECTOR_DIMENSION", "384")),
            "ssl_assert_hostname": env_config.get("ML_OPENSEARCH_SSL_ASSERT_HOSTNAME", "false").lower() == "true",
            "ssl_show_warn": env_config.get("ML_OPENSEARCH_SSL_SHOW_WARN", "false").lower() == "true",
        }
        
        return VectorStoreFactory.create_from_config(config)
    
    else:
        raise ValueError(f"Unsupported vector backend: {backend}")
