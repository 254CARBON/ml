"""Tests for vector store functionality."""

import pytest
import numpy as np
from libs.vector_store.pgvector import PgVectorStore
from libs.vector_store.factory import create_vector_store_from_env


@pytest.mark.asyncio
async def test_pgvector_store_operations():
    """Test basic PgVector store operations."""
    # This would require a test database setup
    # For now, just test the interface
    
    # Mock configuration
    env_config = {
        "ML_VECTOR_DB_DSN": "postgresql://test:test@localhost:5432/test",
        "ML_VECTOR_POOL_SIZE": "5",
        "ML_VECTOR_MAX_QUERIES": "1000",
        "ML_VECTOR_COMMAND_TIMEOUT": "30"
    }
    
    # Test store creation
    store = create_vector_store_from_env(env_config)
    assert isinstance(store, PgVectorStore)
    
    # Test health check (would fail without real DB)
    # health = await store.health_check()
    # assert isinstance(health, bool)


@pytest.mark.asyncio
async def test_vector_similarity():
    """Test vector similarity calculations."""
    # Test cosine similarity calculation
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    vec3 = np.array([1.0, 0.0, 0.0])
    
    # Calculate cosine similarity
    def cosine_similarity(a, b):
        """Compute cosine similarity between two 1-D numpy arrays."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_orthogonal = cosine_similarity(vec1, vec2)
    sim_identical = cosine_similarity(vec1, vec3)
    
    assert abs(sim_orthogonal) < 0.1  # Should be close to 0
    assert abs(sim_identical - 1.0) < 0.1  # Should be close to 1


def test_vector_dimensions():
    """Test vector dimension handling."""
    # Test different vector dimensions
    dims = [128, 256, 384, 512]
    
    for dim in dims:
        vec = np.random.rand(dim)
        assert len(vec) == dim
        assert vec.shape == (dim,)
