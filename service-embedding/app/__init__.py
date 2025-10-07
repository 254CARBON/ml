"""Embedding service package.

Layout:
- ``api``: FastAPI route handlers and request/response models.
- ``encoders``: ``EmbeddingManager`` that loads models and generates vectors.
- ``runtime``: service-local metrics and runtime helpers.

Import convenience:
- from app.encoders.embedding_manager import EmbeddingManager
"""
