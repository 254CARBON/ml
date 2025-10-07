"""Vector store adapters and utilities.

Primary components:
- ``base``: abstract ``VectorStore`` interface and common exceptions.
- ``pgvector``: PostgreSQL/pgvector implementation of the interface.
- ``factory``: helpers to construct a store from typed config or env.

Guidance:
- Prefer constructing via ``factory.create_vector_store_from_env`` so runtime
  services remain decoupled from specific backends.
"""
