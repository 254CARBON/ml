"""Shared libraries for the ML Platform.

Subpackages:
- ``libs.common``: configuration, logging, authentication, metrics, and events.
- ``libs.vector_store``: vector store abstractions and concrete backends.

Usage:
- Import stable, reusable functionality from here to keep service code lean.

Notes:
- Avoid service-specific logic; keep modules cohesive and broadly useful.
"""
