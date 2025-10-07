"""Common utilities shared across services.

Includes:
- ``config``: Pydantic-based service configuration from environment variables.
- ``logging``: structured logging setup with structlog.
- ``metrics``: Prometheus metrics helpers and decorators.
- ``events``: Redis pub/sub event models, publisher, and subscriber.
- ``auth``: JWT helpers and simple tenant context utilities.

Import pattern:
- from libs.common.config import BaseConfig
- from libs.common.logging import configure_logging
"""
