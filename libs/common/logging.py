"""Structured logging configuration for ML platform services.

This module standardizes logging across services using ``structlog``. It
produces either JSON (for machines) or a pretty console format (for humans)
and binds consistent service context so logs are useful when aggregated.

Typical usage
- Call ``configure_logging(service_name, log_level, log_format)`` at startup
- Acquire loggers via ``structlog.get_logger(name)`` or ``ServiceLogger``
"""

import logging
import sys
from typing import Any, Dict, Optional
import structlog
from structlog.stdlib import LoggerFactory, add_logger_name


def configure_logging(
    service_name: str,
    log_level: str = "INFO",
    log_format: str = "json",
    **kwargs: Any
) -> None:
    """Configure structured logging for a service.

    Parameters
    - service_name: Logical service identifier bound to each log line
    - log_level: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` (case‑insensitive)
    - log_format: ``json`` for production; ``console`` for local dev
    - kwargs: Reserved for future custom processors/overrides
    """
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_logger_name,
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add service context
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class ServiceLogger:
    """Service-specific logger with common context."""
    
    def __init__(self, service_name: str, **context: Any):
        self.service_name = service_name
        self.logger = structlog.get_logger(service_name)
        self.context = context
    
    def bind(self, **kwargs: Any) -> "ServiceLogger":
        """Bind additional context to the logger.

        Returns a new ``ServiceLogger`` carrying the merged context so the
        original instance stays unchanged (chainable style).
        """
        return ServiceLogger(self.service_name, **{**self.context, **kwargs})
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self.logger.info(message, **{**self.context, **kwargs})
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self.logger.warning(message, **{**self.context, **kwargs})
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self.logger.error(message, **{**self.context, **kwargs})
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **{**self.context, **kwargs})
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, **{**self.context, **kwargs})


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """Log a function call with parameters.

    Useful for lightweight auditing or ad‑hoc tracing when full distributed
    tracing is not warranted.
    """
    logger = get_logger("function_calls")
    logger.info(f"Calling {func_name}", function=func_name, **kwargs)


def log_performance(operation: str, duration_ms: float, **kwargs: Any) -> None:
    """Log performance metrics.

    Parameters
    - operation: A stable identifier for the measured unit of work
    - duration_ms: Elapsed time in milliseconds
    - kwargs: Additional dimensions (e.g., model name, status)
    """
    logger = get_logger("performance")
    logger.info(
        f"Operation {operation} completed",
        operation=operation,
        duration_ms=duration_ms,
        **kwargs
    )
