"""Tests for common utilities."""

import pytest
from libs.common.config import BaseConfig, ModelServingConfig, EmbeddingConfig
from libs.common.logging import configure_logging
from libs.common.metrics import MetricsCollector
from libs.common.events import EventPublisher, EventSubscriber


def test_config_loading():
    """Test configuration loading."""
    config = BaseConfig()
    assert config.ml_env == "local"
    assert config.ml_log_level == "INFO"
    assert config.ml_gpu_preference == "auto"


def test_model_serving_config():
    """Test model serving configuration."""
    config = ModelServingConfig()
    assert config.ml_model_serving_port == 9005
    assert config.ml_model_default_name == "curve_forecaster"


def test_embedding_config():
    """Test embedding configuration."""
    config = EmbeddingConfig()
    assert config.ml_embedding_port == 9006
    assert config.ml_embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_logging_configuration():
    """Test logging configuration."""
    # This should not raise an exception
    configure_logging("test-service", "INFO", "json")


def test_metrics_collector():
    """Test metrics collector."""
    collector = MetricsCollector("test-service")
    assert collector.service_name == "test-service"
    
    # Test metrics recording
    collector.record_http_request("GET", "/test", 200, 0.1)
    collector.record_inference("test-model", "1.0", 0.05)
    
    # Test metrics retrieval
    metrics = collector.get_metrics()
    assert isinstance(metrics, str)
    assert "http_requests_total" in metrics


def test_event_publisher():
    """Test event publisher."""
    publisher = EventPublisher("redis://localhost:6379")
    assert publisher.channel_prefix == "ml_events"
    
    # Test event creation (without publishing)
    from libs.common.events import ModelPromotedEvent
    event = ModelPromotedEvent(
        timestamp=1234567890,
        model_name="test-model",
        version="1.0",
        stage="Production",
        run_id="test-run"
    )
    
    assert event.model_name == "test-model"
    assert event.version == "1.0"
    assert event.stage == "Production"


def test_event_subscriber():
    """Test event subscriber."""
    subscriber = EventSubscriber("redis://localhost:6379")
    assert subscriber.channel_prefix == "ml_events"
    assert isinstance(subscriber.handlers, dict)
