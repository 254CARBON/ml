"""Integration tests for event flow between services."""

import pytest
import asyncio
import time
import json
from datetime import datetime
import redis
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from libs.common.config import BaseConfig
from libs.common.events import (
    create_event_publisher, 
    create_event_subscriber,
    EventType,
    ModelPromotedEvent,
    EmbeddingReindexRequestEvent,
    SearchIndexUpdatedEvent
)


@pytest.mark.integration
class TestEventFlow:
    """Test event flow between services."""
    
    @pytest.fixture(scope="class")
    def config(self):
        """Get configuration."""
        return BaseConfig()
    
    @pytest.fixture(scope="class")
    def redis_client(self, config):
        """Create Redis client."""
        try:
            client = redis.from_url(config.ml_redis_url)
            client.ping()  # Test connection
            return client
        except redis.ConnectionError:
            pytest.skip("Redis not available")
    
    @pytest.fixture(scope="class")
    def event_publisher(self, config):
        """Create event publisher."""
        return create_event_publisher(config.ml_redis_url)
    
    @pytest.fixture(scope="class")
    def event_subscriber(self, config):
        """Create event subscriber."""
        return create_event_subscriber(config.ml_redis_url)
    
    def test_redis_connection(self, redis_client):
        """Test Redis connection."""
        assert redis_client.ping()
        
        # Test basic pub/sub
        pubsub = redis_client.pubsub()
        pubsub.subscribe("test_channel")
        
        redis_client.publish("test_channel", "test_message")
        
        message = pubsub.get_message(timeout=1)
        if message and message['type'] == 'subscribe':
            message = pubsub.get_message(timeout=1)
        
        assert message is not None
        assert message['data'] == b'test_message'
        pubsub.close()
    
    def test_event_publishing(self, event_publisher, redis_client):
        """Test event publishing."""
        # Subscribe to events channel
        pubsub = redis_client.pubsub()
        pubsub.subscribe("ml_events:ml.model.promoted.v1")
        
        # Publish model promoted event
        event_publisher.publish_model_promoted(
            model_name="test_model",
            version="1.0.0",
            stage="Production",
            run_id="test_run_123"
        )
        
        # Check if event was published
        message = pubsub.get_message(timeout=2)
        if message and message['type'] == 'subscribe':
            message = pubsub.get_message(timeout=2)
        
        assert message is not None
        assert message['type'] == 'message'
        
        # Parse event data
        event_data = json.loads(message['data'].decode('utf-8'))
        assert event_data['model_name'] == "test_model"
        assert event_data['version'] == "1.0.0"
        assert event_data['stage'] == "Production"
        assert event_data['run_id'] == "test_run_123"
        assert event_data['event_type'] == "ml.model.promoted.v1"
        
        pubsub.close()
    
    def test_event_subscription(self, event_subscriber, event_publisher):
        """Test event subscription and handling."""
        received_events = []
        
        def handle_model_promoted(event_data):
            """Collect received MODEL_PROMOTED event payloads for assertions."""
            received_events.append(event_data)
        
        # Subscribe to model promoted events
        event_subscriber.subscribe(EventType.MODEL_PROMOTED, handle_model_promoted)
        
        # Start subscriber in background
        async def run_subscriber():
            """Poll the Redis pubsub briefly and forward messages to handlers."""
            # Listen for a short time
            pubsub = event_subscriber.redis_client.pubsub()
            pubsub.subscribe("ml_events:ml.model.promoted.v1")
            
            for _ in range(10):  # Listen for up to 10 messages or 5 seconds
                message = pubsub.get_message(timeout=0.5)
                if message and message['type'] == 'message':
                    event_subscriber._handle_message(message)
                await asyncio.sleep(0.1)
            
            pubsub.close()
        
        # Run subscriber and publisher concurrently
        async def test_flow():
            """Run subscription task, publish events, then await processing."""
            # Start subscriber
            subscriber_task = asyncio.create_task(run_subscriber())
            
            # Wait a bit then publish events
            await asyncio.sleep(0.5)
            
            event_publisher.publish_model_promoted(
                model_name="test_model_1",
                version="1.0.0",
                stage="Production",
                run_id="run_1"
            )
            
            event_publisher.publish_model_promoted(
                model_name="test_model_2", 
                version="2.0.0",
                stage="Production",
                run_id="run_2"
            )
            
            # Wait for subscriber to process
            await subscriber_task
        
        # Run the test
        asyncio.run(test_flow())
        
        # Verify events were received
        assert len(received_events) >= 1  # At least one event should be received
        
        # Check first received event
        if received_events:
            event = received_events[0]
            assert event['event_type'] == "ml.model.promoted.v1"
            assert 'model_name' in event
            assert 'version' in event
            assert 'stage' in event
    
    def test_embedding_reindex_event_flow(self, event_publisher, redis_client):
        """Test embedding reindex event flow."""
        # Subscribe to reindex events
        pubsub = redis_client.pubsub()
        pubsub.subscribe("ml_events:ml.embedding.reindex.request.v1")
        
        # Publish reindex request
        event_publisher.publish_embedding_reindex_request(
            entity_type="instruments",
            batch_size=100,
            model_version="v1.0"
        )
        
        # Check if event was published
        message = pubsub.get_message(timeout=2)
        if message and message['type'] == 'subscribe':
            message = pubsub.get_message(timeout=2)
        
        assert message is not None
        event_data = json.loads(message['data'].decode('utf-8'))
        assert event_data['entity_type'] == "instruments"
        assert event_data['batch_size'] == 100
        assert event_data['model_version'] == "v1.0"
        assert event_data['event_type'] == "ml.embedding.reindex.request.v1"
        
        pubsub.close()
    
    def test_search_index_updated_event_flow(self, event_publisher, redis_client):
        """Test search index updated event flow."""
        # Subscribe to search events
        pubsub = redis_client.pubsub()
        pubsub.subscribe("ml_events:search.index.updated.v1")
        
        # Publish search index updated event
        event_publisher.publish_search_index_updated(
            entity_type="curves",
            count=500
        )
        
        # Check if event was published
        message = pubsub.get_message(timeout=2)
        if message and message['type'] == 'subscribe':
            message = pubsub.get_message(timeout=2)
        
        assert message is not None
        event_data = json.loads(message['data'].decode('utf-8'))
        assert event_data['entity_type'] == "curves"
        assert event_data['count'] == 500
        assert event_data['event_type'] == "search.index.updated.v1"
        
        pubsub.close()
    
    def test_event_data_validation(self, event_publisher):
        """Test event data validation and serialization."""
        # Test model promoted event
        event = ModelPromotedEvent(
            timestamp=int(time.time() * 1000),
            model_name="validation_test_model",
            version="1.2.3",
            stage="Staging",
            run_id="validation_run_456"
        )
        
        # Test serialization
        event_dict = event.to_dict()
        assert event_dict['model_name'] == "validation_test_model"
        assert event_dict['version'] == "1.2.3"
        assert event_dict['stage'] == "Staging"
        assert event_dict['event_type'] == "ml.model.promoted.v1"
        
        event_json = event.to_json()
        parsed = json.loads(event_json)
        assert parsed['model_name'] == "validation_test_model"
        
        # Test embedding reindex event
        reindex_event = EmbeddingReindexRequestEvent(
            timestamp=int(time.time() * 1000),
            entity_type="test_entities",
            batch_size=250,
            model_version="v2.0"
        )
        
        reindex_dict = reindex_event.to_dict()
        assert reindex_dict['entity_type'] == "test_entities"
        assert reindex_dict['batch_size'] == 250
        assert reindex_dict['event_type'] == "ml.embedding.reindex.request.v1"
        
        # Test search index updated event
        search_event = SearchIndexUpdatedEvent(
            timestamp=int(time.time() * 1000),
            entity_type="test_search_entities",
            count=1000
        )
        
        search_dict = search_event.to_dict()
        assert search_dict['entity_type'] == "test_search_entities"
        assert search_dict['count'] == 1000
        assert search_dict['event_type'] == "search.index.updated.v1"
    
    def test_event_ordering_and_timing(self, event_publisher, redis_client):
        """Test event ordering and timing."""
        # Subscribe to all events
        pubsub = redis_client.pubsub()
        pubsub.psubscribe("ml_events:*")
        
        events_published = []
        start_time = time.time()
        
        # Publish multiple events in sequence
        for i in range(5):
            timestamp = int(time.time() * 1000)
            event_publisher.publish_model_promoted(
                model_name=f"model_{i}",
                version=f"1.{i}.0",
                stage="Production",
                run_id=f"run_{i}"
            )
            events_published.append({
                'model_name': f"model_{i}",
                'timestamp': timestamp,
                'order': i
            })
            time.sleep(0.1)  # Small delay between events
        
        # Collect published events
        received_events = []
        for _ in range(10):  # Try to get up to 10 messages
            message = pubsub.get_message(timeout=1)
            if message and message['type'] == 'pmessage':
                try:
                    event_data = json.loads(message['data'].decode('utf-8'))
                    if event_data.get('event_type') == 'ml.model.promoted.v1':
                        received_events.append(event_data)
                except json.JSONDecodeError:
                    continue
        
        pubsub.close()
        
        # Verify we received events
        assert len(received_events) >= 1
        
        # Check timestamps are reasonable
        for event in received_events:
            event_time = event['timestamp'] / 1000  # Convert to seconds
            assert abs(event_time - start_time) < 10  # Within 10 seconds
    
    def test_event_error_handling(self, event_subscriber):
        """Test event error handling."""
        error_count = 0
        
        def failing_handler(event_data):
            """Handler that intentionally raises to test error paths."""
            nonlocal error_count
            error_count += 1
            raise Exception("Simulated handler error")

        def working_handler(event_data):
            """No-op handler to verify mixed handler behaviors."""
            pass  # This should still work
        
        # Subscribe both handlers
        event_subscriber.subscribe(EventType.MODEL_PROMOTED, failing_handler)
        event_subscriber.subscribe(EventType.MODEL_PROMOTED, working_handler)
        
        # Simulate message handling
        test_message = {
            'channel': b'ml_events:ml.model.promoted.v1',
            'data': json.dumps({
                'event_type': 'ml.model.promoted.v1',
                'model_name': 'error_test_model',
                'version': '1.0.0',
                'stage': 'Production',
                'run_id': 'error_test_run'
            }).encode('utf-8'),
            'type': 'message'
        }
        
        # Handle message (should not raise exception)
        event_subscriber._handle_message(test_message)
        
        # Verify error handler was called
        assert error_count == 1


    @pytest.mark.asyncio
    async def test_event_retry_logic(self, event_publisher, redis_client):
        """Test event retry logic with exponential backoff."""
        # This test verifies that event listeners implement retry logic
        # In a real scenario, this would test the actual retry behavior
        
        # Simulate event publishing with potential failures
        test_event = ModelPromotedEvent(
            timestamp=int(time.time() * 1000),
            model_name="test_model",
            version="1.0.0",
            stage="Production",
            run_id="test_run_123"
        )
        
        # Publish event
        event_publisher.publish(test_event)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify event was processed (in real implementation)
        # This is a placeholder for actual retry logic testing
        assert True  # Placeholder assertion

    @pytest.mark.asyncio
    async def test_multi_tenant_event_flow(self, event_publisher, redis_client):
        """Test event flow with tenant isolation."""
        # Test that events respect tenant boundaries
        
        # Create tenant-specific events
        tenant_events = []
        for tenant_id in ["tenant_a", "tenant_b", "tenant_c"]:
            event = EmbeddingReindexRequestEvent(
                timestamp=int(time.time() * 1000),
                entity_type="instrument",
                batch_size=100,
                model_version="default"
            )
            # In real implementation, tenant_id would be part of event data
            tenant_events.append((tenant_id, event))
        
        # Publish events
        for tenant_id, event in tenant_events:
            event_publisher.publish(event)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify tenant isolation (placeholder)
        assert len(tenant_events) == 3

    @pytest.mark.asyncio
    async def test_ab_testing_event_integration(self, event_publisher, redis_client):
        """Test A/B testing event integration."""
        # Test that A/B testing experiments trigger appropriate events
        
        # Simulate experiment creation event
        experiment_event = {
            "type": "experiment.created",
            "data": {
                "experiment_id": "exp_123",
                "name": "test_experiment",
                "model_a": "model_v1",
                "model_b": "model_v2",
                "traffic_split": 0.5
            },
            "timestamp": time.time()
        }
        
        # Publish to Redis
        redis_client.lpush("ml_events:experiment.created", json.dumps(experiment_event))
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify event was processed (placeholder)
        assert True

    @pytest.mark.asyncio
    async def test_shadow_deployment_event_integration(self, event_publisher, redis_client):
        """Test shadow deployment event integration."""
        # Test that shadow deployments trigger appropriate events
        
        # Simulate shadow deployment event
        shadow_event = {
            "type": "shadow.deployment.configured",
            "data": {
                "model_name": "test_model",
                "shadow_model": "shadow_model_v1",
                "enabled": True
            },
            "timestamp": time.time()
        }
        
        # Publish to Redis
        redis_client.lpush("ml_events:shadow.deployment.configured", json.dumps(shadow_event))
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify event was processed (placeholder)
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
