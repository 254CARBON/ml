"""Event system for inter-service communication.

This module defines a compact eventing contract across services using Redis
pub/sub. Producers publish JSON payloads on namespaced channels derived from
``EventType``; consumers subscribe and register Python callbacks.

Key concepts
- "EventType" stable identifiers are versioned (``.v1`` suffix)
- ``EventPublisher`` composes channel names as ``{prefix}:{event_type}``
- ``EventSubscriber`` manages a map of event handlers and message dispatch

The goal is to keep event shapes explicit and easy to evolve.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import redis.asyncio as redis_async
import structlog

logger = structlog.get_logger("events")


class EventType(Enum):
    """Event types for the ML platform."""
    MODEL_PROMOTED = "ml.model.promoted.v1"
    INFERENCE_USAGE = "ml.inference.usage.v1"
    EMBEDDING_REINDEX_REQUEST = "ml.embedding.reindex.request.v1"
    SEARCH_INDEX_UPDATED = "search.index.updated.v1"
    EMBEDDING_GENERATED = "ml.embedding.generated.v1"


@dataclass
class BaseEvent:
    """Base event class.

    Child events should set their ``event_type`` in ``__post_init__`` and can
    extend the payload with any additional fields relevant to the domain.
    """
    timestamp: int
    event_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ModelPromotedEvent(BaseEvent):
    """Event emitted when a model is promoted to production."""
    model_name: str
    version: str
    stage: str
    run_id: str
    
    def __post_init__(self):
        self.event_type = EventType.MODEL_PROMOTED.value
        if not self.timestamp:
            self.timestamp = int(time.time() * 1000)


@dataclass
class InferenceUsageEvent(BaseEvent):
    """Event emitted for inference usage tracking."""
    model_name: str
    version: str
    latency_ms: int
    request_size: int
    
    def __post_init__(self):
        self.event_type = EventType.INFERENCE_USAGE.value
        if not self.timestamp:
            self.timestamp = int(time.time() * 1000)


@dataclass
class EmbeddingReindexRequestEvent(BaseEvent):
    """Event emitted to request embedding reindexing."""
    entity_type: str
    batch_size: int
    model_version: str
    
    def __post_init__(self):
        self.event_type = EventType.EMBEDDING_REINDEX_REQUEST.value
        if not self.timestamp:
            self.timestamp = int(time.time() * 1000)


@dataclass
class SearchIndexUpdatedEvent(BaseEvent):
    """Event emitted when search index is updated."""
    entity_type: str
    count: int
    
    def __post_init__(self):
        self.event_type = EventType.SEARCH_INDEX_UPDATED.value
        if not self.timestamp:
            self.timestamp = int(time.time() * 1000)


@dataclass
class EmbeddingGeneratedEvent(BaseEvent):
    """Event emitted when an embedding is generated."""
    entity_type: str
    entity_id: str
    model_version: str
    tenant_id: str = "default"
    
    def __post_init__(self):
        self.event_type = EventType.EMBEDDING_GENERATED.value
        if not self.timestamp:
            self.timestamp = int(time.time() * 1000)


class EventPublisher:
    """Publishes events to Redis.

    Notes
    - Publishing is fire‑and‑forget; failures are logged and re‑raised.
    - Messages are serialized as JSON to keep consumers language‑agnostic.
    """
    
    def __init__(self, redis_url: str, channel_prefix: str = "ml_events"):
        self.redis_client = redis.from_url(redis_url)
        self.channel_prefix = channel_prefix
    
    def publish(self, event: BaseEvent) -> None:
        """Publish an event with retry logic.

        The channel is derived from the event's type to allow subscribers to
        filter efficiently without payload inspection.
        """
        max_retries = 3
        base_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                channel = f"{self.channel_prefix}:{event.event_type}"
                message = event.to_json()
                self.redis_client.publish(channel, message)
                logger.info(
                    "Event published",
                    event_type=event.event_type,
                    channel=channel
                )
                return  # Success, exit retry loop
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "Failed to publish event after all retries",
                        event_type=event.event_type,
                        error=str(e)
                    )
                    raise
                
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    "Event publish failed, retrying",
                    event_type=event.event_type,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e)
                )
                time.sleep(delay)
    
    def publish_model_promoted(
        self,
        model_name: str,
        version: str,
        stage: str,
        run_id: str
    ) -> None:
        """Publish model promoted event."""
        event = ModelPromotedEvent(
            timestamp=int(time.time() * 1000),
            model_name=model_name,
            version=version,
            stage=stage,
            run_id=run_id
        )
        self.publish(event)
    
    def publish_inference_usage(
        self,
        model_name: str,
        version: str,
        latency_ms: int,
        request_size: int
    ) -> None:
        """Publish inference usage event."""
        event = InferenceUsageEvent(
            timestamp=int(time.time() * 1000),
            model_name=model_name,
            version=version,
            latency_ms=latency_ms,
            request_size=request_size
        )
        self.publish(event)
    
    def publish_embedding_reindex_request(
        self,
        entity_type: str,
        batch_size: int,
        model_version: str
    ) -> None:
        """Publish embedding reindex request event."""
        event = EmbeddingReindexRequestEvent(
            timestamp=int(time.time() * 1000),
            entity_type=entity_type,
            batch_size=batch_size,
            model_version=model_version
        )
        self.publish(event)
    
    def publish_search_index_updated(
        self,
        entity_type: str,
        count: int
    ) -> None:
        """Publish search index updated event."""
        event = SearchIndexUpdatedEvent(
            timestamp=int(time.time() * 1000),
            entity_type=entity_type,
            count=count
        )
        self.publish(event)
    
    def publish_embedding_generated(
        self,
        entity_type: str,
        entity_id: str,
        model_version: str,
        tenant_id: str = "default"
    ) -> None:
        """Publish embedding generated event."""
        event = EmbeddingGeneratedEvent(
            timestamp=int(time.time() * 1000),
            entity_type=entity_type,
            entity_id=entity_id,
            model_version=model_version,
            tenant_id=tenant_id
        )
        self.publish(event)


class EventSubscriber:
    """Subscribes to events from Redis.

    Maintains a mapping of ``event_type -> List[callables]``. When a message
    arrives, ``_handle_message`` decodes JSON and invokes each registered
    handler with the raw dictionary payload.
    """
    
    def __init__(self, redis_url: str, channel_prefix: str = "ml_events"):
        self.redis_client = redis_async.from_url(redis_url, decode_responses=False)
        self.channel_prefix = channel_prefix
        self.handlers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to an event type."""
        if event_type.value not in self.handlers:
            self.handlers[event_type.value] = []
        self.handlers[event_type.value].append(handler)
        logger.info(
            "Subscribed to event",
            event_type=event_type.value,
            handler=handler.__name__
        )
    
    async def start_listening(self) -> None:
        """Start listening for events asynchronously.

        This is a non-blocking async loop that can be run as a background task.
        Includes retry logic and graceful shutdown handling.
        """
        pubsub = self.redis_client.pubsub()
        
        try:
            # Subscribe to all event channels asynchronously
            channels = [
                f"{self.channel_prefix}:{event_type.value}"
                for event_type in EventType
            ]
            await pubsub.subscribe(*channels)
            
            logger.info("Started listening for events")
            
            # Non-blocking message processing
            while True:
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0
                    )
                    if message and message.get("type") == "message":
                        self._handle_message(message)
                    
                    # Yield control to the event loop
                    await asyncio.sleep(0.01)
                    
                except asyncio.CancelledError:
                    logger.info("Event listener cancelled")
                    raise
                except Exception as e:
                    logger.error("Error in event listener loop", error=str(e))
                    # Continue listening despite transient errors
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            logger.info("Event listener cancelled")
            raise
        except Exception as e:
            logger.error("Event listener failed", error=str(e))
            raise
        finally:
            try:
                await pubsub.close()
                logger.info("Event listener stopped")
            except Exception as e:
                logger.warning("Error closing pubsub", error=str(e))
    
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming event message.

        Dispatch errors from individual handlers are logged and do not prevent
        other handlers from executing.
        """
        try:
            channel_raw = message.get('channel')
            data_raw = message.get('data')
            
            if isinstance(channel_raw, (bytes, bytearray)):
                channel = channel_raw.decode('utf-8')
            else:
                channel = str(channel_raw)
            
            if isinstance(data_raw, (bytes, bytearray)):
                payload = json.loads(data_raw.decode('utf-8'))
            else:
                payload = json.loads(data_raw)
            
            # Extract event type from channel
            event_type = channel.split(':')[-1]
            
            if event_type == EventType.EMBEDDING_GENERATED.value and "tenant_id" not in payload:
                # Backward compatibility: older publishers may omit tenant_id
                payload["tenant_id"] = "default"
            
            if event_type in self.handlers:
                for handler in self.handlers[event_type]:
                    try:
                        handler(payload)
                    except Exception as e:
                        logger.error(
                            "Error handling event",
                            event_type=event_type,
                            handler=handler.__name__,
                            error=str(e)
                        )
            else:
                logger.warning(
                    "No handlers for event type",
                    event_type=event_type
                )
                
        except Exception as e:
            logger.error(
                "Error processing event message",
                error=str(e)
            )

    async def close(self) -> None:
        """Close the Redis client used by the subscriber."""
        try:
            await self.redis_client.aclose()
        except Exception as e:
            logger.warning("Error closing redis client", error=str(e))


def create_event_publisher(redis_url: str) -> EventPublisher:
    """Create an event publisher."""
    return EventPublisher(redis_url)


def create_event_subscriber(redis_url: str) -> EventSubscriber:
    """Create an event subscriber."""
    return EventSubscriber(redis_url)
