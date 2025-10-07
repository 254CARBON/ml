"""Retry handler with exponential backoff for embedding operations."""

import asyncio
import random
import time
from typing import Any, Callable, Optional, Type
import structlog

logger = structlog.get_logger("retry_handler")


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        operation_name: str = "unknown",
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(
                        "Operation succeeded after retry",
                        operation=operation_name,
                        attempt=attempt + 1,
                        total_attempts=self.config.max_attempts
                    )
                
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    # Last attempt failed
                    logger.error(
                        "Operation failed after all retries",
                        operation=operation_name,
                        attempts=self.config.max_attempts,
                        error=str(e)
                    )
                    raise e
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    "Operation failed, retrying",
                    operation=operation_name,
                    attempt=attempt + 1,
                    total_attempts=self.config.max_attempts,
                    delay_seconds=delay,
                    error=str(e)
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry logic error")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Cap at max_delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(delay, 0.1)  # Minimum 100ms delay


class EmbeddingRetryHandler(RetryHandler):
    """Specialized retry handler for embedding operations."""
    
    def __init__(self):
        # Configure for embedding-specific scenarios
        config = RetryConfig(
            max_attempts=3,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                RuntimeError,  # GPU/model loading errors
                torch.cuda.OutOfMemoryError if torch.cuda.is_available() else Exception,
            )
        )
        super().__init__(config)
    
    async def generate_embeddings_with_retry(
        self,
        embedding_func: Callable,
        items: List[Dict[str, Any]],
        model_name: str,
        **kwargs
    ) -> Any:
        """Generate embeddings with retry logic."""
        
        return await self.execute_with_retry(
            embedding_func,
            items=items,
            model_name=model_name,
            operation_name=f"generate_embeddings_{model_name}",
            **kwargs
        )
    
    async def load_model_with_retry(
        self,
        load_func: Callable,
        model_name: str,
        **kwargs
    ) -> Any:
        """Load model with retry logic."""
        
        return await self.execute_with_retry(
            load_func,
            model_name=model_name,
            operation_name=f"load_model_{model_name}",
            **kwargs
        )


class BatchRetryHandler:
    """Handles retry logic for batch operations."""
    
    def __init__(self, retry_config: RetryConfig):
        self.retry_config = retry_config
        self.retry_handler = RetryHandler(retry_config)
    
    async def process_batch_with_retry(
        self,
        batch_func: Callable,
        items: List[Any],
        batch_size: int,
        operation_name: str = "batch_operation"
    ) -> List[Any]:
        """Process items in batches with retry logic."""
        
        results = []
        failed_batches = []
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_id = f"batch_{i//batch_size}"
            
            try:
                batch_result = await self.retry_handler.execute_with_retry(
                    batch_func,
                    batch,
                    operation_name=f"{operation_name}_{batch_id}"
                )
                results.extend(batch_result)
                
            except Exception as e:
                logger.error(
                    "Batch processing failed after retries",
                    batch_id=batch_id,
                    batch_size=len(batch),
                    error=str(e)
                )
                failed_batches.append({
                    "batch_id": batch_id,
                    "items": batch,
                    "error": str(e)
                })
        
        if failed_batches:
            logger.warning(
                "Some batches failed",
                total_batches=len(range(0, len(items), batch_size)),
                failed_batches=len(failed_batches),
                success_rate=(len(results) / len(items)) * 100
            )
        
        return results, failed_batches


def create_embedding_retry_handler() -> EmbeddingRetryHandler:
    """Create embedding retry handler."""
    return EmbeddingRetryHandler()


def create_batch_retry_handler(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0
) -> BatchRetryHandler:
    """Create batch retry handler."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=2.0,
        jitter=True
    )
    return BatchRetryHandler(config)


async def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    operation_name: str = "operation",
    *args,
    **kwargs
) -> Any:
    """Simple retry function with exponential backoff."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay
    )
    
    handler = RetryHandler(config)
    return await handler.execute_with_retry(
        func,
        *args,
        operation_name=operation_name,
        **kwargs
    )
