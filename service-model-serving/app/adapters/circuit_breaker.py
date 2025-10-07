"""Circuit breaker implementation for external service calls."""

import asyncio
import time
from typing import Any, Callable, Optional
from enum import Enum
import structlog

logger = structlog.get_logger("circuit_breaker")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "circuit_breaker"
    ):
        """Configure a circuit breaker.

        Parameters
        - failure_threshold: Failures before opening the breaker
        - recovery_timeout: Seconds to wait before HALF_OPEN probe
        - expected_exception: Exception type(s) treated as failures
        - name: Identifier for logs/metrics
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN", name=self.name)
                else:
                    logger.warning("Circuit breaker is OPEN, rejecting call", name=self.name)
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is open")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            await self._on_success()
            return result
            
        except self.expected_exception as e:
            await self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker reset to CLOSED", name=self.name)
            
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(
                    "Circuit breaker opened due to failures",
                    name=self.name,
                    failure_count=self.failure_count,
                    threshold=self.failure_threshold
                )
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }
    
    async def force_open(self):
        """Force circuit breaker to open state."""
        async with self._lock:
            self.state = CircuitBreakerState.OPEN
            self.last_failure_time = time.time()
            logger.warning("Circuit breaker forced to OPEN", name=self.name)
    
    async def force_close(self):
        """Force circuit breaker to closed state."""
        async with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info("Circuit breaker forced to CLOSED", name=self.name)


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self):
        """Create a registry for named circuit breakers."""
        self.breakers: dict[str, CircuitBreaker] = {}
    
    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                name=name
            )
        
        return self.breakers[name]
    
    def get_all_stats(self) -> dict:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            await breaker.force_close()
        
        logger.info("All circuit breakers reset")


# Global circuit breaker manager
_circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception
) -> CircuitBreaker:
    """Get a circuit breaker instance."""
    return _circuit_breaker_manager.get_breaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )


def get_circuit_breaker_stats() -> dict:
    """Get statistics for all circuit breakers."""
    return _circuit_breaker_manager.get_all_stats()


async def reset_all_circuit_breakers():
    """Reset all circuit breakers."""
    await _circuit_breaker_manager.reset_all()
