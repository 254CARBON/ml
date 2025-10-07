"""Security utilities for ML platform services."""

import os
import secrets
import hashlib
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
from passlib.context import CryptContext
import structlog

logger = structlog.get_logger("security")


class SecretManager:
    """Manages encryption and decryption of sensitive data."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
        else:
            # Generate a new key (for development only)
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            logger.warning("Using generated encryption key - not suitable for production")
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            raise
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> str:
        """Hash data with optional salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return f"{salt}:{hash_obj.hex()}"
    
    def verify_hash(self, data: str, hashed_data: str) -> bool:
        """Verify data against hash."""
        try:
            salt, hash_hex = hashed_data.split(':', 1)
            hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
            return hash_obj.hex() == hash_hex
        except Exception as e:
            logger.error("Hash verification failed", error=str(e))
            return False


class APIKeyManager:
    """Manages API keys for service authentication."""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.active_keys: Set[str] = set()
        self.revoked_keys: Set[str] = set()
    
    def generate_api_key(self, service_name: str, expires_in_days: int = 365) -> Dict[str, Any]:
        """Generate a new API key."""
        # Generate secure random key
        key = secrets.token_urlsafe(32)
        
        # Create key metadata
        key_data = {
            "key": key,
            "service_name": service_name,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=expires_in_days),
            "key_id": hashlib.sha256(key.encode()).hexdigest()[:16]
        }
        
        # Hash the key for storage
        hashed_key = self.pwd_context.hash(key)
        key_data["hashed_key"] = hashed_key
        
        # Add to active keys
        self.active_keys.add(key_data["key_id"])
        
        logger.info(
            "API key generated",
            service_name=service_name,
            key_id=key_data["key_id"],
            expires_at=key_data["expires_at"]
        )
        
        return key_data
    
    def validate_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key."""
        try:
            key_id = hashlib.sha256(key.encode()).hexdigest()[:16]
            
            # Check if key is revoked
            if key_id in self.revoked_keys:
                logger.warning("Revoked API key used", key_id=key_id)
                return None
            
            # Check if key is active
            if key_id not in self.active_keys:
                logger.warning("Unknown API key used", key_id=key_id)
                return None
            
            # In production, you'd query the database for key details
            # For now, return basic validation
            return {
                "key_id": key_id,
                "valid": True,
                "validated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error("API key validation failed", error=str(e))
            return None
    
    def revoke_api_key(self, key_id: str):
        """Revoke an API key."""
        self.revoked_keys.add(key_id)
        self.active_keys.discard(key_id)
        
        logger.info("API key revoked", key_id=key_id)
    
    def rotate_api_key(self, old_key_id: str, service_name: str) -> Dict[str, Any]:
        """Rotate an API key."""
        # Revoke old key
        self.revoke_api_key(old_key_id)
        
        # Generate new key
        new_key_data = self.generate_api_key(service_name)
        
        logger.info(
            "API key rotated",
            old_key_id=old_key_id,
            new_key_id=new_key_data["key_id"],
            service_name=service_name
        )
        
        return new_key_data


class InputSanitizer:
    """Sanitizes and validates user inputs."""
    
    def __init__(self):
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            r"<script",
            r"javascript:",
            r"data:text/html",
            r"vbscript:",
            r"onload=",
            r"onerror=",
            r"eval\(",
            r"exec\(",
            r"import\s+os",
            r"__import__",
            r"subprocess",
            r"system\("
        ]
    
    def sanitize_text_input(self, text: str, max_length: int = 10000) -> str:
        """Sanitize text input."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning("Input truncated", original_length=len(text), max_length=max_length)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Basic HTML escaping
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Check for dangerous patterns
        import re
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("Dangerous pattern detected in input", pattern=pattern)
                raise ValueError(f"Input contains potentially dangerous content")
        
        return text.strip()
    
    def validate_entity_id(self, entity_id: str) -> str:
        """Validate entity ID format."""
        if not isinstance(entity_id, str):
            raise ValueError("Entity ID must be a string")
        
        # Check length
        if len(entity_id) > 255:
            raise ValueError("Entity ID too long")
        
        # Check for valid characters (alphanumeric, underscore, dash, dot)
        import re
        if not re.match(r'^[a-zA-Z0-9_.-]+$', entity_id):
            raise ValueError("Entity ID contains invalid characters")
        
        return entity_id
    
    def validate_tenant_id(self, tenant_id: str) -> str:
        """Validate tenant ID format."""
        if not isinstance(tenant_id, str):
            raise ValueError("Tenant ID must be a string")
        
        # Check length
        if len(tenant_id) > 100:
            raise ValueError("Tenant ID too long")
        
        # Check for valid characters
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', tenant_id):
            raise ValueError("Tenant ID contains invalid characters")
        
        return tenant_id
    
    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata dictionary."""
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        
        sanitized = {}
        max_keys = 50
        max_value_length = 1000
        
        if len(metadata) > max_keys:
            logger.warning("Metadata has too many keys", count=len(metadata), max=max_keys)
            # Take first max_keys items
            metadata = dict(list(metadata.items())[:max_keys])
        
        for key, value in metadata.items():
            # Sanitize key
            if not isinstance(key, str) or len(key) > 100:
                continue
            
            # Sanitize value
            if isinstance(value, str):
                sanitized_value = self.sanitize_text_input(value, max_value_length)
                sanitized[key] = sanitized_value
            elif isinstance(value, (int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Sanitize list items
                sanitized_list = []
                for item in value[:10]:  # Limit list size
                    if isinstance(item, str):
                        sanitized_list.append(self.sanitize_text_input(item, 100))
                    elif isinstance(item, (int, float, bool)):
                        sanitized_list.append(item)
                sanitized[key] = sanitized_list
            # Skip other types
        
        return sanitized


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, redis_url: str):
        import redis
        self.redis_client = redis.from_url(redis_url)
        self.default_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000
        }
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limits: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Check if request is within rate limits."""
        if limits is None:
            limits = self.default_limits
        
        current_time = int(time.time())
        results = {"allowed": True, "limits": limits, "usage": {}}
        
        try:
            # Check each time window
            for window, limit in limits.items():
                if window == "requests_per_minute":
                    window_seconds = 60
                elif window == "requests_per_hour":
                    window_seconds = 3600
                elif window == "requests_per_day":
                    window_seconds = 86400
                else:
                    continue
                
                # Create window key
                window_start = current_time - (current_time % window_seconds)
                key = f"rate_limit:{identifier}:{endpoint}:{window_start}"
                
                # Get current count
                current_count = self.redis_client.get(key)
                current_count = int(current_count) if current_count else 0
                
                results["usage"][window] = {
                    "current": current_count,
                    "limit": limit,
                    "remaining": max(0, limit - current_count),
                    "reset_at": window_start + window_seconds
                }
                
                # Check if limit exceeded
                if current_count >= limit:
                    results["allowed"] = False
                    results["exceeded_limit"] = window
                    break
            
            # If allowed, increment counters
            if results["allowed"]:
                for window, limit in limits.items():
                    if window == "requests_per_minute":
                        window_seconds = 60
                    elif window == "requests_per_hour":
                        window_seconds = 3600
                    elif window == "requests_per_day":
                        window_seconds = 86400
                    else:
                        continue
                    
                    window_start = current_time - (current_time % window_seconds)
                    key = f"rate_limit:{identifier}:{endpoint}:{window_start}"
                    
                    # Increment and set expiry
                    pipeline = self.redis_client.pipeline()
                    pipeline.incr(key)
                    pipeline.expire(key, window_seconds)
                    pipeline.execute()
            
            return results
            
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e))
            # Fail open - allow request if rate limiting fails
            return {"allowed": True, "error": str(e)}


class SecurityAuditor:
    """Audits security-related events."""
    
    def __init__(self, redis_url: str):
        import redis
        self.redis_client = redis.from_url(redis_url)
        self.audit_key_prefix = "security:audit:"
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ):
        """Log a security-related event."""
        try:
            audit_entry = {
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "service_name": service_name,
                "details": details or {},
                "severity": severity
            }
            
            # Store in Redis with TTL
            key = f"{self.audit_key_prefix}{event_type}:{int(time.time())}"
            await self.redis_client.setex(
                key,
                86400 * 30,  # 30 days retention
                json.dumps(audit_entry, default=str)
            )
            
            logger.info(
                "Security event logged",
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                service_name=service_name
            )
            
        except Exception as e:
            logger.error("Failed to log security event", error=str(e))
    
    async def get_security_events(
        self,
        event_type: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent security events."""
        try:
            pattern = f"{self.audit_key_prefix}{event_type or '*'}:*"
            keys = await self.redis_client.keys(pattern)
            
            events = []
            for key in keys:
                event_data = await self.redis_client.get(key)
                if event_data:
                    event = json.loads(event_data)
                    
                    # Filter by time window
                    event_time = datetime.fromisoformat(event["timestamp"])
                    if (datetime.utcnow() - event_time).total_seconds() <= hours * 3600:
                        events.append(event)
            
            # Sort by timestamp
            events.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return events
            
        except Exception as e:
            logger.error("Failed to get security events", error=str(e))
            return []


class DataMasker:
    """Masks sensitive data in logs and responses."""
    
    def __init__(self):
        self.sensitive_fields = {
            "password", "secret", "key", "token", "credential",
            "ssn", "social_security", "credit_card", "bank_account"
        }
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'\b\d{3}-?\d{3}-?\d{4}\b'
    
    def mask_sensitive_data(self, data: Any) -> Any:
        """Mask sensitive data in various data structures."""
        if isinstance(data, dict):
            return self._mask_dict(data)
        elif isinstance(data, list):
            return [self.mask_sensitive_data(item) for item in data]
        elif isinstance(data, str):
            return self._mask_string(data)
        else:
            return data
    
    def _mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in dictionary."""
        masked = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if key indicates sensitive data
            if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                masked[key] = "***MASKED***"
            else:
                masked[key] = self.mask_sensitive_data(value)
        
        return masked
    
    def _mask_string(self, text: str) -> str:
        """Mask sensitive patterns in strings."""
        import re
        
        # Mask email addresses
        text = re.sub(self.email_pattern, "***EMAIL***", text)
        
        # Mask phone numbers
        text = re.sub(self.phone_pattern, "***PHONE***", text)
        
        # Mask potential API keys (long alphanumeric strings)
        text = re.sub(r'\b[A-Za-z0-9]{32,}\b', "***API_KEY***", text)
        
        return text


class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }


def create_secret_manager(encryption_key: Optional[str] = None) -> SecretManager:
    """Create secret manager."""
    return SecretManager(encryption_key)


def create_api_key_manager() -> APIKeyManager:
    """Create API key manager."""
    return APIKeyManager()


def create_input_sanitizer() -> InputSanitizer:
    """Create input sanitizer."""
    return InputSanitizer()


def create_security_auditor(redis_url: str) -> SecurityAuditor:
    """Create security auditor."""
    return SecurityAuditor(redis_url)


def create_data_masker() -> DataMasker:
    """Create data masker."""
    return DataMasker()
