"""Authentication and authorization utilities.

Central helpers for JWT issuance/verification, password hashing, simple
service‑to‑service authentication, and lightweight tenant scoping.

Design
- "AuthManager" is for end‑user tokens
- "InternalAuthManager" is for service tokens ("type=internal")
- Small FastAPI dependencies offer a drop‑in integration path
"""

import os
import jwt
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, Optional, Union, Callable
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import structlog

logger = structlog.get_logger("auth")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer()


class AuthManager:
    """Authentication manager for ML platform services.

    Issues and validates end‑user JWTs. Keep payloads minimal (subject,
    roles/permissions, tenant) and avoid sensitive data.
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30
    ):
        """Configure JWT settings for end-user tokens.

        Parameters
        - secret_key: Symmetric key for signing/verification
        - algorithm: JWT algorithm (default HS256)
        - access_token_expire_minutes: Default token TTL in minutes
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token.

        Raises ``HTTPException`` with 401 on invalid/expired tokens.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)


class InternalAuthManager:
    """Internal authentication manager for service-to-service communication.

    Encodes a ``type=internal`` claim so downstream checks can distinguish
    machine tokens from end‑user tokens.
    """
    
    def __init__(self, secret_key: str):
        """Set up internal-token manager with a signing key."""
        self.secret_key = secret_key
        self.algorithm = "HS256"
    
    def create_internal_token(self, service_name: str) -> str:
        """Create an internal service token."""
        data = {
            "service": service_name,
            "type": "internal",
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(data, self.secret_key, algorithm=self.algorithm)
    
    def verify_internal_token(self, token: str) -> Dict[str, Any]:
        """Verify an internal service token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "internal":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Internal token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid internal token"
            )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends()
) -> Dict[str, Any]:
    """Get current user from JWT token."""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    return payload


def get_current_service(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: InternalAuthManager = Depends()
) -> Dict[str, Any]:
    """Get current service from internal JWT token."""
    token = credentials.credentials
    payload = auth_manager.verify_internal_token(token)
    return payload


def require_tenant(tenant_id: str) -> str:
    """Require a specific tenant ID."""
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )
    return tenant_id


def require_permission(permission: str) -> Callable:
    """Require a specific permission.

    Note: This decorator logs the requirement but does not enforce it.
    Integrate your RBAC check here. The function currently uses ``functools.wraps``
    but the import may be missing; ensure it is available in your application
    context.
    """
    def decorator(func: Callable) -> Callable:
        """Decorate a function to log a permission requirement before call."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper that logs and then forwards the call unchanged."""
            # This would check permissions against user roles
            # For now, we'll just log the requirement
            logger.info("Permission required", permission=permission)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TenantContext:
    """Context manager for tenant-specific operations."""
    
    def __init__(self, tenant_id: str):
        """Bind a tenant_id for the duration of the context manager."""
        self.tenant_id = tenant_id
        self._previous_tenant = None
    
    def __enter__(self):
        """Set the current tenant on entry and return self."""
        # Store current tenant context
        self._previous_tenant = getattr(tenant_context, 'current_tenant', None)
        tenant_context.current_tenant = self.tenant_id
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the previous tenant context on exit."""
        # Restore previous tenant context
        tenant_context.current_tenant = self._previous_tenant


# Global tenant context
class TenantContextManager:
    """Global tenant context manager."""
    
    def __init__(self):
        """Initialize holder for process-level current_tenant value."""
        self.current_tenant: Optional[str] = None
    
    def get_current_tenant(self) -> Optional[str]:
        """Get current tenant ID."""
        return self.current_tenant
    
    def set_current_tenant(self, tenant_id: str) -> None:
        """Set current tenant ID."""
        self.current_tenant = tenant_id


tenant_context = TenantContextManager()


def get_current_tenant() -> Optional[str]:
    """Get current tenant ID from context."""
    return tenant_context.get_current_tenant()


def set_current_tenant(tenant_id: str) -> None:
    """Set current tenant ID in context."""
    tenant_context.set_current_tenant(tenant_id)


def create_optional_auth_dependency(config) -> Callable:
    """Create optional auth dependency with config."""
    optional_security = HTTPBearer(auto_error=False)
    
    def optional_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security)
    ) -> Optional[Dict[str, Any]]:
        """Optional FastAPI dependency for JWT validation.
        
        Returns user info if token is valid, None if no token provided.
        Controlled by ML_AUTH_ENABLED environment variable.
        """
        auth_enabled = os.getenv("ML_AUTH_ENABLED", "false").lower() == "true"
        
        if not auth_enabled:
            return None
        
        if not credentials:
            return None
        
        try:
            payload = jwt.decode(
                credentials.credentials,
                config.ml_jwt_secret_key,
                algorithms=[config.ml_jwt_algorithm]
            )
            username: str = payload.get("sub")
            tenant_id: str = payload.get("tenant_id", "default")
            
            if username is None:
                return None
                
            return {
                "username": username, 
                "tenant_id": tenant_id,
                "payload": payload
            }
        except jwt.PyJWTError:
            return None
    
    return optional_auth


def create_auth_manager_from_config(config) -> AuthManager:
    """Create auth manager from config."""
    return AuthManager(
        secret_key=config.ml_jwt_secret_key,
        algorithm=config.ml_jwt_algorithm,
        access_token_expire_minutes=config.ml_jwt_access_token_expire_minutes
    )
