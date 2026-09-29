"""Authentication and authorization."""

from .dependencies import (
    get_current_user,
    get_current_user_flexible,
    get_current_user_from_api_key,
    require_role,
)
from .middleware import LocalUser, OptionalAuth
from .models import (
    APIKey,
    APIKeyCreate,
    APIKeyResponse,
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    Session,
    SubscriptionStatus,
    UsageQuotas,
    User,
    UserBase,
    UserCreate,
    UserResponse,
    UserRole,
    UserUpdate,
)
from .security import AuthCache, RoleChecker, SecurityManager, UsageTracker

__all__: list[str] = [
    "APIKey",
    "APIKeyCreate",
    "APIKeyResponse",
    "AuthCache",
    "LocalUser",
    "LoginRequest",
    "LoginResponse",
    "OptionalAuth",
    "RefreshTokenRequest",
    "RoleChecker",
    "SecurityManager",
    "Session",
    "SubscriptionStatus",
    "UsageQuotas",
    "UsageTracker",
    "User",
    "UserBase",
    "UserCreate",
    "UserResponse",
    "UserRole",
    "UserUpdate",
    "get_current_user",
    "get_current_user_flexible",
    "get_current_user_from_api_key",
    "require_role",
]
