"""Authentication dependencies for FastAPI endpoints."""

from collections.abc import Callable
from datetime import timezone

# Python 3.10 compatibility: timezone.utc was added in 3.11
from src.api.utils.datetime_compat import UTC

try:
    from datetime import timezone
except ImportError:
    timezone.utc = timezone.utc  # noqa: UP017

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.api.database import get_db

from .models import APIKey, User, UserRole
from .security import (
    RoleChecker,
    compute_prefix_hash,
    security_manager,
    usage_tracker,
)

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current authenticated user from JWT token."""

    # Verify JWT token
    payload = security_manager.verify_token(credentials.credentials, "access")
    user_id = payload.get("sub")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user  # type: ignore[no-any-return]


def _validate_api_key_format(api_key: str) -> None:
    if not api_key.startswith("gms_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _lookup_cached_api_key(api_key: str, db: Session) -> APIKey | None:
    from .security import auth_cache

    cached_key_id = auth_cache.get(api_key)
    if not cached_key_id:
        return None
    record = db.query(APIKey).filter(APIKey.id == cached_key_id).first()
    if not record or not record.is_active:
        return None
    return record


def _lookup_api_key_by_prefix(api_key: str, db: Session) -> APIKey:
    key_body = api_key[4:]
    if len(key_body) < 8:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    prefix_for_index = key_body[:8]
    prefix_hash = compute_prefix_hash(prefix_for_index)

    try:
        active_keys = (
            db.query(APIKey)
            .filter(APIKey.is_active, APIKey.prefix_hash == prefix_hash)
            .all()
        )
    except (RuntimeError, ValueError, OSError):
        # Fallback: prefix_hash column doesn't exist yet (migration pending)
        active_keys = db.query(APIKey).filter(APIKey.is_active).all()

    if not active_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    for key_candidate in active_keys:
        if security_manager.verify_api_key(api_key, str(key_candidate.key_hash)):
            return key_candidate

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _get_active_user_for_api_key(api_key_record: APIKey, db: Session) -> User:
    user = db.query(User).filter(User.id == api_key_record.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user  # type: ignore[return-value]


def _update_api_key_usage(api_key_record: APIKey, db: Session) -> None:
    from datetime import datetime

    api_key_record.last_used = datetime.now(UTC)  # type: ignore[assignment]
    api_key_record.usage_count = int(api_key_record.usage_count) + 1  # type: ignore[assignment]
    db.commit()


async def get_current_user_from_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current user from API key.

    PERFORMANCE FIX: Uses prefix hash to filter candidates before bcrypt verification,
    reducing O(n) bcrypt calls to O(1) average case.
    """

    api_key = credentials.credentials
    _validate_api_key_format(api_key)

    api_key_record = _lookup_cached_api_key(api_key, db)

    if not api_key_record:
        api_key_record = _lookup_api_key_by_prefix(api_key, db)
        from .security import auth_cache

        auth_cache.set(api_key, api_key_record.id)

    user = _get_active_user_for_api_key(api_key_record, db)
    _update_api_key_usage(api_key_record, db)

    return user


async def get_current_user_flexible(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current user from either JWT token or API key."""

    token = credentials.credentials

    # Try API key first (if it starts with gms_)
    if token.startswith("gms_"):
        return await get_current_user_from_api_key(credentials, db)
    # Try JWT token
    return await get_current_user(credentials, db)


def require_role(required_role: UserRole) -> Callable[[User], User]:
    """Dependency factory for role-based access control."""

    def role_dependency(
        current_user: User = Depends(get_current_user_flexible),
    ) -> User:
        """Verify the current user has the required role."""
        role_checker = RoleChecker(required_role)

        if not role_checker(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}",
            )

        return current_user

    return role_dependency


def check_usage_quota(resource_type: str) -> Callable[[User, Session], User]:
    """Dependency factory for usage quota checking."""

    def quota_dependency(
        current_user: User = Depends(get_current_user_flexible),
        db: Session = Depends(get_db),
    ) -> User:
        """Enforce usage quota for the given resource type."""
        if not usage_tracker.check_quota(current_user, resource_type):
            user_role = UserRole(current_user.role)
            from .models import SUBSCRIPTION_QUOTAS

            quota_limit = getattr(
                SUBSCRIPTION_QUOTAS[user_role], f"{resource_type}_per_month"
            )

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Usage quota exceeded for {resource_type}. "
                f"Limit: {quota_limit} per month. "
                f"Upgrade your subscription for higher limits.",
            )

        # Increment usage counter
        usage_tracker.increment_usage(current_user, resource_type)
        db.commit()

        return current_user

    return quota_dependency


# Common dependency combinations
RequireAuth = Depends(get_current_user_flexible)
RequireProfessional = Depends(require_role(UserRole.PROFESSIONAL))
RequireEnterprise = Depends(require_role(UserRole.ENTERPRISE))
RequireAdmin = Depends(require_role(UserRole.ADMIN))

# Usage quota dependencies
CheckAPIQuota = Depends(check_usage_quota("api_calls"))
CheckVideoQuota = Depends(check_usage_quota("video_analyses"))
CheckSimulationQuota = Depends(check_usage_quota("simulations"))
