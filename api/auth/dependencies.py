"""Authentication dependencies for FastAPI endpoints."""

from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from api.database import get_db

from .models import APIKey, User, UserRole
from .security import RoleChecker, security_manager, usage_tracker

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

    return user


async def get_current_user_from_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current user from API key."""

    api_key = credentials.credentials

    # Check if it's an API key (starts with gms_)
    if not api_key.startswith("gms_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Hash the provided API key
    import hashlib

    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Find API key in database
    api_key_record = (
        db.query(APIKey)
        .filter(APIKey.key_hash == api_key_hash, APIKey.is_active)
        .first()
    )

    if not api_key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get associated user
    user = db.query(User).filter(User.id == api_key_record.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update API key usage
    from datetime import datetime

    api_key_record.last_used = datetime.utcnow()  # type: ignore[assignment]
    api_key_record.usage_count = int(api_key_record.usage_count) + 1  # type: ignore[assignment]
    db.commit()

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
    else:
        # Try JWT token
        return await get_current_user(credentials, db)


def require_role(required_role: UserRole) -> Callable[[User], User]:
    """Dependency factory for role-based access control."""

    def role_dependency(
        current_user: User = Depends(get_current_user_flexible),
    ) -> User:
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
