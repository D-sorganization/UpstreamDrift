"""Authentication routes for user management."""

# Python 3.10 compatibility: UTC was added in 3.11
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from src.api.auth.dependencies import RequireAdmin, RequireAuth
from src.api.auth.models import (
    APIKey,
    APIKeyCreate,
    APIKeyResponse,
    LoginRequest,
    LoginResponse,
    User,
    UserCreate,
    UserResponse,
    UserRole,
)
from src.api.auth.security import security_manager, usage_tracker
from src.api.database import get_db

router = APIRouter(prefix="/auth", tags=["authentication"])

# Rate limiting constants for auth endpoints (Issue #522)
# Protects against credential stuffing and brute force attacks
REGISTRATION_RATE_LIMIT = "3/hour"
LOGIN_RATE_LIMIT = "5/minute"

# Use shared limiter - registered with app.state in server.py
# This ensures proper rate limiting across all routes
limiter = Limiter(key_func=get_remote_address)


@router.post("/register", response_model=UserResponse)
@limiter.limit(
    REGISTRATION_RATE_LIMIT
)  # SECURITY: Limit registration to prevent account farming
async def register_user(
    request: Request, user_data: UserCreate, db: Session = Depends(get_db)
) -> UserResponse:
    """Register a new user."""

    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create new user
    hashed_password = security_manager.hash_password(user_data.password)
    db_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        organization=user_data.organization,
        role=UserRole.FREE.value,  # New users start with free tier
        is_active=True,
        is_verified=False,  # Email verification required
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@router.post("/login", response_model=LoginResponse)
@limiter.limit(
    LOGIN_RATE_LIMIT
)  # SECURITY: Limit login attempts to prevent brute force
async def login(
    request: Request, login_data: LoginRequest, db: Session = Depends(get_db)
) -> LoginResponse:
    """Authenticate user and return tokens."""

    # Find user
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    # Verify password
    if not security_manager.verify_password(
        login_data.password, str(user.hashed_password)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is deactivated"
        )

    # Create tokens
    access_token_expires = timedelta(minutes=30)
    access_token = security_manager.create_access_token(
        data={"sub": str(user.id), "email": user.email},
        expires_delta=access_token_expires,
    )

    refresh_token = security_manager.create_refresh_token(
        data={"sub": str(user.id), "email": user.email}
    )

    # Update last login
    user.last_login = datetime.now(UTC)  # type: ignore[assignment]
    db.commit()

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_token_expires.total_seconds()),
        user=user,
    )


@router.post("/refresh", response_model=dict)
async def refresh_token(
    refresh_token: str, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Refresh access token using refresh token."""

    # Verify refresh token
    payload = security_manager.verify_token(refresh_token, "refresh")
    user_id = payload.get("sub")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    # Get user
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Create new access token
    access_token_expires = timedelta(minutes=30)
    access_token = security_manager.create_access_token(
        data={"sub": str(user.id), "email": user.email},
        expires_delta=access_token_expires,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_token_expires.total_seconds()),
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = RequireAuth) -> UserResponse:
    """Get current user information."""
    return current_user


@router.get("/usage", response_model=dict)
async def get_usage_info(current_user: User = RequireAuth) -> dict[str, Any]:
    """Get current user's usage information."""
    return usage_tracker.get_usage_summary(current_user)


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = RequireAuth,
    db: Session = Depends(get_db),
) -> APIKeyResponse:
    """Create a new API key for the current user."""

    # Generate API key
    api_key = security_manager.generate_api_key()
    api_key_hash = security_manager.hash_api_key(api_key)

    # Create API key record
    db_api_key = APIKey(
        user_id=current_user.id,
        key_hash=api_key_hash,
        name=api_key_data.name,
        is_active=True,
    )

    db.add(db_api_key)
    db.commit()
    db.refresh(db_api_key)

    # Return API key (only time it's shown in plain text)
    response = APIKeyResponse.from_orm(db_api_key)
    response.key = api_key  # Include the actual key in response

    return response  # type: ignore[no-any-return]


@router.get("/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    current_user: User = RequireAuth, db: Session = Depends(get_db)
) -> list[APIKeyResponse]:
    """List all API keys for the current user."""

    api_keys = db.query(APIKey).filter(APIKey.user_id == current_user.id).all()
    return [APIKeyResponse.from_orm(key) for key in api_keys]


@router.delete("/api-keys/{api_key_id}")
async def delete_api_key(
    api_key_id: int, current_user: User = RequireAuth, db: Session = Depends(get_db)
) -> dict[str, str]:
    """Delete an API key."""

    api_key = (
        db.query(APIKey)
        .filter(APIKey.id == api_key_id, APIKey.user_id == current_user.id)
        .first()
    )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    db.delete(api_key)
    db.commit()

    return {"message": "API key deleted successfully"}


# Admin routes
@router.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = RequireAdmin,
    db: Session = Depends(get_db),
) -> list[UserResponse]:
    """List all users (admin only)."""

    users = db.query(User).offset(skip).limit(limit).all()
    return [UserResponse.from_orm(user) for user in users]


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    new_role: UserRole,
    current_user: User = RequireAdmin,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Update user role (admin only)."""

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    user.role = new_role.value  # type: ignore[assignment]
    db.commit()

    return {"message": f"User role updated to {new_role.value}"}


@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    is_active: bool,
    current_user: User = RequireAdmin,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Update user active status (admin only)."""

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    user.is_active = is_active  # type: ignore[assignment]
    db.commit()

    status_text = "activated" if is_active else "deactivated"
    return {"message": f"User {status_text} successfully"}
