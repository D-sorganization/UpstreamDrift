"""Security utilities for authentication and authorization."""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any

import jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext

from .models import User, UserRole

# Security configuration
SECRET_KEY = "your-secret-key-change-in-production"  # SECURITY: Move to environment variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """Handles authentication and authorization security."""

    def __init__(self, secret_key: str = SECRET_KEY):
        """Initialize security manager.

        Args:
            secret_key: JWT signing secret key
        """
        self.secret_key = secret_key
        self.algorithm = ALGORITHM

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return str(pwd_context.hash(password))

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database

        Returns:
            True if password matches, False otherwise
        """
        return bool(pwd_context.verify(plain_password, hashed_password))

    def create_access_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """Create a JWT access token.

        Args:
            data: Token payload data
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})
        return str(jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm))

    def create_refresh_token(self, data: dict[str, Any]) -> str:
        """Create a JWT refresh token.

        Args:
            data: Token payload data

        Returns:
            Encoded JWT refresh token
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        return str(jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm))

    def verify_token(self, token: str, token_type: str = "access") -> dict[str, Any]:
        """Verify and decode a JWT token.

        Args:
            token: JWT token to verify
            token_type: Expected token type (access or refresh)

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Verify token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return dict(payload)

        except jwt.ExpiredSignatureError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e
        except jwt.JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    def generate_api_key(self) -> str:
        """Generate a new API key.

        Returns:
            Generated API key with gms_ prefix
        """
        key = secrets.token_urlsafe(32)
        return f"gms_{key}"

    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage.

        Args:
            api_key: Plain API key

        Returns:
            Hashed API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()


class RoleChecker:
    """Role-based access control checker."""

    def __init__(self, required_role: UserRole):
        """Initialize role checker.

        Args:
            required_role: Minimum required role
        """
        self.required_role = required_role
        self.role_hierarchy = {
            UserRole.FREE: 0,
            UserRole.PROFESSIONAL: 1,
            UserRole.ENTERPRISE: 2,
            UserRole.ADMIN: 3,
        }

    def __call__(self, user: User) -> bool:
        """Check if user has required role.

        Args:
            user: User to check

        Returns:
            True if user has sufficient role
        """
        user_role_level = self.role_hierarchy.get(UserRole(user.role), 0)
        required_role_level = self.role_hierarchy.get(self.required_role, 0)

        return user_role_level >= required_role_level


class UsageTracker:
    """Tracks and enforces usage quotas."""

    def __init__(self) -> None:
        """Initialize usage tracker."""
        pass

    def check_quota(self, user: User, resource_type: str) -> bool:
        """Check if user has quota remaining for a resource.

        Args:
            user: User to check
            resource_type: Type of resource (api_calls, video_analyses, simulations)

        Returns:
            True if user has quota remaining
        """
        from .models import SUBSCRIPTION_QUOTAS

        user_role = UserRole(user.role)
        quotas = SUBSCRIPTION_QUOTAS[user_role]

        if resource_type == "api_calls":
            return int(user.api_calls_this_month) < quotas.api_calls_per_month
        elif resource_type == "video_analyses":
            return int(user.video_analyses_this_month) < quotas.video_analyses_per_month
        elif resource_type == "simulations":
            return int(user.simulations_this_month) < quotas.simulations_per_month

        return False

    def increment_usage(self, user: User, resource_type: str) -> None:
        """Increment usage counter for a user.

        Args:
            user: User to increment usage for
            resource_type: Type of resource used
        """
        if resource_type == "api_calls":
            user.api_calls_this_month = int(user.api_calls_this_month) + 1  # type: ignore[assignment]
        elif resource_type == "video_analyses":
            user.video_analyses_this_month = int(user.video_analyses_this_month) + 1  # type: ignore[assignment]
        elif resource_type == "simulations":
            user.simulations_this_month = int(user.simulations_this_month) + 1  # type: ignore[assignment]

    def get_usage_summary(self, user: User) -> dict[str, Any]:
        """Get usage summary for a user.

        Args:
            user: User to get summary for

        Returns:
            Usage summary dictionary
        """
        from .models import SUBSCRIPTION_QUOTAS

        user_role = UserRole(user.role)
        quotas = SUBSCRIPTION_QUOTAS[user_role]

        api_calls_used = int(user.api_calls_this_month)
        video_analyses_used = int(user.video_analyses_this_month)
        simulations_used = int(user.simulations_this_month)

        return {
            "subscription_tier": user_role.value,
            "api_calls": {
                "used": api_calls_used,
                "limit": quotas.api_calls_per_month,
                "remaining": max(0, quotas.api_calls_per_month - api_calls_used),
            },
            "video_analyses": {
                "used": video_analyses_used,
                "limit": quotas.video_analyses_per_month,
                "remaining": max(
                    0, quotas.video_analyses_per_month - video_analyses_used
                ),
            },
            "simulations": {
                "used": simulations_used,
                "limit": quotas.simulations_per_month,
                "remaining": max(0, quotas.simulations_per_month - simulations_used),
            },
        }


# Global instances
security_manager = SecurityManager()
usage_tracker = UsageTracker()
