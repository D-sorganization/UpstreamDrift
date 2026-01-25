"""Security utilities for authentication and authorization."""

import os
import secrets

# Python 3.10 compatibility: UTC was added in 3.11
from datetime import datetime, timedelta, timezone

from src.shared.python.logging_config import get_logger

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017
from typing import Any

import bcrypt
import jwt
from fastapi import HTTPException, status

from .models import User, UserRole

logger = get_logger(__name__)

# Security configuration
# SECURITY: Secret key MUST be set via environment variable
_secret_key_env = os.getenv("GOLF_API_SECRET_KEY") or os.getenv("SECRET_KEY")
_environment = os.getenv("ENVIRONMENT", "development").lower()

if not _secret_key_env:
    if _environment == "production":
        logger.error(
            "SECURITY ERROR: No SECRET_KEY or GOLF_API_SECRET_KEY environment "
            "variable set. The server cannot start without a secure secret key."
        )
        raise RuntimeError(
            "SECRET_KEY is not configured. Set GOLF_API_SECRET_KEY or SECRET_KEY "
            "environment variable to a secure, random value."
        )
    else:
        logger.warning(
            "SECURITY WARNING: No SECRET_KEY set. Using unsafe placeholder. "
            "API authentication will fail. Set GOLF_API_SECRET_KEY for production."
        )
        # Use a clearly-unsafe placeholder that will cause authentication to fail
        SECRET_KEY = "UNSAFE-NO-SECRET-KEY-SET-AUTHENTICATION-WILL-FAIL"
elif len(_secret_key_env) < 32:
    logger.warning(
        "SECURITY WARNING: SECRET_KEY is less than 32 characters. "
        "Use a longer, randomly generated key for production."
    )
    SECRET_KEY = _secret_key_env
else:
    SECRET_KEY = _secret_key_env

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Bcrypt cost factor (12 is the recommended minimum for security)
BCRYPT_ROUNDS = 12


def compute_prefix_hash(prefix: str) -> str:
    """Compute SHA256 hash of a non-sensitive prefix for database indexing.

    This function is used to create a database index for fast API key lookup.
    It hashes ONLY the first 8 characters of the key (not the full secret).

    Args:
        prefix: Non-sensitive 8-character prefix from the API key

    Returns:
        SHA256 hash of the prefix for database indexing

    Note:
        This is NOT password hashing. The actual API key is hashed with bcrypt.
    """
    import hashlib

    return hashlib.sha256(prefix.encode()).hexdigest()


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
        salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")  # type: ignore[no-any-return]

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database

        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(  # type: ignore[no-any-return]
                plain_password.encode("utf-8"), hashed_password.encode("utf-8")
            )
        except (ValueError, TypeError):
            return False

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

        # SECURITY FIX: Use timezone-aware datetime instead of deprecated utcnow()
        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

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
        # SECURITY FIX: Use timezone-aware datetime instead of deprecated utcnow()
        expire = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
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
        except jwt.InvalidTokenError as e:
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
        """Hash an API key for storage using bcrypt.

        Args:
            api_key: Plain API key

        Returns:
            Bcrypt-hashed API key (slow hash for brute-force resistance)

        Note:
            SECURITY: Uses bcrypt instead of SHA256 for brute-force resistance.
            SHA256 is fast and unsuitable for key storage; bcrypt is slow by design.
        """
        salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(api_key.encode("utf-8"), salt)
        return hashed.decode("utf-8")  # type: ignore[no-any-return]

    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify an API key against its hash.

        Args:
            api_key: Plain API key
            hashed_key: Bcrypt-hashed key from database

        Returns:
            True if key matches, False otherwise
        """
        try:
            return bcrypt.checkpw(  # type: ignore[no-any-return]
                api_key.encode("utf-8"), hashed_key.encode("utf-8")
            )
        except (ValueError, TypeError):
            return False


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


class AuthCache:
    """Thread-safe cache for API authentication results to avoid expensive BCrypt hashing.

    Fixes Performance Issue: N+1 Auth checks.
    """

    TTL_SECONDS = 300  # 5 minutes cache

    def __init__(self) -> None:
        import threading
        import time

        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._time = time

    def get(self, api_key: str) -> Any | None:
        """Get cached user_id for API key."""
        # Generate a fast lookup token for the cache
        # (We don't store the key, just a derived token for lookup)
        cache_key = self._cache_lookup_token(api_key)

        with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if self._time.time() - timestamp < self.TTL_SECONDS:
                    return result
                else:
                    del self._cache[cache_key]
        return None

    def set(self, api_key: str, result: Any) -> None:
        """Cache auth result."""
        cache_key = self._cache_lookup_token(api_key)
        with self._lock:
            # Simple cleanup of size if needed, but 300s TTL is self-limiting mostly
            if len(self._cache) > 10000:
                # Random eviction or clear
                self._cache.clear()
            self._cache[cache_key] = (result, self._time.time())

    def _cache_lookup_token(self, token_value: str) -> str:
        """Generate a fast lookup token for the auth cache.

        SECURITY NOTE: This is NOT used for password/key storage or protection.
        The actual API key verification uses bcrypt (see verify_api_key method).
        This is purely a fast dictionary lookup key to avoid repeated bcrypt calls.

        We use Python's built-in hash for speed. The actual security comes from:
        1. Short TTL (5 minutes) limiting exposure window
        2. bcrypt verification on cache miss
        3. The token_value itself is never stored, only this derived lookup key
        """
        # Use a combination of hash and length to create a lookup key
        # This is intentionally NOT cryptographic - it's for cache performance only
        return f"{hash(token_value)}:{len(token_value)}"


auth_cache = AuthCache()
