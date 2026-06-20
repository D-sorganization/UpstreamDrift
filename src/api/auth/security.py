"""Security utilities for authentication and authorization."""

import os
import secrets
from datetime import datetime, timedelta, timezone

# Python 3.10 compatibility: timezone.utc was added in 3.11
from src.api.utils.datetime_compat import UTC
from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

try:
    from datetime import timezone
except ImportError:
    timezone.utc = timezone.utc  # noqa: UP017
from typing import Any, cast

import bcrypt
import jwt
from fastapi import HTTPException, status
from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session as SQLAlchemySession

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
        # Issue #1779: Generate a random per-process key in non-production mode.
        # A known-public static string in source code allows ANYONE reading the
        # code to forge valid JWT tokens. A randomly generated key is unguessable
        # and scoped to this process lifetime, so forged tokens cannot be externally
        # crafted. Tokens will be invalidated on process restart.
        SECRET_KEY = secrets.token_urlsafe(32)
        logger.warning(
            "SECURITY WARNING: No GOLF_API_SECRET_KEY or SECRET_KEY env var set. "
            "A random per-process key has been generated; all JWT tokens will be "
            "invalidated on restart. Set GOLF_API_SECRET_KEY for production."
        )
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
BCRYPT_MAX_INPUT_BYTES = 72

_USAGE_COUNTER_COLUMNS = {
    "api_calls": User.api_calls_this_month,
    "video_analyses": User.video_analyses_this_month,
    "simulations": User.simulations_this_month,
}

_USAGE_QUOTA_FIELDS = {
    "api_calls": "api_calls_per_month",
    "video_analyses": "video_analyses_per_month",
    "simulations": "simulations_per_month",
}


def _validate_bcrypt_secret(value: str, field_name: str) -> bytes:
    """Return UTF-8 bytes for a bcrypt secret after enforcing bcrypt limits."""
    if value is None:
        raise ValueError(f"{field_name} must be provided")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")

    encoded = value.encode("utf-8")
    if len(encoded) > BCRYPT_MAX_INPUT_BYTES:
        raise ValueError(
            f"{field_name} must be at most {BCRYPT_MAX_INPUT_BYTES} bytes "
            "when UTF-8 encoded for bcrypt"
        )
    return encoded


def _log_bcrypt_verification_failure(
    operation: str, exc: ValueError | TypeError
) -> None:
    """Log failed bcrypt verification without exposing credentials or hashes."""
    operation_label = "API key" if operation == "api_key" else operation
    failure_label = "API key" if operation == "api_key" else "Password"
    logger.warning(
        "%s verification failed: Malformed stored bcrypt hash during %s verification",
        failure_label,
        operation_label,
        exc_info=(type(exc), exc, exc.__traceback__),
        operation=operation,
        exception_type=type(exc).__name__,
    )


@precondition(
    lambda prefix: isinstance(prefix, str) and len(prefix) > 0,
    "prefix must be a non-empty string",
)
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

    def __init__(self, secret_key: str = SECRET_KEY) -> None:
        """Initialize security manager.

        Args:
            secret_key: JWT signing secret key
        """
        if secret_key is None:
            raise ValueError("secret_key must be provided")
        self.secret_key = secret_key
        self.algorithm = ALGORITHM
        self.pwd_context = bcrypt

    @precondition(
        lambda self, password: isinstance(password, str) and len(password) > 0,
        "password must be a non-empty string",
    )
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        password_bytes = _validate_bcrypt_secret(password, "password")
        salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(password_bytes, salt)
        if not isinstance(hashed, bytes):
            raise TypeError("bcrypt.hashpw must return bytes")
        return hashed.decode("utf-8")

    @precondition(
        lambda self, plain_password, hashed_password: (
            isinstance(plain_password, str) and len(plain_password) > 0
        ),
        "plain_password must be a non-empty string",
    )
    @precondition(
        lambda self, plain_password, hashed_password: (
            isinstance(hashed_password, str) and len(hashed_password) > 0
        ),
        "hashed_password must be a non-empty string",
    )
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database

        Returns:
            True if password matches, False otherwise
        """
        try:
            plain_password_bytes = _validate_bcrypt_secret(
                plain_password, "plain_password"
            )
            hashed_password_bytes = _validate_bcrypt_secret(
                hashed_password, "hashed_password"
            )
            return bool(bcrypt.checkpw(plain_password_bytes, hashed_password_bytes))
        except (ValueError, TypeError) as exc:
            _log_bcrypt_verification_failure("password", exc)
            return False

    @precondition(
        lambda self, data, expires_delta=None: isinstance(data, dict) and "sub" in data,
        "data must be a dict containing a 'sub' (subject) claim",
    )
    def create_access_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """Create a JWT access token.

        Args:
            data: Token payload data (must contain 'sub' key)
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        if data is None:
            raise ValueError("data must be provided")
        if not isinstance(data, dict):
            raise TypeError("data must be a dict containing a 'sub' claim")
        if "sub" not in data:
            raise ValueError("data must contain a 'sub' claim")
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
        if data is None:
            raise ValueError("data must be provided")
        to_encode = data.copy()
        # SECURITY FIX: Use timezone-aware datetime instead of deprecated utcnow()
        expire = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        return str(jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm))

    @precondition(
        lambda self, token, token_type="access": (
            isinstance(token, str) and len(token) > 0
        ),
        "token must be a non-empty string",
    )
    @precondition(
        lambda self, token, token_type="access": token_type in ("access", "refresh"),
        "token_type must be 'access' or 'refresh'",
    )
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
        api_key_bytes = _validate_bcrypt_secret(api_key, "api_key")
        salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(api_key_bytes, salt)
        if not isinstance(hashed, bytes):
            raise TypeError("bcrypt.hashpw must return bytes")
        return hashed.decode("utf-8")

    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify an API key against its hash.

        Args:
            api_key: Plain API key
            hashed_key: Bcrypt-hashed key from database

        Returns:
            True if key matches, False otherwise
        """
        try:
            api_key_bytes = _validate_bcrypt_secret(api_key, "api_key")
            hashed_key_bytes = _validate_bcrypt_secret(hashed_key, "hashed_key")
            return bool(bcrypt.checkpw(api_key_bytes, hashed_key_bytes))
        except (ValueError, TypeError) as exc:
            _log_bcrypt_verification_failure("api_key", exc)
            return False


class RoleChecker:
    """Role-based access control checker."""

    def __init__(self, required_role: UserRole) -> None:
        """Initialize role checker.

        Args:
            required_role: Minimum required role
        """
        if required_role is None:
            raise ValueError("required_role must be provided")
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
        if user is None:
            raise ValueError("user must be provided")
        user_role_level = self.role_hierarchy.get(UserRole(user.role), 0)
        required_role_level = self.role_hierarchy.get(self.required_role, 0)

        return user_role_level >= required_role_level


class UsageTracker:
    """Tracks and enforces usage quotas."""

    def __init__(self) -> None:
        """Initialize usage tracker."""

    @precondition(
        lambda self, user, resource_type: (
            resource_type in ("api_calls", "video_analyses", "simulations")
        ),
        "resource_type must be 'api_calls', 'video_analyses', or 'simulations'",
    )
    def check_quota(self, user: User, resource_type: str) -> bool:
        """Check if user has quota remaining for a resource.

        Args:
            user: User to check
            resource_type: Type of resource (api_calls, video_analyses, simulations)

        Returns:
            True if user has quota remaining
        """
        if user is None:
            raise ValueError("user must be provided")

        if resource_type == "api_calls":
            return bool(
                int(user.api_calls_this_month) < self.quota_limit(user, resource_type)
            )
        if resource_type == "video_analyses":
            return bool(
                int(user.video_analyses_this_month)
                < self.quota_limit(user, resource_type)
            )
        if resource_type == "simulations":
            return bool(
                int(user.simulations_this_month) < self.quota_limit(user, resource_type)
            )

        return False

    @precondition(
        lambda self, user, resource_type: (
            resource_type in ("api_calls", "video_analyses", "simulations")
        ),
        "resource_type must be 'api_calls', 'video_analyses', or 'simulations'",
    )
    def quota_limit(self, user: User, resource_type: str) -> int:
        """Return the monthly limit for a user's resource type."""
        if user is None:
            raise ValueError("user must be provided")
        from .models import SUBSCRIPTION_QUOTAS

        user_role = UserRole(user.role)
        quotas = SUBSCRIPTION_QUOTAS[user_role]
        return int(getattr(quotas, _USAGE_QUOTA_FIELDS[resource_type]))

    @precondition(
        lambda self, db, user, resource_type: (
            resource_type in ("api_calls", "video_analyses", "simulations")
        ),
        "resource_type must be 'api_calls', 'video_analyses', or 'simulations'",
    )
    def consume_quota(
        self, db: SQLAlchemySession, user: User, resource_type: str
    ) -> bool:
        """Atomically consume one quota unit if the user is still below the limit.

        The bounded UPDATE is the authoritative quota transition. It runs in a
        short independent transaction so metering does not commit unrelated
        pending work from the request session.
        """
        if db is None:
            raise ValueError("db must be provided")
        if user is None:
            raise ValueError("user must be provided")

        return self._apply_quota_delta(db, user, resource_type, 1)

    @precondition(
        lambda self, db, user, resource_type: (
            resource_type in ("api_calls", "video_analyses", "simulations")
        ),
        "resource_type must be 'api_calls', 'video_analyses', or 'simulations'",
    )
    def refund_quota(
        self, db: SQLAlchemySession, user: User, resource_type: str
    ) -> bool:
        """Atomically refund one previously reserved quota unit."""
        if db is None:
            raise ValueError("db must be provided")
        if user is None:
            raise ValueError("user must be provided")

        return self._apply_quota_delta(db, user, resource_type, -1)

    def _apply_quota_delta(
        self,
        db: SQLAlchemySession,
        user: User,
        resource_type: str,
        delta: int,
    ) -> bool:
        if delta not in (-1, 1):
            raise ValueError("delta must be -1 or 1")

        counter_column = _USAGE_COUNTER_COLUMNS[resource_type]
        boundary = (
            counter_column < self.quota_limit(user, resource_type)
            if delta > 0
            else counter_column > 0
        )
        statement = (
            update(User)
            .where(User.id == user.id, boundary)
            .values({counter_column: counter_column + delta})
        )

        with SQLAlchemySession(bind=db.get_bind(), autoflush=False) as quota_db:
            result = cast(CursorResult[Any], quota_db.execute(statement))
            if result.rowcount != 1:
                quota_db.rollback()
                return False
            quota_db.commit()

        db.expire(user, [counter_column.key])
        return True

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
        if user is None:
            raise ValueError("user must be provided")
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
    """Thread-safe cache for API authentication results to avoid expensive BCrypt hashing.  # noqa: E501

    Fixes Performance Issue: N+1 Auth checks.
    """

    TTL_SECONDS = 300  # 5 minutes cache (default; overridable via env)
    MAX_ENTRIES = 10_000  # default; overridable via env

    def __init__(
        self,
        ttl_seconds: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        import threading
        import time

        # Resolve sizing from env at construction so multi-worker
        # deployments can tune cache pressure without code changes.
        # Class-level constants remain the source of defaults and are
        # honoured when callers (or tests) monkeypatch them.
        from src.shared.python.config.environment import (
            get_auth_cache_max_entries,
            get_auth_cache_ttl_seconds,
        )

        if ttl_seconds is None:
            ttl_seconds = get_auth_cache_ttl_seconds(default=self.TTL_SECONDS)
        if ttl_seconds < 1:
            raise ValueError(f"ttl_seconds must be >= 1, got {ttl_seconds}")
        if max_entries is None:
            max_entries = get_auth_cache_max_entries(default=self.MAX_ENTRIES)
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")

        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._time = time

    def get(self, api_key: str) -> Any | None:
        """Get cached user_id for API key."""
        # Generate a fast lookup token for the cache
        # (We don't store the key, just a derived token for lookup)
        if api_key is None:
            raise ValueError("api_key must be provided")
        cache_key = self._cache_lookup_token(api_key)

        with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if self._time.time() - timestamp < self._effective_ttl_seconds():
                    return result
                del self._cache[cache_key]
        return None

    def _effective_ttl_seconds(self) -> int:
        """Return the active TTL.

        A monkeypatched class-level ``TTL_SECONDS`` wins over the
        env-resolved instance attribute (preserves test ergonomics);
        otherwise the instance value applies.
        """
        cls_value = type(self).TTL_SECONDS
        if cls_value != _ORIGINAL_TTL_SECONDS:
            return cls_value
        return self._ttl_seconds

    def _effective_max_entries(self) -> int:
        """Return the active max-entries (see :meth:`_effective_ttl_seconds`)."""
        cls_value = type(self).MAX_ENTRIES
        if cls_value != _ORIGINAL_MAX_ENTRIES:
            return cls_value
        return self._max_entries

    def set(self, api_key: str, result: Any) -> None:
        """Cache auth result."""
        if api_key is None:
            raise ValueError("api_key must be provided")
        cache_key = self._cache_lookup_token(api_key)
        with self._lock:
            self._cache.pop(cache_key, None)
            self._evict_overflow_entries()
            self._cache[cache_key] = (result, self._time.time())

    def _evict_overflow_entries(self) -> None:
        """Keep the cache bounded without flushing unrelated auth results."""
        max_entries = self._effective_max_entries()
        while len(self._cache) >= max_entries:
            self._cache.pop(next(iter(self._cache)))

    def _cache_lookup_token(self, token_value: str) -> str:
        """Generate a lookup token for the auth cache.

        SECURITY NOTE: This is NOT used for password/key storage or protection.
        The actual API key verification uses bcrypt (see verify_api_key method).
        This is purely a deterministic dictionary lookup key to avoid repeated
        bcrypt calls across multiple requests.

        We use SHA-256 (not Python's built-in hash()) because:
        1. Python's hash() is randomised per-process (PYTHONHASHSEED), making
           cache keys inconsistent across workers and process restarts.
        2. SHA-256 is deterministic, ensuring cache correctness in multi-worker
           deployments and preventing potential authentication-bypass scenarios.

        The actual security comes from:
        1. Short TTL (5 minutes) limiting exposure window
        2. bcrypt verification on cache miss
        3. The token_value itself is never stored, only this derived lookup key
        """
        if token_value is None:
            raise ValueError("token_value must be provided")
        import hashlib

        # Use SHA-256 for a deterministic, process-stable lookup key.
        # PYTHONHASHSEED does not affect hashlib, so this is safe across workers.
        return hashlib.sha256(token_value.encode()).hexdigest()


# Snapshot the original class-level defaults so we can detect when tests
# monkeypatch them and prefer the patched value over env-resolved values.
_ORIGINAL_TTL_SECONDS = AuthCache.TTL_SECONDS
_ORIGINAL_MAX_ENTRIES = AuthCache.MAX_ENTRIES

auth_cache = AuthCache()
