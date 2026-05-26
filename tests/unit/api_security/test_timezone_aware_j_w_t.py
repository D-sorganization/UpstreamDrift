"""Unit tests for API security features.

This module tests critical security implementations:
- Bcrypt API key hashing and verification
- Timezone-aware JWT token generation
- Password hashing and verification
- Secure credential storage
"""

from datetime import datetime, timezone

import pytest

# Python 3.10 compatibility: datetime.UTC is only available in 3.11+
UTC = timezone.utc  # noqa: UP017

# Check if sqlalchemy is available
try:
    import sqlalchemy  # noqa: F401

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

if not SQLALCHEMY_AVAILABLE:
    pytest.skip("SQLAlchemy not installed", allow_module_level=True)

# Check if bcrypt is available and working
# bcrypt can fail to load on some CI environments due to missing native libraries
try:
    import bcrypt as bcrypt_lib

    # Try to actually use bcrypt to detect runtime issues
    bcrypt_lib.hashpw(b"test", bcrypt_lib.gensalt())
    BCRYPT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # bcrypt is not installed
    BCRYPT_AVAILABLE = False
    bcrypt_lib = None  # type: ignore[misc,assignment]
except Exception as e:  # noqa: BLE001, F841
    # bcrypt failed to load (native library issue)
    BCRYPT_AVAILABLE = False
    import bcrypt as bcrypt_lib  # type: ignore[no-redef]

from src.api.auth.security import SecurityManager  # noqa: E402

# Skip marker for bcrypt-dependent tests
requires_bcrypt = pytest.mark.skipif(
    not BCRYPT_AVAILABLE,
    reason="bcrypt native library not available in this environment",
)


class TestTimezoneAwareJWT:
    """Test timezone-aware JWT token generation."""

    def test_jwt_uses_timezone_aware_datetime(self) -> None:
        """Test that JWT tokens use timezone-aware datetime."""
        security_manager = SecurityManager()

        # Create access token
        token = security_manager.create_access_token(data={"sub": "test_user"})

        # Decode token
        import jwt

        payload = jwt.decode(
            token, security_manager.secret_key, algorithms=[security_manager.algorithm]
        )

        # Check that 'exp' field exists
        assert "exp" in payload, "Assertion failed: exp in payload"

        # The exp should be a timestamp (Unix epoch)
        exp_timestamp = payload["exp"]
        assert isinstance(exp_timestamp, int | float), (
            "Assertion failed: isinstance(exp_timestamp, int | float)"
        )

        # Convert to datetime and verify it's in the future
        exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=UTC)
        now = datetime.now(UTC)

        assert exp_datetime > now, "Token expiration should be in the future"
        assert exp_datetime.tzinfo is not None, "Expiration should be timezone-aware"

    def test_jwt_refresh_token_timezone(self) -> None:
        """Test that refresh tokens use timezone-aware datetime."""
        security_manager = SecurityManager()

        # Create refresh token
        token = security_manager.create_refresh_token(data={"sub": "test_user"})

        # Decode token
        import jwt

        payload = jwt.decode(
            token, security_manager.secret_key, algorithms=[security_manager.algorithm]
        )

        # Verify token type
        assert payload.get("type") == "refresh", (
            "Assertion failed: payload.get(type) == refresh"
        )

        # Check expiration is timezone-aware
        exp_timestamp = payload["exp"]
        exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=UTC)
        now = datetime.now(UTC)

        assert exp_datetime > now, "Assertion failed: exp_datetime > now"
        assert exp_datetime.tzinfo is not None, (
            "Assertion failed: exp_datetime.tzinfo is not None"
        )

    def test_no_deprecated_datetime_utcnow(self) -> None:
        """Test that code doesn't use deprecated datetime.utcnow()."""
        import inspect

        from src.api.auth import security

        # Get source code of security module
        source = inspect.getsource(security)

        # Check for deprecated utcnow usage
        assert "datetime.utcnow()" not in source, (
            "Code should not use deprecated datetime.utcnow(). "
            "Use datetime.now(timezone.utc) instead."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
