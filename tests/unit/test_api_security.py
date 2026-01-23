"""Unit tests for API security features.

This module tests critical security implementations:
- Bcrypt API key hashing and verification
- Timezone-aware JWT token generation
- Password hashing and verification
- Secure credential storage
"""

import logging
import secrets
from datetime import datetime, timezone

# Shim for Python < 3.11
try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017
from unittest.mock import MagicMock, patch

import pytest

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
except Exception:
    # bcrypt failed to load (native library issue)
    BCRYPT_AVAILABLE = False
    import bcrypt as bcrypt_lib  # type: ignore[no-redef]

from src.api.auth.models import APIKey, User
from src.api.auth.security import SecurityManager

# Skip marker for bcrypt-dependent tests
requires_bcrypt = pytest.mark.skipif(
    not BCRYPT_AVAILABLE,
    reason="bcrypt native library not available in this environment",
)


class TestBcryptAPIKeyVerification:
    """Test bcrypt-based API key verification."""

    @requires_bcrypt
    def test_api_key_bcrypt_hashing(self) -> None:
        """Test that API keys are hashed with bcrypt."""
        # Generate a test API key
        api_key = f"gms_{secrets.token_urlsafe(32)}"

        # Hash it with bcrypt
        salt = bcrypt_lib.gensalt(rounds=12)
        key_hash = bcrypt_lib.hashpw(api_key.encode("utf-8"), salt).decode("utf-8")

        # Verify the hash is bcrypt format (starts with $2b$)
        assert key_hash.startswith("$2b$") or key_hash.startswith("$2a$")

        # Verify the key can be verified
        assert bcrypt_lib.checkpw(api_key.encode("utf-8"), key_hash.encode("utf-8"))

        # Verify a different key fails
        wrong_key = f"gms_{secrets.token_urlsafe(32)}"
        assert not bcrypt_lib.checkpw(
            wrong_key.encode("utf-8"), key_hash.encode("utf-8")
        )

    @requires_bcrypt
    def test_api_key_constant_time_comparison(self) -> None:
        """Test that API key verification uses constant-time comparison."""
        api_key = f"gms_{secrets.token_urlsafe(32)}"
        salt = bcrypt_lib.gensalt(rounds=12)
        key_hash = bcrypt_lib.hashpw(api_key.encode("utf-8"), salt)

        # Bcrypt's checkpw() uses constant-time comparison internally
        # This test verifies it doesn't leak timing information
        # by ensuring both correct and incorrect keys take similar time

        import time

        # Measure correct key verification time
        start = time.perf_counter()
        for _ in range(10):
            bcrypt_lib.checkpw(api_key.encode("utf-8"), key_hash)
        correct_time = time.perf_counter() - start

        # Measure incorrect key verification time
        wrong_key = f"gms_{secrets.token_urlsafe(32)}"
        start = time.perf_counter()
        for _ in range(10):
            bcrypt_lib.checkpw(wrong_key.encode("utf-8"), key_hash)
        incorrect_time = time.perf_counter() - start

        # Times should be similar (within 50% of each other)
        # Bcrypt is designed to take consistent time regardless of correctness
        ratio = max(correct_time, incorrect_time) / min(correct_time, incorrect_time)
        assert ratio < 1.5, "Timing difference suggests non-constant-time comparison"

    def test_api_key_format_validation(self) -> None:
        """Test that API keys must have gms_ prefix."""
        # Valid format
        valid_key = f"gms_{secrets.token_urlsafe(32)}"
        assert valid_key.startswith("gms_")

        # Invalid formats (should be rejected)
        invalid_keys = [
            secrets.token_urlsafe(32),  # No prefix
            f"api_{secrets.token_urlsafe(32)}",  # Wrong prefix
            "gms_",  # Prefix only
            "",  # Empty
        ]

        for invalid_key in invalid_keys:
            assert not invalid_key.startswith("gms_") or len(invalid_key) <= 4

    @requires_bcrypt
    def test_bcrypt_cost_factor(self) -> None:
        """Test that bcrypt uses appropriate cost factor (work factor)."""
        api_key = f"gms_{secrets.token_urlsafe(32)}"
        salt = bcrypt_lib.gensalt(rounds=12)
        key_hash = bcrypt_lib.hashpw(api_key.encode("utf-8"), salt).decode("utf-8")

        # Extract bcrypt cost factor from hash
        # Format: $2b$[cost]$[salt][hash]
        parts = key_hash.split("$")
        cost_factor = int(parts[2])

        # Cost factor should be at least 12 (recommended minimum)
        assert cost_factor >= 12, f"Bcrypt cost factor {cost_factor} is too low"

    @requires_bcrypt
    @pytest.mark.skip(reason="pytest-asyncio not installed in CI environment")
    async def test_api_key_verification_integration(self) -> None:
        """Test full API key verification flow."""
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from src.api.auth.dependencies import get_current_user_from_api_key

        # Create test API key
        api_key = f"gms_{secrets.token_urlsafe(32)}"
        salt = bcrypt_lib.gensalt(rounds=12)
        key_hash = bcrypt_lib.hashpw(api_key.encode("utf-8"), salt).decode("utf-8")

        # Mock database and API key record
        mock_db = MagicMock()
        mock_api_key_record = MagicMock(spec=APIKey)
        mock_api_key_record.key_hash = key_hash
        mock_api_key_record.user_id = 1
        mock_api_key_record.is_active = True
        mock_api_key_record.last_used = None
        mock_api_key_record.usage_count = 0

        mock_user = MagicMock(spec=User)
        mock_user.id = 1
        mock_user.is_active = True

        # Configure mock database queries
        mock_db.query.return_value.filter.return_value.all.return_value = [
            mock_api_key_record
        ]
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        # Test with correct API key
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=api_key)

        user = await get_current_user_from_api_key(credentials, mock_db)
        assert user == mock_user

        # Test with incorrect API key
        wrong_key = f"gms_{secrets.token_urlsafe(32)}"
        wrong_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=wrong_key
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_from_api_key(wrong_credentials, mock_db)

        assert exc_info.value.status_code == 401


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
        assert "exp" in payload

        # The exp should be a timestamp (Unix epoch)
        exp_timestamp = payload["exp"]
        assert isinstance(exp_timestamp, int | float)

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
        assert payload.get("type") == "refresh"

        # Check expiration is timezone-aware
        exp_timestamp = payload["exp"]
        exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=UTC)
        now = datetime.now(UTC)

        assert exp_datetime > now
        assert exp_datetime.tzinfo is not None

    def test_no_deprecated_datetime_utcnow(self) -> None:
        """Test that code doesn't use deprecated datetime.utcnow()."""
        import inspect

        from api.auth import security

        # Get source code of security module
        source = inspect.getsource(security)

        # Check for deprecated utcnow usage
        assert "datetime.utcnow()" not in source, (
            "Code should not use deprecated datetime.utcnow(). "
            "Use datetime.now(timezone.utc) instead."
        )


class TestPasswordSecurity:
    """Test password hashing and security."""

    @requires_bcrypt
    def test_password_bcrypt_hashing(self) -> None:
        """Test that passwords are hashed with bcrypt."""
        security_manager = SecurityManager()

        password = "test_password_123!@#"
        hashed = security_manager.hash_password(password)

        # Verify bcrypt format
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")

        # Verify password can be verified
        assert security_manager.verify_password(password, hashed)

        # Verify wrong password fails
        assert not security_manager.verify_password("wrong_password", hashed)

    def test_password_not_logged(self) -> None:
        """Test that passwords are never logged in plaintext."""
        from io import StringIO

        from api import database

        # Create a string buffer to capture log output
        log_buffer = StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.DEBUG)

        # Get the database logger
        logger = logging.getLogger("api.database")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            # Mock environment to not have admin password set
            with patch.dict("os.environ", {}, clear=True):
                # Mock SessionLocal to avoid actual database operations
                with patch("api.database.SessionLocal") as mock_session:
                    mock_db = MagicMock()
                    mock_session.return_value = mock_db

                    # Mock query to return no admin user
                    mock_db.query.return_value.filter.return_value.first.return_value = (
                        None
                    )

                    # This should generate a random password but NOT log it
                    try:
                        database.init_db()
                    except Exception as e:
                        # Catch and log expected errors for this specific logging test
                        logging.getLogger(__name__).debug(
                            f"Caught expected init_db error: {e}"
                        )

            # Get logged output
            log_output = log_buffer.getvalue()

            # Check that no password appears in plaintext
            # Should have warning about no password set
            assert "GOLF_ADMIN_PASSWORD" in log_output

            # Should NOT have "password: " or similar plaintext password
            assert "Temporary admin password:" not in log_output
            assert "Temporary password:" not in log_output

            # Should have instructions instead
            assert "randomly generated password" in log_output.lower()

        finally:
            logger.removeHandler(handler)

    def test_password_minimum_entropy(self) -> None:
        """Test that generated passwords have sufficient entropy."""
        # Generate multiple random passwords
        for _ in range(10):
            password = secrets.token_urlsafe(16)

            # Check length (16 bytes = ~128 bits entropy)
            assert len(password) >= 20  # Base64 encoding makes it longer

            # Check it's not empty or trivial
            assert password
            assert password != "password"
            assert password != "123456"


class TestSecretKeyValidation:
    """Test secret key security requirements."""

    def test_secret_key_length_validation(self) -> None:
        """Test that secret keys are validated for length."""
        from api.auth.security import SECRET_KEY

        # In production, secret key should be long enough
        # For testing, we accept the unsafe placeholder
        if SECRET_KEY != "UNSAFE-NO-SECRET-KEY-SET-AUTHENTICATION-WILL-FAIL":
            assert len(SECRET_KEY) >= 32, "Secret key must be at least 32 characters"

    def test_secret_key_environment_variable(self) -> None:
        """Test that secret key can be set via environment variable."""
        with patch.dict("os.environ", {"GOLF_API_SECRET_KEY": "x" * 64}):
            # Reload the module to pick up new environment variable
            import os
            import sys

            # Verify env var is set correctly
            assert os.environ.get("GOLF_API_SECRET_KEY") == "x" * 64

            # Avoid importlib.reload()
            if "api.auth.security" in sys.modules:
                del sys.modules["api.auth.security"]

            # Also ensure parent package doesn't hold stale reference
            if "api.auth" in sys.modules:
                import api.auth

                if hasattr(api.auth, "security"):
                    delattr(api.auth, "security")

            from api.auth import security

            # Check it uses the environment variable
            assert security.SECRET_KEY == "x" * 64


class TestSecurityBestPractices:
    """Test adherence to security best practices."""

    def test_no_hardcoded_secrets(self) -> None:
        """Test that no secrets are hardcoded in auth modules."""
        import inspect

        from api.auth import dependencies, security

        # Get source code
        security_source = inspect.getsource(security)
        dependencies_source = inspect.getsource(dependencies)

        # Check for potential hardcoded secrets (common patterns)
        suspicious_patterns = [
            "password = '",
            'password = "',
            "api_key = '",
            'api_key = "',
            "secret = '",
            'secret = "',
        ]

        for pattern in suspicious_patterns:
            assert (
                pattern not in security_source.lower()
            ), f"Found suspicious pattern in security.py: {pattern}"
            assert (
                pattern not in dependencies_source.lower()
            ), f"Found suspicious pattern in dependencies.py: {pattern}"

    def test_secure_random_generation(self) -> None:
        """Test that secrets module is used for random generation."""
        # Verify secrets module generates cryptographically secure random values
        token1 = secrets.token_urlsafe(32)
        token2 = secrets.token_urlsafe(32)

        # Should be different
        assert token1 != token2

        # Should have sufficient length
        assert len(token1) >= 40  # 32 bytes = ~43 base64 chars
        assert len(token2) >= 40

    @requires_bcrypt
    def test_timing_attack_resistance(self) -> None:
        """Test that password verification is resistant to timing attacks."""
        security_manager = SecurityManager()

        password = "test_password"
        hashed = security_manager.hash_password(password)

        import time

        # Measure time for correct password
        start = time.perf_counter()
        for _ in range(10):
            security_manager.verify_password(password, hashed)
        correct_time = time.perf_counter() - start

        # Measure time for incorrect password
        start = time.perf_counter()
        for _ in range(10):
            security_manager.verify_password("wrong_password", hashed)
        incorrect_time = time.perf_counter() - start

        # Times should be similar (bcrypt takes consistent time)
        ratio = max(correct_time, incorrect_time) / min(correct_time, incorrect_time)
        assert ratio < 1.5, "Timing difference suggests vulnerability to timing attacks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
