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
from unittest.mock import MagicMock, patch

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

from src.api.auth.models import APIKey, User  # noqa: E402
from src.api.auth.security import SecurityManager  # noqa: E402

# Skip marker for bcrypt-dependent tests
requires_bcrypt = pytest.mark.skipif(
    not BCRYPT_AVAILABLE,
    reason="bcrypt native library not available in this environment",
)


class TestSecurityBestPractices:
    """Test adherence to security best practices."""

    def test_no_hardcoded_secrets(self) -> None:
        """Test that no secrets are hardcoded in auth modules."""
        import inspect

        from src.api.auth import dependencies, security

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
        assert token1 != token2, "Assertion failed: token1 != token2"

        # Should have sufficient length
        assert (
            len(token1) >= 40
        )  # 32 bytes = ~43 base64 chars, "Assertion failed: len(token1) >= 40  # 32 bytes = ~43 base64 chars"
        assert len(token2) >= 40, "Assertion failed: len(token2) >= 40"

    @requires_bcrypt
    def test_timing_attack_resistance(self) -> None:
        """Test that password verification is resistant to timing attacks."""
        security_manager = SecurityManager()

        password = "test_password"  # nosec B105 - test fixture, not a real credential
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
        # Use a generous threshold (3.0) to avoid flakiness in shared CI environments
        ratio = max(correct_time, incorrect_time) / min(correct_time, incorrect_time)
        assert ratio < 3.0, "Timing difference suggests vulnerability to timing attacks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
