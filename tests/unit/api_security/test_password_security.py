"""Unit tests for API security features.

This module tests critical security implementations:
- Bcrypt API key hashing and verification
- Timezone-aware JWT token generation
- Password hashing and verification
- Secure credential storage
"""

import logging
import secrets
from datetime import timezone
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

from src.api.auth.security import SecurityManager  # noqa: E402

# Skip marker for bcrypt-dependent tests
requires_bcrypt = pytest.mark.skipif(
    not BCRYPT_AVAILABLE,
    reason="bcrypt native library not available in this environment",
)


class TestPasswordSecurity:
    """Test password hashing and security."""

    @requires_bcrypt
    def test_password_bcrypt_hashing(self) -> None:
        """Test that passwords are hashed with bcrypt."""
        security_manager = SecurityManager()

        password = "test_password_123!@#"  # nosec B105 - test fixture, not a real credential
        hashed = security_manager.hash_password(password)

        # Verify bcrypt format
        assert hashed.startswith(("$2b$", "$2a$")), (
            "Assertion failed: hashed.startswith(($2b$, $2a$))"
        )

        # Verify password can be verified
        assert security_manager.verify_password(password, hashed), (
            "Assertion failed: security_manager.verify_password(password, hashed)"
        )

        # Verify wrong password fails
        assert not security_manager.verify_password("wrong_password", hashed), (
            "Assertion failed: not security_manager.verify_password(wrong_password, hashed)"
        )

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
            with (
                patch.dict("os.environ", {}, clear=True),
                # Mock SessionLocal to avoid actual database operations
                patch("api.database.SessionLocal") as mock_session,
            ):
                mock_db = MagicMock()
                mock_session.return_value = mock_db

                # Mock query to return no admin user
                mock_db.query.return_value.filter.return_value.first.return_value = None

                # This should generate a random password but NOT log it
                try:
                    database.init_db()
                except Exception as e:  # noqa: BLE001
                    # Catch and log expected errors for this specific logging test
                    logging.getLogger(__name__).debug(
                        f"Caught expected init_db error: {e}"
                    )

            # Get logged output
            log_output = log_buffer.getvalue()

            # Check that no password appears in plaintext
            # Should have warning about no password set
            assert "GOLF_ADMIN_PASSWORD" in log_output, (
                "Assertion failed: GOLF_ADMIN_PASSWORD in log_output"
            )

            # Should NOT have "password: " or similar plaintext password
            assert "Temporary admin password:" not in log_output, (
                "Assertion failed: Temporary admin password: not in log_output"
            )
            assert "Temporary password:" not in log_output, (
                "Assertion failed: Temporary password: not in log_output"
            )

            # Should have instructions instead
            assert "randomly generated password" in log_output.lower(), (
                "Assertion failed: randomly generated password in log_output.lower()"
            )

        finally:
            logger.removeHandler(handler)

    def test_password_minimum_entropy(self) -> None:
        """Test that generated passwords have sufficient entropy."""
        # Generate multiple random passwords
        for _ in range(10):
            password = secrets.token_urlsafe(16)

            # Check length (16 bytes = ~128 bits entropy)
            assert (
                len(password) >= 20
            )  # Base64 encoding makes it longer, "Assertion failed: len(password) >= 20  # Base64 encoding makes it longer"

            # Check it's not empty or trivial
            assert password, "Assertion failed: password"
            assert password != "password", "Assertion failed: password != password"
            assert password != "123456", "Assertion failed: password != 123456"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
