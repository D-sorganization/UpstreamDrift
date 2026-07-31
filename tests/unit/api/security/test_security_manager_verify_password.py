"""Tests for security - Authentication and authorization utilities.

These tests verify the security module using Design by Contract principles.
"""

import os
from unittest.mock import patch

import pytest

# Configure async tests to use asyncio backend only
pytestmark = [pytest.mark.unit, pytest.mark.anyio]


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


class TestSecurityManagerVerifyPassword:
    """Tests for SecurityManager.verify_password."""

    def test_correct_password_returns_true(self) -> None:
        """Test that correct password returns True."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            password = (
                "correct_password"  # nosec B105 - test fixture, not a real credential
            )
            hashed = manager.hash_password(password)
            assert manager.verify_password(password, hashed) is True

    def test_wrong_password_returns_false(self) -> None:
        """Test that wrong password returns False."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            hashed = manager.hash_password("correct_password")
            assert manager.verify_password("wrong_password", hashed) is False

    def test_invalid_hash_returns_false(self) -> None:
        """Test that invalid hash returns False."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            assert manager.verify_password("password", "invalid_hash") is False

    def test_malformed_hash_logs_before_returning_false(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A corrupt stored hash must leave a log trace (issue #7700)."""
        import logging

        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            with caplog.at_level(logging.WARNING, logger="src.api.auth.security"):
                # A truncated, non-bcrypt stored hash makes bcrypt.checkpw raise
                # ValueError, which the verify path swallows into False.
                result = manager.verify_password("password", "not-a-bcrypt-hash")

        assert result is False
        assert any(
            "Password verification failed" in record.getMessage()
            and record.levelno >= logging.WARNING
            for record in caplog.records
        ), "expected a logged trace for the malformed stored hash"

    def test_wrong_password_stays_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A normal wrong-password verification emits no log record."""
        import logging

        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            hashed = manager.hash_password("correct_password")
            with caplog.at_level(logging.WARNING, logger="src.api.auth.security"):
                result = manager.verify_password("wrong_password", hashed)

        assert result is False
        assert not [
            record
            for record in caplog.records
            if "verification failed" in record.getMessage()
        ], "wrong-password path must stay silent"
