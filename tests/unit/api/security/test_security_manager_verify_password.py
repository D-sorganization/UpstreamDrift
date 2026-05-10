"""Tests for security - Authentication and authorization utilities.

These tests verify the security module using Design by Contract principles.
"""

import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

# Configure async tests to use asyncio backend only
pytestmark = pytest.mark.anyio


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
