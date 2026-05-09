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


class TestSecurityManagerHashPassword:
    """Tests for SecurityManager.hash_password."""

    def test_security_returns_string(self) -> None:
        """Test that hash_password returns a string."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            result = manager.hash_password("password123")
            assert isinstance(result, str)

    def test_hash_differs_from_input(self) -> None:
        """Test that hash differs from input password."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            password = "password123"  # nosec B105 - test fixture, not a real credential
            hashed = manager.hash_password(password)
            assert hashed != password

    def test_same_password_different_hashes(self) -> None:
        """Test that same password produces different hashes (salt)."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            password = "password123"  # nosec B105 - test fixture, not a real credential
            hash1 = manager.hash_password(password)
            hash2 = manager.hash_password(password)
            assert hash1 != hash2  # Different salts
