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


class TestSecurityManagerApiKey:
    """Tests for SecurityManager API key operations."""

    def test_generate_api_key_returns_string(self) -> None:
        """Test that generate_api_key returns a string."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            assert isinstance(key, str)

    def test_api_key_has_prefix(self) -> None:
        """Test that API key has gms_ prefix."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            assert key.startswith("gms_")

    def test_api_keys_are_unique(self) -> None:
        """Test that generated API keys are unique."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            keys = {manager.generate_api_key() for _ in range(100)}
            assert len(keys) == 100

    def test_hash_api_key_returns_string(self) -> None:
        """Test that hash_api_key returns a string."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            hashed = manager.hash_api_key(key)
            assert isinstance(hashed, str)

    def test_verify_api_key_correct(self) -> None:
        """Test verifying correct API key."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            hashed = manager.hash_api_key(key)
            assert manager.verify_api_key(key, hashed) is True

    def test_verify_api_key_wrong(self) -> None:
        """Test verifying wrong API key."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            hashed = manager.hash_api_key(key)
            assert manager.verify_api_key("wrong_key", hashed) is False
