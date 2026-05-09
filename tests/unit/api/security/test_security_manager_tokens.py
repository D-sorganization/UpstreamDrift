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


class TestSecurityManagerTokens:
    """Tests for SecurityManager token operations."""

    def test_create_access_token_returns_string(self) -> None:
        """Test that create_access_token returns a string."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            token = manager.create_access_token({"sub": "user123"})
            assert isinstance(token, str)
            assert len(token) > 0

    def test_create_refresh_token_returns_string(self) -> None:
        """Test that create_refresh_token returns a string."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            token = manager.create_refresh_token({"sub": "user123"})
            assert isinstance(token, str)
            assert len(token) > 0

    def test_access_and_refresh_tokens_differ(self) -> None:
        """Test that access and refresh tokens are different."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            data = {"sub": "user123"}
            access = manager.create_access_token(data)
            refresh = manager.create_refresh_token(data)
            assert access != refresh

    def test_verify_access_token(self) -> None:
        """Test verifying access token."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            data = {"sub": "user123", "email": "test@example.com"}
            token = manager.create_access_token(data)
            payload = manager.verify_token(token, "access")
            assert payload["sub"] == "user123"
            assert payload["email"] == "test@example.com"
            assert payload["type"] == "access"

    def test_verify_refresh_token(self) -> None:
        """Test verifying refresh token."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            data = {"sub": "user123"}
            token = manager.create_refresh_token(data)
            payload = manager.verify_token(token, "refresh")
            assert payload["sub"] == "user123"
            assert payload["type"] == "refresh"

    def test_verify_token_wrong_type_raises(self) -> None:
        """Test that verifying with wrong type raises HTTPException."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from fastapi import HTTPException
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            access_token = manager.create_access_token({"sub": "user123"})

            with pytest.raises(HTTPException) as exc_info:
                manager.verify_token(access_token, "refresh")

            assert exc_info.value.status_code == 401

    def test_verify_invalid_token_raises(self) -> None:
        """Test that invalid token raises HTTPException."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from fastapi import HTTPException
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")

            with pytest.raises(HTTPException) as exc_info:
                manager.verify_token("invalid.token.here", "access")

            assert exc_info.value.status_code == 401

    def test_custom_expiration(self) -> None:
        """Test creating token with custom expiration."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            token = manager.create_access_token(
                {"sub": "user123"}, expires_delta=timedelta(hours=1)
            )
            payload = manager.verify_token(token, "access")
            assert payload["sub"] == "user123"
