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


class TestAuthCache:
    """Functional tests for AuthCache."""

    def test_get_returns_none_for_missing(self) -> None:
        """Test get returns None for missing key."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import AuthCache

            cache = AuthCache()
            assert cache.get("nonexistent_key") is None

    def test_set_and_get_round_trip(self) -> None:
        """Test set and get work together."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import AuthCache

            cache = AuthCache()
            api_key = (
                "gms_test_key_12345"  # nosec B105 - test fixture, not a real credential
            )
            user_id = 42

            cache.set(api_key, user_id)
            result = cache.get(api_key)

            assert result == user_id

    def test_different_keys_cached_separately(self) -> None:
        """Test different keys are cached separately."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import AuthCache

            cache = AuthCache()
            cache.set("key1", "value1")
            cache.set("key2", "value2")

            assert cache.get("key1") == "value1"
            assert cache.get("key2") == "value2"
