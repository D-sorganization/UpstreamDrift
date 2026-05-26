"""Tests for security - Authentication and authorization utilities.

These tests verify the security module using Design by Contract principles.
"""

import os
from unittest.mock import patch

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
            api_key = "gms_test_key_12345"  # nosec B105 - test fixture, not a real credential
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

    def test_overflow_evicts_oldest_entry_without_flushing_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test cache overflow evicts the oldest entry instead of clearing all entries."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import AuthCache

            monkeypatch.setattr(AuthCache, "MAX_ENTRIES", 3)
            cache = AuthCache()

            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.set("key3", "value3")
            cache.set("key4", "value4")

            assert cache.get("key1") is None
            assert cache.get("key2") == "value2"
            assert cache.get("key3") == "value3"
            assert cache.get("key4") == "value4"

    def test_constructor_args_override_class_defaults(self) -> None:
        """Explicit constructor args take precedence over class-level defaults."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import AuthCache

            cache = AuthCache(ttl_seconds=600, max_entries=42)
            assert cache._ttl_seconds == 600
            assert cache._max_entries == 42

    def test_env_vars_tune_cache_sizing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GOLF_AUTH_CACHE_TTL_SECONDS / MAX_ENTRIES tune the cache without code edits."""
        monkeypatch.setenv("GOLF_API_SECRET_KEY", "test-secret-key-32chars-long!!")
        monkeypatch.setenv("GOLF_AUTH_CACHE_TTL_SECONDS", "60")
        monkeypatch.setenv("GOLF_AUTH_CACHE_MAX_ENTRIES", "5")

        from src.api.auth.security import AuthCache

        cache = AuthCache()
        assert cache._ttl_seconds == 60
        assert cache._max_entries == 5

    def test_invalid_constructor_args_rejected(self) -> None:
        """DbC: non-positive TTL / max-entries are rejected."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import AuthCache

            with pytest.raises(ValueError, match="ttl_seconds"):
                AuthCache(ttl_seconds=0)
            with pytest.raises(ValueError, match="max_entries"):
                AuthCache(max_entries=0)
