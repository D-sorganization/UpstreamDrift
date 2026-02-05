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
def anyio_backend():
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


class TestSecurityManagerContract:
    """Design by Contract tests for SecurityManager class."""

    def test_instantiates(self):
        """Postcondition: SecurityManager can be instantiated."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            assert manager is not None

    def test_has_required_methods(self):
        """Postcondition: SecurityManager has required methods."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            assert hasattr(manager, "hash_password")
            assert hasattr(manager, "verify_password")
            assert hasattr(manager, "create_access_token")
            assert hasattr(manager, "create_refresh_token")
            assert hasattr(manager, "verify_token")
            assert hasattr(manager, "generate_api_key")
            assert hasattr(manager, "hash_api_key")
            assert hasattr(manager, "verify_api_key")


class TestSecurityManagerHashPassword:
    """Tests for SecurityManager.hash_password."""

    def test_returns_string(self):
        """Test that hash_password returns a string."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            result = manager.hash_password("password123")
            assert isinstance(result, str)

    def test_hash_differs_from_input(self):
        """Test that hash differs from input password."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            password = "password123"
            hashed = manager.hash_password(password)
            assert hashed != password

    def test_same_password_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            password = "password123"
            hash1 = manager.hash_password(password)
            hash2 = manager.hash_password(password)
            assert hash1 != hash2  # Different salts


class TestSecurityManagerVerifyPassword:
    """Tests for SecurityManager.verify_password."""

    def test_correct_password_returns_true(self):
        """Test that correct password returns True."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            password = "correct_password"
            hashed = manager.hash_password(password)
            assert manager.verify_password(password, hashed) is True

    def test_wrong_password_returns_false(self):
        """Test that wrong password returns False."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            hashed = manager.hash_password("correct_password")
            assert manager.verify_password("wrong_password", hashed) is False

    def test_invalid_hash_returns_false(self):
        """Test that invalid hash returns False."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            assert manager.verify_password("password", "invalid_hash") is False


class TestSecurityManagerTokens:
    """Tests for SecurityManager token operations."""

    def test_create_access_token_returns_string(self):
        """Test that create_access_token returns a string."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            token = manager.create_access_token({"sub": "user123"})
            assert isinstance(token, str)
            assert len(token) > 0

    def test_create_refresh_token_returns_string(self):
        """Test that create_refresh_token returns a string."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            token = manager.create_refresh_token({"sub": "user123"})
            assert isinstance(token, str)
            assert len(token) > 0

    def test_access_and_refresh_tokens_differ(self):
        """Test that access and refresh tokens are different."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            data = {"sub": "user123"}
            access = manager.create_access_token(data)
            refresh = manager.create_refresh_token(data)
            assert access != refresh

    def test_verify_access_token(self):
        """Test verifying access token."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            data = {"sub": "user123", "email": "test@example.com"}
            token = manager.create_access_token(data)
            payload = manager.verify_token(token, "access")
            assert payload["sub"] == "user123"
            assert payload["email"] == "test@example.com"
            assert payload["type"] == "access"

    def test_verify_refresh_token(self):
        """Test verifying refresh token."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            data = {"sub": "user123"}
            token = manager.create_refresh_token(data)
            payload = manager.verify_token(token, "refresh")
            assert payload["sub"] == "user123"
            assert payload["type"] == "refresh"

    def test_verify_token_wrong_type_raises(self):
        """Test that verifying with wrong type raises HTTPException."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from fastapi import HTTPException

            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            access_token = manager.create_access_token({"sub": "user123"})

            with pytest.raises(HTTPException) as exc_info:
                manager.verify_token(access_token, "refresh")

            assert exc_info.value.status_code == 401

    def test_verify_invalid_token_raises(self):
        """Test that invalid token raises HTTPException."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from fastapi import HTTPException

            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")

            with pytest.raises(HTTPException) as exc_info:
                manager.verify_token("invalid.token.here", "access")

            assert exc_info.value.status_code == 401

    def test_custom_expiration(self):
        """Test creating token with custom expiration."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret-32-chars-long!!")
            token = manager.create_access_token(
                {"sub": "user123"}, expires_delta=timedelta(hours=1)
            )
            payload = manager.verify_token(token, "access")
            assert payload["sub"] == "user123"


class TestSecurityManagerApiKey:
    """Tests for SecurityManager API key operations."""

    def test_generate_api_key_returns_string(self):
        """Test that generate_api_key returns a string."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            assert isinstance(key, str)

    def test_api_key_has_prefix(self):
        """Test that API key has gms_ prefix."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            assert key.startswith("gms_")

    def test_api_keys_are_unique(self):
        """Test that generated API keys are unique."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            keys = {manager.generate_api_key() for _ in range(100)}
            assert len(keys) == 100

    def test_hash_api_key_returns_string(self):
        """Test that hash_api_key returns a string."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            hashed = manager.hash_api_key(key)
            assert isinstance(hashed, str)

    def test_verify_api_key_correct(self):
        """Test verifying correct API key."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            hashed = manager.hash_api_key(key)
            assert manager.verify_api_key(key, hashed) is True

    def test_verify_api_key_wrong(self):
        """Test verifying wrong API key."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            key = manager.generate_api_key()
            hashed = manager.hash_api_key(key)
            assert manager.verify_api_key("wrong_key", hashed) is False


class TestRoleCheckerContract:
    """Design by Contract tests for RoleChecker class."""

    def test_instantiates(self):
        """Postcondition: RoleChecker can be instantiated."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.PROFESSIONAL)
            assert checker is not None

    def test_is_callable(self):
        """Postcondition: RoleChecker is callable."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.FREE)
            assert callable(checker)


class TestRoleChecker:
    """Functional tests for RoleChecker."""

    def test_user_with_exact_role_passes(self):
        """Test user with exact required role passes."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.PROFESSIONAL)
            user = MagicMock()
            user.role = UserRole.PROFESSIONAL.value
            assert checker(user) is True

    def test_user_with_higher_role_passes(self):
        """Test user with higher role passes."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.PROFESSIONAL)
            user = MagicMock()
            user.role = UserRole.ADMIN.value
            assert checker(user) is True

    def test_user_with_lower_role_fails(self):
        """Test user with lower role fails."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.ENTERPRISE)
            user = MagicMock()
            user.role = UserRole.FREE.value
            assert checker(user) is False

    def test_role_hierarchy(self):
        """Test complete role hierarchy."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            # Admin can access everything
            admin_user = MagicMock()
            admin_user.role = UserRole.ADMIN.value

            for role in [UserRole.FREE, UserRole.PROFESSIONAL, UserRole.ENTERPRISE, UserRole.ADMIN]:
                checker = RoleChecker(role)
                assert checker(admin_user) is True

            # Free can only access free
            free_user = MagicMock()
            free_user.role = UserRole.FREE.value

            free_checker = RoleChecker(UserRole.FREE)
            assert free_checker(free_user) is True

            pro_checker = RoleChecker(UserRole.PROFESSIONAL)
            assert pro_checker(free_user) is False


class TestUsageTrackerContract:
    """Design by Contract tests for UsageTracker class."""

    def test_instantiates(self):
        """Postcondition: UsageTracker can be instantiated."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            assert tracker is not None


class TestUsageTracker:
    """Functional tests for UsageTracker."""

    def test_check_quota_within_limit(self):
        """Test check_quota when within limit."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.models import UserRole
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            user = MagicMock()
            user.role = UserRole.FREE.value
            user.api_calls_this_month = 100  # Free tier limit is 1000

            assert tracker.check_quota(user, "api_calls") is True

    def test_check_quota_exceeded(self):
        """Test check_quota when exceeded."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.models import UserRole
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            user = MagicMock()
            user.role = UserRole.FREE.value
            user.api_calls_this_month = 1001  # Exceeds free tier limit of 1000

            assert tracker.check_quota(user, "api_calls") is False

    def test_increment_usage(self):
        """Test incrementing usage counter."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            user = MagicMock()
            user.api_calls_this_month = 10

            tracker.increment_usage(user, "api_calls")
            assert user.api_calls_this_month == 11

    def test_get_usage_summary(self):
        """Test getting usage summary."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.models import UserRole
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            user = MagicMock()
            user.role = UserRole.FREE.value
            user.api_calls_this_month = 100
            user.video_analyses_this_month = 2
            user.simulations_this_month = 5

            summary = tracker.get_usage_summary(user)

            assert summary["subscription_tier"] == "free"
            assert summary["api_calls"]["used"] == 100
            assert summary["api_calls"]["remaining"] == 900
            assert summary["video_analyses"]["used"] == 2
            assert summary["simulations"]["used"] == 5


class TestAuthCacheContract:
    """Design by Contract tests for AuthCache class."""

    def test_instantiates(self):
        """Postcondition: AuthCache can be instantiated."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import AuthCache

            cache = AuthCache()
            assert cache is not None


class TestAuthCache:
    """Functional tests for AuthCache."""

    def test_get_returns_none_for_missing(self):
        """Test get returns None for missing key."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import AuthCache

            cache = AuthCache()
            assert cache.get("nonexistent_key") is None

    def test_set_and_get_round_trip(self):
        """Test set and get work together."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import AuthCache

            cache = AuthCache()
            api_key = "gms_test_key_12345"
            user_id = 42

            cache.set(api_key, user_id)
            result = cache.get(api_key)

            assert result == user_id

    def test_different_keys_cached_separately(self):
        """Test different keys are cached separately."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import AuthCache

            cache = AuthCache()
            cache.set("key1", "value1")
            cache.set("key2", "value2")

            assert cache.get("key1") == "value1"
            assert cache.get("key2") == "value2"


class TestComputePrefixHash:
    """Tests for compute_prefix_hash function."""

    def test_returns_string(self):
        """Test that compute_prefix_hash returns a string."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import compute_prefix_hash

            result = compute_prefix_hash("gms_test")
            assert isinstance(result, str)

    def test_returns_hex_string(self):
        """Test that result is a valid hex string."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import compute_prefix_hash

            result = compute_prefix_hash("gms_test")
            # SHA256 produces 64 hex characters
            assert len(result) == 64
            int(result, 16)  # Should not raise

    def test_same_input_same_output(self):
        """Test deterministic output."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import compute_prefix_hash

            hash1 = compute_prefix_hash("gms_abcd")
            hash2 = compute_prefix_hash("gms_abcd")
            assert hash1 == hash2

    def test_different_input_different_output(self):
        """Test different inputs produce different outputs."""
        with patch.dict(os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}):
            from src.api.auth.security import compute_prefix_hash

            hash1 = compute_prefix_hash("prefix_a")
            hash2 = compute_prefix_hash("prefix_b")
            assert hash1 != hash2
