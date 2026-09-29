"""Tests for authentication and authorization security utilities."""

import logging
import os
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.unit

# Patch environment variable before importing security module
os.environ["GOLF_API_SECRET_KEY"] = (
    "super-secret-test-key-must-be-at-least-thirty-two-bytes-long"
)

from src.api.auth.models import User, UserRole
from src.api.auth.security import (
    AuthCache,
    RoleChecker,
    SecurityManager,
    UsageTracker,
    compute_prefix_hash,
)

# Initialize a test manager
security_manager = SecurityManager()


def test_compute_prefix_hash() -> None:
    """Test prefix hashing."""
    prefix = "testpref"
    res1 = compute_prefix_hash(prefix)
    res2 = compute_prefix_hash(prefix)
    assert res1 == res2
    assert len(res1) == 64  # SHA256 hex length


def test_password_hashing() -> None:
    """Test bcrypt password hashing and verification."""
    password = "my_secure_password"  # nosec B105 - test fixture, not a real credential
    hashed = security_manager.hash_password(password)

    assert hashed != password
    assert security_manager.verify_password(password, hashed)
    assert not security_manager.verify_password("wrong_password", hashed)


def test_verify_password_logs_malformed_stored_hash_without_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed stored password hashes fail closed with diagnostic context."""
    password = "my_secure_password"  # nosec B105 - test fixture, not a real credential
    malformed_hash = "not-a-bcrypt-hash"

    with caplog.at_level(logging.WARNING, logger="src.api.auth.security"):
        assert not security_manager.verify_password(password, malformed_hash)

    [record] = caplog.records
    assert record.exc_info is not None
    assert record.operation == "password"
    assert record.exception_type == "ValueError"
    assert "Malformed stored bcrypt hash during password verification" in record.message
    assert password not in record.message
    assert malformed_hash not in record.message


def test_verify_password_wrong_password_stays_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normal credential mismatch is not noisy."""
    hashed = security_manager.hash_password("expected-password")

    with caplog.at_level(logging.WARNING, logger="src.api.auth.security"):
        assert not security_manager.verify_password("wrong-password", hashed)

    assert caplog.records == []


def test_create_access_token() -> None:
    """Test access token creation."""
    data = {"sub": "user_123"}
    token = security_manager.create_access_token(data)
    assert isinstance(token, str)
    assert len(token) > 0


def test_create_access_token_with_expiry() -> None:
    """Test access token creation with specific expiry."""
    data = {"sub": "user_456"}
    token = security_manager.create_access_token(data, timedelta(minutes=5))

    decoded = security_manager.verify_token(token)
    assert decoded["sub"] == "user_456"
    assert decoded["type"] == "access"


def test_create_refresh_token() -> None:
    """Test refresh token creation."""
    data = {"sub": "user_123"}
    token = security_manager.create_refresh_token(data)
    assert isinstance(token, str)

    decoded = security_manager.verify_token(token, token_type="refresh")
    assert decoded["sub"] == "user_123"
    assert decoded["type"] == "refresh"


def test_verify_token_invalid_type() -> None:
    """Test verifying token with wrong type raises HTTPException."""
    data = {"sub": "user"}
    token = security_manager.create_access_token(data)

    with pytest.raises(HTTPException) as excinfo:
        security_manager.verify_token(token, token_type="refresh")
    assert excinfo.value.status_code == 401
    assert "Invalid token type" in str(excinfo.value.detail)


def test_verify_token_expired() -> None:
    """Test verifying an expired token raises HTTPException."""
    data = {"sub": "user"}
    # Create token that expired 1 minute ago
    token = security_manager.create_access_token(data, timedelta(minutes=-1))

    with pytest.raises(HTTPException) as excinfo:
        security_manager.verify_token(token)
    assert excinfo.value.status_code == 401
    assert "Token has expired" in str(excinfo.value.detail)


def test_generate_and_verify_api_key() -> None:
    """Test API key generation and hashing."""
    api_key = security_manager.generate_api_key()
    assert api_key.startswith("gms_")

    hashed = security_manager.hash_api_key(api_key)
    assert security_manager.verify_api_key(api_key, hashed)
    assert not security_manager.verify_api_key("gms_wrong_key", hashed)


def test_verify_api_key_logs_malformed_stored_hash_without_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed stored API-key hashes fail closed with diagnostic context."""
    api_key = "gms_test_api_key"
    malformed_hash = "not-a-bcrypt-hash"

    with caplog.at_level(logging.WARNING, logger="src.api.auth.security"):
        assert not security_manager.verify_api_key(api_key, malformed_hash)

    [record] = caplog.records
    assert record.exc_info is not None
    assert record.operation == "api_key"
    assert record.exception_type == "ValueError"
    assert "Malformed stored bcrypt hash during API key verification" in record.message
    assert api_key not in record.message
    assert malformed_hash not in record.message


def test_verify_api_key_wrong_key_stays_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normal API-key mismatch is not noisy."""
    hashed = security_manager.hash_api_key("gms_expected_key")

    with caplog.at_level(logging.WARNING, logger="src.api.auth.security"):
        assert not security_manager.verify_api_key("gms_wrong_key", hashed)

    assert caplog.records == []


def test_role_checker() -> None:
    """Test role checker privileges."""
    check_admin = RoleChecker(UserRole.ADMIN)
    check_pro = RoleChecker(UserRole.PROFESSIONAL)

    admin_user = MagicMock(spec=User)
    admin_user.role = "admin"

    free_user = MagicMock(spec=User)
    free_user.role = "free"

    pro_user = MagicMock(spec=User)
    pro_user.role = "professional"

    assert check_admin(admin_user)
    assert not check_admin(free_user)
    assert not check_admin(pro_user)

    assert check_pro(admin_user)
    assert check_pro(pro_user)
    assert not check_pro(free_user)


def test_usage_tracker() -> None:
    """Test quota enforcement."""
    tracker = UsageTracker()

    user = MagicMock(spec=User)
    user.role = "free"
    user.api_calls_this_month = 0
    user.video_analyses_this_month = 0
    user.simulations_this_month = 0

    # Should have quota
    assert tracker.check_quota(user, "api_calls")

    # Increment
    tracker.increment_usage(user, "api_calls")
    assert user.api_calls_this_month == 1

    # Check summary
    summary = tracker.get_usage_summary(user)
    assert summary["api_calls"]["used"] == 1


def test_auth_cache() -> None:
    """Test AuthCache functionality."""
    cache = AuthCache()
    key = "test_key"
    mock_user = MagicMock()

    assert cache.get(key) is None

    cache.set(key, mock_user)
    assert cache.get(key) is mock_user

    # Test token is deterministic
    token1 = cache._cache_lookup_token("token A")
    token2 = cache._cache_lookup_token("token B")
    token3 = cache._cache_lookup_token("token A")
    assert token1 != token2
    assert token1 == token3
