"""Tests for security module Design-by-Contract preconditions.

Verifies that precondition decorators reject invalid inputs with
ContractViolationError for all security-critical functions.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# All tests require GOLF_API_SECRET_KEY to be set
ENV_PATCH = {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}


class TestHashPasswordContracts:
    """Test hash_password precondition: password must be non-empty string."""

    def test_empty_password_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import SecurityManager

            mgr = SecurityManager(secret_key="test-secret")
            with pytest.raises((ValueError, AssertionError), match="non-empty"):
                mgr.hash_password("")

    def test_none_password_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import SecurityManager

            mgr = SecurityManager(secret_key="test-secret")
            with pytest.raises((ValueError, AssertionError)):
                mgr.hash_password(None)  # type: ignore[arg-type]


class TestVerifyPasswordContracts:
    """Test verify_password preconditions on both arguments."""

    def test_empty_plain_password_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import SecurityManager

            mgr = SecurityManager(secret_key="test-secret")
            with pytest.raises((ValueError, AssertionError), match="non-empty"):
                mgr.verify_password("", "$2b$12$somehash")

    def test_empty_hashed_password_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import SecurityManager

            mgr = SecurityManager(secret_key="test-secret")
            with pytest.raises((ValueError, AssertionError), match="non-empty"):
                mgr.verify_password("password", "")


class TestCreateAccessTokenContracts:
    """Test create_access_token precondition: data must contain 'sub'."""

    def test_missing_sub_claim_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import SecurityManager

            mgr = SecurityManager(secret_key="test-secret-32-chars-long!!")
            with pytest.raises((ValueError, AssertionError), match="sub"):
                mgr.create_access_token({"email": "test@example.com"})

    def test_non_dict_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import SecurityManager

            mgr = SecurityManager(secret_key="test-secret-32-chars-long!!")
            with pytest.raises((ValueError, AssertionError, TypeError)):
                mgr.create_access_token("not a dict")  # type: ignore[arg-type]


class TestVerifyTokenContracts:
    """Test verify_token preconditions on token and token_type."""

    def test_empty_token_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import SecurityManager

            mgr = SecurityManager(secret_key="test-secret-32-chars-long!!")
            with pytest.raises((ValueError, AssertionError), match="non-empty"):
                mgr.verify_token("")

    def test_invalid_token_type_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import SecurityManager

            mgr = SecurityManager(secret_key="test-secret-32-chars-long!!")
            with pytest.raises((ValueError, AssertionError), match="access.*refresh"):
                mgr.verify_token("some.jwt.token", "invalid_type")


class TestCheckQuotaContracts:
    """Test check_quota precondition on resource_type."""

    def test_invalid_resource_type_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import UsageTracker

            tracker = UsageTracker()
            user = MagicMock()
            user.role = "free"
            with pytest.raises((ValueError, AssertionError), match="resource_type"):
                tracker.check_quota(user, "invalid_resource")


class TestComputePrefixHashContracts:
    """Test compute_prefix_hash precondition: non-empty string."""

    def test_empty_prefix_raises(self) -> None:
        with patch.dict(os.environ, ENV_PATCH):
            from src.api.auth.security import compute_prefix_hash

            with pytest.raises((ValueError, AssertionError), match="non-empty"):
                compute_prefix_hash("")
