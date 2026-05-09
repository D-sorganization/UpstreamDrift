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


class TestSecurityManagerContract:
    """Design by Contract tests for SecurityManager class."""

    def test_security_instantiates(self) -> None:
        """Postcondition: SecurityManager can be instantiated."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import SecurityManager

            manager = SecurityManager(secret_key="test-secret")
            assert manager is not None

    def test_has_required_methods(self) -> None:
        """Postcondition: SecurityManager has required methods."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
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
