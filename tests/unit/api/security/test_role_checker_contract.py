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


class TestRoleCheckerContract:
    """Design by Contract tests for RoleChecker class."""

    def test_security_instantiates(self) -> None:
        """Postcondition: RoleChecker can be instantiated."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.PROFESSIONAL)
            assert checker is not None

    def test_is_callable(self) -> None:
        """Postcondition: RoleChecker is callable."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.FREE)
            assert callable(checker)
