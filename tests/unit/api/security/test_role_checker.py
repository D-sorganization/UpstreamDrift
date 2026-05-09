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


class TestRoleChecker:
    """Functional tests for RoleChecker."""

    def test_user_with_exact_role_passes(self) -> None:
        """Test user with exact required role passes."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.PROFESSIONAL)
            user = MagicMock()
            user.role = UserRole.PROFESSIONAL.value
            assert checker(user) is True

    def test_user_with_higher_role_passes(self) -> None:
        """Test user with higher role passes."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.PROFESSIONAL)
            user = MagicMock()
            user.role = UserRole.ADMIN.value
            assert checker(user) is True

    def test_user_with_lower_role_fails(self) -> None:
        """Test user with lower role fails."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            checker = RoleChecker(UserRole.ENTERPRISE)
            user = MagicMock()
            user.role = UserRole.FREE.value
            assert checker(user) is False

    def test_role_hierarchy(self) -> None:
        """Test complete role hierarchy."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.models import UserRole
            from src.api.auth.security import RoleChecker

            # Admin can access everything
            admin_user = MagicMock()
            admin_user.role = UserRole.ADMIN.value

            for role in [
                UserRole.FREE,
                UserRole.PROFESSIONAL,
                UserRole.ENTERPRISE,
                UserRole.ADMIN,
            ]:
                checker = RoleChecker(role)
                assert checker(admin_user) is True

            # Free can only access free
            free_user = MagicMock()
            free_user.role = UserRole.FREE.value

            free_checker = RoleChecker(UserRole.FREE)
            assert free_checker(free_user) is True

            pro_checker = RoleChecker(UserRole.PROFESSIONAL)
            assert pro_checker(free_user) is False
