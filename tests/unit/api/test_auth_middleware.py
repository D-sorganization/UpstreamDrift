"""Tests for auth/middleware - Authentication middleware.

These tests verify the authentication middleware using Design by Contract principles.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestIsLocalModeContract:
    """Design by Contract tests for is_local_mode function."""

    def test_returns_bool(self):
        """Postcondition: Returns a boolean."""
        from src.api.auth.middleware import is_local_mode

        result = is_local_mode()
        assert isinstance(result, bool)


class TestIsLocalMode:
    """Functional tests for is_local_mode."""

    def test_returns_true_when_golf_suite_mode_local(self):
        """Test returning True when GOLF_SUITE_MODE=local."""
        from src.api.auth.middleware import is_local_mode

        with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}):
            assert is_local_mode() is True

    def test_returns_true_when_auth_disabled(self):
        """Test returning True when GOLF_AUTH_DISABLED=true."""
        from src.api.auth.middleware import is_local_mode

        with patch.dict(
            os.environ, {"GOLF_SUITE_MODE": "cloud", "GOLF_AUTH_DISABLED": "true"}
        ):
            assert is_local_mode() is True

    def test_returns_true_when_auth_disabled_uppercase(self):
        """Test returning True when GOLF_AUTH_DISABLED=TRUE (case insensitive)."""
        from src.api.auth.middleware import is_local_mode

        with patch.dict(
            os.environ, {"GOLF_SUITE_MODE": "cloud", "GOLF_AUTH_DISABLED": "TRUE"}
        ):
            assert is_local_mode() is True

    def test_returns_false_when_cloud_mode(self):
        """Test returning False when GOLF_SUITE_MODE=cloud."""
        from src.api.auth.middleware import is_local_mode

        with patch.dict(
            os.environ,
            {"GOLF_SUITE_MODE": "cloud", "GOLF_AUTH_DISABLED": "false"},
            clear=True,
        ):
            assert is_local_mode() is False

    def test_defaults_to_local_mode(self):
        """Test defaulting to local mode when env vars not set."""
        from src.api.auth.middleware import is_local_mode

        with patch.dict(os.environ, {}, clear=True):
            # When not set, defaults to "local"
            assert is_local_mode() is True


class TestLocalUserContract:
    """Design by Contract tests for LocalUser class."""

    def test_instantiates(self):
        """Postcondition: LocalUser can be instantiated."""
        from src.api.auth.middleware import LocalUser

        user = LocalUser()
        assert user is not None

    def test_has_id(self):
        """Postcondition: LocalUser has id attribute."""
        from src.api.auth.middleware import LocalUser

        user = LocalUser()
        assert hasattr(user, "id")
        assert user.id == "local-user"

    def test_has_email(self):
        """Postcondition: LocalUser has email attribute."""
        from src.api.auth.middleware import LocalUser

        user = LocalUser()
        assert hasattr(user, "email")
        assert user.email == "local@localhost"

    def test_has_role(self):
        """Postcondition: LocalUser has admin role."""
        from src.api.auth.middleware import LocalUser

        user = LocalUser()
        assert hasattr(user, "role")
        assert user.role == "ADMIN"

    def test_has_unlimited_quota(self):
        """Postcondition: LocalUser has unlimited quota."""
        from src.api.auth.middleware import LocalUser

        user = LocalUser()
        assert hasattr(user, "quota_remaining")
        assert user.quota_remaining == float("inf")


class TestLocalUserHasPermission:
    """Tests for LocalUser.has_permission method."""

    def test_has_permission_method(self):
        """Postcondition: LocalUser has has_permission method."""
        from src.api.auth.middleware import LocalUser

        user = LocalUser()
        assert hasattr(user, "has_permission")
        assert callable(user.has_permission)

    def test_always_returns_true(self):
        """Postcondition: has_permission always returns True for local user."""
        from src.api.auth.middleware import LocalUser

        user = LocalUser()
        assert user.has_permission("any_permission") is True
        assert user.has_permission("admin") is True
        assert user.has_permission("delete_everything") is True


class TestOptionalAuthContract:
    """Design by Contract tests for OptionalAuth class."""

    def test_inherits_from_http_bearer(self):
        """Postcondition: OptionalAuth inherits from HTTPBearer."""
        from fastapi.security import HTTPBearer

        from src.api.auth.middleware import OptionalAuth

        assert issubclass(OptionalAuth, HTTPBearer)

    def test_instantiates(self):
        """Postcondition: OptionalAuth can be instantiated."""
        from src.api.auth.middleware import OptionalAuth

        auth = OptionalAuth()
        assert auth is not None

    def test_accepts_auto_error_parameter(self):
        """Postcondition: Accepts auto_error parameter."""
        from src.api.auth.middleware import OptionalAuth

        auth_with_error = OptionalAuth(auto_error=True)
        auth_without_error = OptionalAuth(auto_error=False)
        assert auth_with_error is not None
        assert auth_without_error is not None


# Configure async tests
pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend():
    """Use asyncio backend only."""
    return "asyncio"


class TestOptionalAuthCall:
    """Tests for OptionalAuth.__call__ method."""

    async def test_returns_local_user_in_local_mode(self):
        """Test returning LocalUser in local mode."""
        from src.api.auth.middleware import LocalUser, OptionalAuth

        mock_request = MagicMock()

        with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}):
            auth = OptionalAuth()
            result = await auth(mock_request)

            assert isinstance(result, LocalUser)
            assert result.role == "ADMIN"

    async def test_calls_parent_in_cloud_mode(self):
        """Test calling parent HTTPBearer in cloud mode."""
        from src.api.auth.middleware import OptionalAuth

        mock_request = MagicMock()
        mock_credentials = MagicMock()

        with (
            patch.dict(
                os.environ,
                {"GOLF_SUITE_MODE": "cloud", "GOLF_AUTH_DISABLED": "false"},
                clear=True,
            ),
            patch(
                "fastapi.security.HTTPBearer.__call__",
                return_value=mock_credentials,
            ) as mock_parent,
        ):
            auth = OptionalAuth()
            result = await auth(mock_request)

            mock_parent.assert_called_once_with(mock_request)
            assert result == mock_credentials
