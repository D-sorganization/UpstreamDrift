"""Unit tests for production-credential safety check (issue #5920).

Verifies that ``_assert_production_secrets`` hard-fails when the server would
start in production mode with the exact placeholder values from .env.example.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROD_ENV = {"ENVIRONMENT": "production"}
_DEV_ENV = {"ENVIRONMENT": "development"}

_DEFAULT_SECRET_KEY = "generate_a_random_string_here"
_DEFAULT_ADMIN_PASSWORD = "change_me_in_production"
_SAFE_SECRET_KEY = "a-sufficiently-long-and-random-production-secret-key-xyz-12345"
_SAFE_ADMIN_PASSWORD = "S3cur3P@ssw0rd!XYZ"


def _call(env: dict[str, str]) -> None:
    """Import and call _assert_production_secrets with a clean env patch."""
    # Clear the functools.cache on get_environment so env patches take effect.
    from src.shared.python.config import environment as env_mod

    env_mod.get_environment.cache_clear()
    try:
        from src.shared.python.security import env_validator

        importlib.reload(env_validator)
        with patch.dict(os.environ, env, clear=True):
            env_mod.get_environment.cache_clear()
            env_validator._assert_production_secrets()
    finally:
        env_mod.get_environment.cache_clear()


# ---------------------------------------------------------------------------
# Tests: production mode with default credentials - must raise
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssertProductionSecretsRaises:
    """_assert_production_secrets must RuntimeError in production with defaults."""

    def test_raises_with_default_secret_key_in_production(self) -> None:
        """RuntimeError when GOLF_API_SECRET_KEY is the .env.example placeholder."""
        env = {
            **_PROD_ENV,
            "GOLF_API_SECRET_KEY": _DEFAULT_SECRET_KEY,
            "GOLF_ADMIN_PASSWORD": _SAFE_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            from src.shared.python.security.env_validator import (
                _assert_production_secrets,
            )

            with pytest.raises(RuntimeError, match="GOLF_API_SECRET_KEY"):
                _assert_production_secrets()

    def test_raises_with_default_admin_password_in_production(self) -> None:
        """RuntimeError when GOLF_ADMIN_PASSWORD is the .env.example placeholder."""
        env = {
            **_PROD_ENV,
            "GOLF_API_SECRET_KEY": _SAFE_SECRET_KEY,
            "GOLF_ADMIN_PASSWORD": _DEFAULT_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            from src.shared.python.security.env_validator import (
                _assert_production_secrets,
            )

            with pytest.raises(RuntimeError, match="GOLF_ADMIN_PASSWORD"):
                _assert_production_secrets()

    def test_error_message_references_env_example(self) -> None:
        """Error message should mention .env.example so ops can find docs."""
        env = {
            **_PROD_ENV,
            "GOLF_API_SECRET_KEY": _DEFAULT_SECRET_KEY,
            "GOLF_ADMIN_PASSWORD": _SAFE_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            from src.shared.python.security.env_validator import (
                _assert_production_secrets,
            )

            with pytest.raises(RuntimeError, match=".env.example"):
                _assert_production_secrets()


# ---------------------------------------------------------------------------
# Tests: development mode - must NOT raise even with default credentials
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssertProductionSecretsDevMode:
    """_assert_production_secrets must be a no-op outside of production."""

    def test_no_raise_in_development_with_default_secret_key(self) -> None:
        """Development mode: default GOLF_API_SECRET_KEY does not raise."""
        env = {
            **_DEV_ENV,
            "GOLF_API_SECRET_KEY": _DEFAULT_SECRET_KEY,
            "GOLF_ADMIN_PASSWORD": _DEFAULT_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            from src.shared.python.security.env_validator import (
                _assert_production_secrets,
            )

            _assert_production_secrets()  # should not raise

    def test_no_raise_in_development_with_default_admin_password(self) -> None:
        """Development mode: default GOLF_ADMIN_PASSWORD does not raise."""
        env = {
            **_DEV_ENV,
            "GOLF_API_SECRET_KEY": _DEFAULT_SECRET_KEY,
            "GOLF_ADMIN_PASSWORD": _DEFAULT_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            from src.shared.python.security.env_validator import (
                _assert_production_secrets,
            )

            _assert_production_secrets()  # should not raise

    def test_no_raise_when_environment_unset(self) -> None:
        """Unset ENVIRONMENT defaults to development - must not raise."""
        env = {
            "GOLF_API_SECRET_KEY": _DEFAULT_SECRET_KEY,
            "GOLF_ADMIN_PASSWORD": _DEFAULT_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            from src.shared.python.security.env_validator import (
                _assert_production_secrets,
            )

            _assert_production_secrets()  # should not raise


# ---------------------------------------------------------------------------
# Tests: production mode with proper credentials - must NOT raise
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssertProductionSecretsPassesWithSafeCredentials:
    """_assert_production_secrets must pass silently in production with real creds."""

    def test_no_raise_in_production_with_safe_credentials(self) -> None:
        """Production mode with proper credentials: no exception raised."""
        env = {
            **_PROD_ENV,
            "GOLF_API_SECRET_KEY": _SAFE_SECRET_KEY,
            "GOLF_ADMIN_PASSWORD": _SAFE_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            from src.shared.python.security.env_validator import (
                _assert_production_secrets,
            )

            _assert_production_secrets()  # should not raise

    def test_no_raise_in_production_with_no_admin_password_set(self) -> None:
        """Production mode: unset GOLF_ADMIN_PASSWORD does not trigger default check."""
        env = {
            **_PROD_ENV,
            "GOLF_API_SECRET_KEY": _SAFE_SECRET_KEY,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            from src.shared.python.security.env_validator import (
                _assert_production_secrets,
            )

            _assert_production_secrets()  # should not raise (password not set != default)

    def test_no_raise_in_production_with_no_secret_key_set(self) -> None:
        """Production mode: unset GOLF_API_SECRET_KEY does not trigger default check.

        The default-credential guard only fires when the key is *exactly* the
        placeholder string; an absent key is handled separately by other validators.
        """
        env = {
            **_PROD_ENV,
            "GOLF_ADMIN_PASSWORD": _SAFE_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            from src.shared.python.security.env_validator import (
                _assert_production_secrets,
            )

            _assert_production_secrets()  # should not raise


# ---------------------------------------------------------------------------
# Tests: logger.critical is called before RuntimeError
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssertProductionSecretsLogging:
    """Verify logger.critical() is called before RuntimeError is raised."""

    def test_critical_logged_for_default_secret_key(self) -> None:
        """logger.critical() must be called when secret key is the default."""
        env = {
            **_PROD_ENV,
            "GOLF_API_SECRET_KEY": _DEFAULT_SECRET_KEY,
            "GOLF_ADMIN_PASSWORD": _SAFE_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            import src.shared.python.security.env_validator as ev_module

            with patch.object(ev_module.logger, "critical") as mock_critical:
                with pytest.raises(RuntimeError):
                    ev_module._assert_production_secrets()
                mock_critical.assert_called_once()

    def test_critical_logged_for_default_admin_password(self) -> None:
        """logger.critical() must be called when admin password is the default."""
        env = {
            **_PROD_ENV,
            "GOLF_API_SECRET_KEY": _SAFE_SECRET_KEY,
            "GOLF_ADMIN_PASSWORD": _DEFAULT_ADMIN_PASSWORD,
        }
        with patch.dict(os.environ, env, clear=True):
            from src.shared.python.config.environment import get_environment

            get_environment.cache_clear()
            import src.shared.python.security.env_validator as ev_module

            with patch.object(ev_module.logger, "critical") as mock_critical:
                with pytest.raises(RuntimeError):
                    ev_module._assert_production_secrets()
                mock_critical.assert_called_once()
