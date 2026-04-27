"""Tests for auth-bypass default security (issue #2449).

Engine mutation endpoints must require authentication by default.
Auth bypass must only activate when GOLF_SUITE_MODE=local is explicitly set,
not when the variable is absent.
"""

from __future__ import annotations

import os
from unittest.mock import patch


class TestAuthDefaultIsSecure:
    """When no environment is configured, auth must be required (not bypassed)."""

    def test_is_auth_disabled_returns_false_when_env_unset(self) -> None:
        """Auth must NOT be disabled when GOLF_SUITE_MODE is absent."""
        from src.shared.python.config.environment import is_auth_disabled

        with patch.dict(os.environ, {}, clear=True):
            assert is_auth_disabled() is False, (
                "Auth should be required when GOLF_SUITE_MODE is not set. "
                "Deployers must explicitly opt-in to local/auth-free mode."
            )

    def test_get_golf_suite_mode_defaults_to_remote(self) -> None:
        """Suite mode must default to 'remote' (auth-required) when env var is absent."""
        from src.shared.python.config.environment import get_golf_suite_mode

        with patch.dict(os.environ, {}, clear=True):
            mode = get_golf_suite_mode()
            assert mode == "remote", (
                f"Default suite mode should be 'remote', got '{mode}'. "
                "The default must be secure (auth-required), not 'local'."
            )

    def test_is_local_mode_returns_false_when_env_unset(self) -> None:
        """is_local_mode() must return False when no env vars are configured."""
        from src.api.auth.middleware import is_local_mode

        with patch.dict(os.environ, {}, clear=True):
            assert (
                is_local_mode() is False
            ), "Local (auth-bypass) mode must not be active by default."

    def test_auth_disabled_only_when_explicitly_set_local(self) -> None:
        """Auth bypass only activates when GOLF_SUITE_MODE=local is explicit."""
        from src.shared.python.config.environment import is_auth_disabled

        with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}, clear=True):
            assert is_auth_disabled() is True

    def test_auth_required_when_suite_mode_is_remote(self) -> None:
        """Auth is required when GOLF_SUITE_MODE=remote."""
        from src.shared.python.config.environment import is_auth_disabled

        with patch.dict(
            os.environ,
            {"GOLF_SUITE_MODE": "remote", "GOLF_AUTH_DISABLED": "false"},
            clear=True,
        ):
            assert is_auth_disabled() is False

    def test_auth_disabled_flag_still_overrides(self) -> None:
        """GOLF_AUTH_DISABLED=true must still override even in remote mode."""
        from src.shared.python.config.environment import is_auth_disabled

        with patch.dict(
            os.environ,
            {"GOLF_SUITE_MODE": "remote", "GOLF_AUTH_DISABLED": "true"},
            clear=True,
        ):
            assert is_auth_disabled() is True
