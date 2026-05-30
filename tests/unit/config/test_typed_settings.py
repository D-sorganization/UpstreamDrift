"""Tests for the unified pydantic-settings ``Settings`` class (issue #6565).

These tests pin the behavior-preserving contract: each field reads the same
env var name with the same default as the legacy accessor it replaces.
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.shared.python.config.typed_settings import (
    DEFAULT_ALLOWED_HOSTS,
    DEFAULT_CORS_ORIGINS,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    Settings,
    get_settings,
)


def test_defaults_match_documented_values() -> None:
    """No env vars set -> documented defaults."""
    with patch.dict(os.environ, {}, clear=True):
        settings = get_settings()
        assert settings.server_host == DEFAULT_SERVER_HOST == "127.0.0.1"
        assert settings.server_port == DEFAULT_SERVER_PORT == 8000
        assert settings.allowed_hosts == DEFAULT_ALLOWED_HOSTS
        assert settings.cors_origins == DEFAULT_CORS_ORIGINS


def test_server_host_reads_api_host_env() -> None:
    with patch.dict(os.environ, {"API_HOST": "0.0.0.0"}):
        assert get_settings().server_host == "0.0.0.0"


def test_server_port_reads_api_port_env() -> None:
    with patch.dict(os.environ, {"API_PORT": "9090"}):
        assert get_settings().server_port == 9090


def test_server_port_invalid_int_raises() -> None:
    with (
        patch.dict(os.environ, {"API_PORT": "not-an-int"}),
        pytest.raises(ValidationError),
    ):
        get_settings()


def test_server_port_out_of_range_raises() -> None:
    with patch.dict(os.environ, {"API_PORT": "0"}), pytest.raises(ValueError):
        Settings()
    with patch.dict(os.environ, {"API_PORT": "70000"}), pytest.raises(ValueError):
        Settings()


def test_allowed_hosts_env_override_csv() -> None:
    with patch.dict(os.environ, {"ALLOWED_HOSTS": "host1.local,  host2.remote ,"}):
        assert get_settings().allowed_hosts == ["host1.local", "host2.remote"]


def test_cors_origins_env_override_csv() -> None:
    with patch.dict(os.environ, {"CORS_ORIGINS": "https://a.com, https://b.com"}):
        assert get_settings().cors_origins == ["https://a.com", "https://b.com"]


def test_get_settings_is_not_cached() -> None:
    """Each call observes the current environment (no process caching)."""
    with patch.dict(os.environ, {"API_HOST": "1.2.3.4"}):
        assert get_settings().server_host == "1.2.3.4"
    with patch.dict(os.environ, {"API_HOST": "5.6.7.8"}):
        assert get_settings().server_host == "5.6.7.8"
