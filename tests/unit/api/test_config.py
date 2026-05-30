"""Unit tests for the UpstreamDrift API configuration module."""

import os
from unittest.mock import patch

import pytest
from src.api.config import (
    DEFAULT_ALLOWED_HOSTS,
    DEFAULT_CORS_ORIGINS,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    get_allowed_hosts,
    get_cors_origins,
    get_server_host,
    get_server_port,
)


def test_get_allowed_hosts_default() -> None:
    """Test get_allowed_hosts returns defaults when no env var is set."""
    with patch.dict(os.environ, {}, clear=True):
        hosts = get_allowed_hosts()
        assert hosts == DEFAULT_ALLOWED_HOSTS


def test_get_allowed_hosts_env() -> None:
    """Test get_allowed_hosts parses environment override."""
    with patch.dict(os.environ, {"ALLOWED_HOSTS": "host1.local,  host2.remote ,"}):
        hosts = get_allowed_hosts()
        assert hosts == ["host1.local", "host2.remote"]


def test_get_cors_origins_default() -> None:
    """Test get_cors_origins returns defaults when no env var is set."""
    with patch.dict(os.environ, {}, clear=True):
        origins = get_cors_origins()
        assert origins == DEFAULT_CORS_ORIGINS


def test_get_cors_origins_env() -> None:
    """Test get_cors_origins parses environment override."""
    with patch.dict(os.environ, {"CORS_ORIGINS": "https://a.com, https://b.com"}):
        origins = get_cors_origins()
        assert origins == ["https://a.com", "https://b.com"]


def test_get_server_host_default() -> None:
    with patch.dict(os.environ, {}, clear=True):
        host = get_server_host()
        assert host == DEFAULT_SERVER_HOST


def test_get_server_host_env() -> None:
    with patch.dict(os.environ, {"API_HOST": "0.0.0.0"}):
        host = get_server_host()
        assert host == "0.0.0.0"


def test_get_server_port_default() -> None:
    with patch.dict(os.environ, {}, clear=True):
        port = get_server_port()
        assert port == DEFAULT_SERVER_PORT


def test_get_server_port_env_valid() -> None:
    with patch.dict(os.environ, {"API_PORT": "9090"}):
        port = get_server_port()
        assert port == 9090


def test_get_server_port_env_invalid_type() -> None:
    with (
        patch.dict(os.environ, {"API_PORT": "invalid"}),
        pytest.raises(ValueError, match="Invalid API_PORT value: 'invalid'"),
    ):
        get_server_port()


def test_get_server_port_env_out_of_bounds() -> None:
    with (
        patch.dict(os.environ, {"API_PORT": "0"}),
        pytest.raises(ValueError, match="Invalid API_PORT value: '0'"),
    ):
        get_server_port()

    with (
        patch.dict(os.environ, {"API_PORT": "70000"}),
        pytest.raises(ValueError, match="Invalid API_PORT value: '70000'"),
    ):
        get_server_port()


def test_get_cors_origins_wildcard_rejected() -> None:
    """Test that wildcard '*' in CORS_ORIGINS is rejected."""
    with (
        patch.dict(os.environ, {"CORS_ORIGINS": "https://a.com,*,https://b.com"}),
        pytest.raises(ValueError, match="Cannot use wildcard '\\*' in CORS_ORIGINS"),
    ):
        get_cors_origins()
