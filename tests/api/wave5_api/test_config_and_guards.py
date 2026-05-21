"""Tests for src/api/config.py, debug_guard.py, rate_limit.py."""

from __future__ import annotations

import pytest

from src.api import config, debug_guard, rate_limit

pytestmark = pytest.mark.unit


# ----- config.py -----


def test_default_allowed_hosts_unmodified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOWED_HOSTS", raising=False)
    hosts = config.get_allowed_hosts()
    assert hosts == config.DEFAULT_ALLOWED_HOSTS
    hosts.append("evil")
    # Should return a copy, not the canonical list
    assert "evil" not in config.DEFAULT_ALLOWED_HOSTS


def test_allowed_hosts_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_HOSTS", "a.com, b.com ,  ,c.com")
    assert config.get_allowed_hosts() == ["a.com", "b.com", "c.com"]


def test_allowed_hosts_empty_env_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_HOSTS", "")
    assert config.get_allowed_hosts() == config.DEFAULT_ALLOWED_HOSTS


def test_cors_origins_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://x , http://y")
    assert config.get_cors_origins() == ["http://x", "http://y"]


def test_cors_origins_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert config.get_cors_origins() == config.DEFAULT_CORS_ORIGINS


def test_server_host_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_HOST", raising=False)
    assert config.get_server_host() == config.DEFAULT_SERVER_HOST


def test_server_host_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    assert config.get_server_host() == "0.0.0.0"


def test_server_port_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_PORT", raising=False)
    assert config.get_server_port() == config.DEFAULT_SERVER_PORT


def test_server_port_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PORT", "9000")
    assert config.get_server_port() == 9000


@pytest.mark.parametrize("bad", ["0", "70000", "-1", "notanumber"])
def test_server_port_invalid_raises(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("API_PORT", bad)
    with pytest.raises(ValueError):
        config.get_server_port()


def test_constants_sane() -> None:
    assert config.MAX_UPLOAD_SIZE_BYTES > 0
    assert config.MAX_UPLOAD_SIZE_MB == config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
    assert (
        0 <= config.MIN_CONFIDENCE <= config.DEFAULT_CONFIDENCE <= config.MAX_CONFIDENCE
    )
    assert "mediapipe" in config.VALID_ESTIMATOR_TYPES


# ----- debug_guard.py -----


def test_debug_default_non_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_DRIFT_ENV", "development")
    monkeypatch.delenv("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", raising=False)
    assert debug_guard.debug_endpoints_enabled() is True


def test_debug_production_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_DRIFT_ENV", "production")
    monkeypatch.delenv("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", raising=False)
    assert debug_guard.debug_endpoints_enabled() is False


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_debug_production_truthy_overrides(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setenv("UPSTREAM_DRIFT_ENV", "production")
    monkeypatch.setenv("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", flag)
    assert debug_guard.debug_endpoints_enabled() is True


@pytest.mark.parametrize("flag", ["0", "false", "no", "off", "", "maybe"])
def test_debug_production_falsy_stays_off(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setenv("UPSTREAM_DRIFT_ENV", "production")
    monkeypatch.setenv("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", flag)
    assert debug_guard.debug_endpoints_enabled() is False


# ----- rate_limit.py -----


def test_rate_limit_default_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UD_TEST_LIMIT", raising=False)
    assert rate_limit.get_limit("UD_TEST_LIMIT", "5/minute") == "5/minute"


def test_rate_limit_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UD_TEST_LIMIT", "10/second")
    assert rate_limit.get_limit("UD_TEST_LIMIT", "5/minute") == "10/second"


def test_rate_limit_blank_env_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UD_TEST_LIMIT", "   ")
    assert rate_limit.get_limit("UD_TEST_LIMIT", "5/minute") == "5/minute"


def test_limiter_exists() -> None:
    assert rate_limit.limiter is not None
    assert callable(rate_limit.limiter.limit)
