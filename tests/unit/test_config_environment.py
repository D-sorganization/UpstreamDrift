"""Tests for src.shared.python.config.environment (Issues #1949, #1744)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from src.shared.python.config.environment import (
    EnvironmentError,
    get_api_host,
    get_api_port,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_list,
    get_realtime_host,
    get_realtime_port,
)

_TEST_VAR = "_UPSTREAM_DRIFT_TEST_VAR"


def _set(**kwargs: str) -> patch[dict[str, str]]:
    """Convenience: patch os.environ with the given key→value pairs."""
    return patch.dict(os.environ, kwargs)


def _unset(*keys: str) -> patch[dict[str, str]]:
    """Convenience: ensure the given keys are absent from os.environ."""
    env = {k: v for k, v in os.environ.items() if k not in keys}
    return patch.dict(os.environ, env, clear=True)


# ---------------------------------------------------------------------------
# get_env
# ---------------------------------------------------------------------------


class TestGetEnv:
    def test_returns_set_value(self) -> None:
        with _set(**{_TEST_VAR: "hello"}):
            assert get_env(_TEST_VAR) == "hello"

    def test_strips_whitespace_by_default(self) -> None:
        with _set(**{_TEST_VAR: "  hello  "}):
            assert get_env(_TEST_VAR) == "hello"

    def test_no_strip_preserves_whitespace(self) -> None:
        with _set(**{_TEST_VAR: "  hello  "}):
            assert get_env(_TEST_VAR, strip=False) == "  hello  "

    def test_returns_default_when_unset(self) -> None:
        with _unset(_TEST_VAR):
            assert get_env(_TEST_VAR, default="fallback") == "fallback"

    def test_returns_none_when_unset_no_default(self) -> None:
        with _unset(_TEST_VAR):
            assert get_env(_TEST_VAR) is None

    def test_required_raises_when_unset(self) -> None:
        with _unset(_TEST_VAR), pytest.raises(EnvironmentError):
            get_env(_TEST_VAR, required=True)

    def test_required_succeeds_when_set(self) -> None:
        with _set(**{_TEST_VAR: "val"}):
            assert get_env(_TEST_VAR, required=True) == "val"


# ---------------------------------------------------------------------------
# get_env_bool
# ---------------------------------------------------------------------------


class TestGetEnvBool:
    @pytest.mark.parametrize("truthy", ["true", "True", "TRUE", "yes", "1", "on"])
    def test_truthy_values(self, truthy: str) -> None:
        with _set(**{_TEST_VAR: truthy}):
            assert get_env_bool(_TEST_VAR) is True

    @pytest.mark.parametrize("falsy", ["false", "False", "FALSE", "no", "0", "off"])
    def test_falsy_values(self, falsy: str) -> None:
        with _set(**{_TEST_VAR: falsy}):
            assert get_env_bool(_TEST_VAR) is False

    def test_returns_default_when_unset(self) -> None:
        with _unset(_TEST_VAR):
            assert get_env_bool(_TEST_VAR, default=True) is True

    def test_unrecognized_returns_default(self) -> None:
        with _set(**{_TEST_VAR: "maybe"}):
            assert get_env_bool(_TEST_VAR, default=False) is False


# ---------------------------------------------------------------------------
# get_env_int
# ---------------------------------------------------------------------------


class TestGetEnvInt:
    def test_valid_integer(self) -> None:
        with _set(**{_TEST_VAR: "42"}):
            assert get_env_int(_TEST_VAR) == 42

    def test_returns_default_when_unset(self) -> None:
        with _unset(_TEST_VAR):
            assert get_env_int(_TEST_VAR, default=99) == 99

    def test_invalid_value_raises(self) -> None:
        with _set(**{_TEST_VAR: "not_a_number"}), pytest.raises(EnvironmentError):
            get_env_int(_TEST_VAR)

    def test_below_min_raises(self) -> None:
        with _set(**{_TEST_VAR: "0"}), pytest.raises(EnvironmentError):
            get_env_int(_TEST_VAR, min_value=1)

    def test_above_max_raises(self) -> None:
        with _set(**{_TEST_VAR: "100"}), pytest.raises(EnvironmentError):
            get_env_int(_TEST_VAR, max_value=50)

    def test_within_range_passes(self) -> None:
        with _set(**{_TEST_VAR: "8000"}):
            assert get_env_int(_TEST_VAR, min_value=1, max_value=65535) == 8000

    def test_negative_integer(self) -> None:
        with _set(**{_TEST_VAR: "-5"}):
            assert get_env_int(_TEST_VAR) == -5


# ---------------------------------------------------------------------------
# get_env_float
# ---------------------------------------------------------------------------


class TestGetEnvFloat:
    def test_valid_float(self) -> None:
        with _set(**{_TEST_VAR: "3.14"}):
            result = get_env_float(_TEST_VAR)
            assert abs(result - 3.14) < 1e-10

    def test_returns_default_when_unset(self) -> None:
        with _unset(_TEST_VAR):
            assert get_env_float(_TEST_VAR, default=0.5) == pytest.approx(0.5)

    def test_invalid_value_raises(self) -> None:
        with _set(**{_TEST_VAR: "abc"}), pytest.raises(EnvironmentError):
            get_env_float(_TEST_VAR)

    def test_below_min_raises(self) -> None:
        with _set(**{_TEST_VAR: "-1.0"}), pytest.raises(EnvironmentError):
            get_env_float(_TEST_VAR, min_value=0.0)

    def test_above_max_raises(self) -> None:
        with _set(**{_TEST_VAR: "2.0"}), pytest.raises(EnvironmentError):
            get_env_float(_TEST_VAR, max_value=1.0)


# ---------------------------------------------------------------------------
# get_env_list
# ---------------------------------------------------------------------------


class TestGetEnvList:
    def test_comma_separated(self) -> None:
        with _set(**{_TEST_VAR: "a,b,c"}):
            result = get_env_list(_TEST_VAR)
            assert result == ["a", "b", "c"]

    def test_returns_empty_when_unset(self) -> None:
        with _unset(_TEST_VAR):
            assert get_env_list(_TEST_VAR) == []

    def test_single_value(self) -> None:
        with _set(**{_TEST_VAR: "only"}):
            assert get_env_list(_TEST_VAR) == ["only"]

    def test_strips_whitespace(self) -> None:
        with _set(**{_TEST_VAR: " a , b , c "}):
            result = get_env_list(_TEST_VAR)
            assert result == ["a", "b", "c"]


class TestSocketAccessors:
    def test_get_api_host_defaults_to_loopback(self) -> None:
        with _unset("GOLF_API_HOST"):
            assert get_api_host() == "127.0.0.1"

    def test_get_api_port_defaults_to_8000(self) -> None:
        with _unset("GOLF_API_PORT"):
            assert get_api_port() == 8000

    def test_get_realtime_host_defaults_to_loopback(self) -> None:
        with _unset("GOLF_REALTIME_HOST"):
            assert get_realtime_host() == "127.0.0.1"

    def test_get_realtime_host_reads_env_override(self) -> None:
        with _set(GOLF_REALTIME_HOST="0.0.0.0"):
            assert get_realtime_host() == "0.0.0.0"

    def test_get_realtime_port_defaults_to_8765(self) -> None:
        with _unset("GOLF_REALTIME_PORT"):
            assert get_realtime_port() == 8765

    def test_get_realtime_port_reads_env_override(self) -> None:
        with _set(GOLF_REALTIME_PORT="9999"):
            assert get_realtime_port() == 9999
