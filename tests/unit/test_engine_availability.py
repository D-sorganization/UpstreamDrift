"""Tests for src.shared.python.engine_core.engine_availability (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.engine_core.engine_availability import (
    EngineStatus,
    get_available_engines,
    get_engine_error,
    get_engine_status,
    get_unavailable_engines,
    is_engine_available,
    require_engine,
)

# ---------------------------------------------------------------------------
# EngineStatus enum
# ---------------------------------------------------------------------------


class TestEngineStatus:
    def test_three_states(self) -> None:
        assert len(EngineStatus) == 3

    def test_available_state(self) -> None:
        assert EngineStatus.AVAILABLE.value == "available"

    def test_not_installed_state(self) -> None:
        assert EngineStatus.NOT_INSTALLED.value == "not_installed"

    def test_broken_state(self) -> None:
        assert EngineStatus.BROKEN.value == "broken"


# ---------------------------------------------------------------------------
# get_engine_status — real system checks (non-mocked)
# ---------------------------------------------------------------------------


class TestGetEngineStatus:
    def test_numpy_is_available(self) -> None:
        # numpy is definitely installed in the test environment
        status = get_engine_status("numpy")
        assert status == EngineStatus.AVAILABLE

    def test_nonexistent_engine_not_installed(self) -> None:
        status = get_engine_status("_definitely_not_installed_xyz_engine")
        assert status == EngineStatus.NOT_INSTALLED

    def test_returns_engine_status_type(self) -> None:
        status = get_engine_status("sys")
        assert isinstance(status, EngineStatus)

    def test_engine_availability_case_insensitive(self) -> None:
        s1 = get_engine_status("numpy")
        s2 = get_engine_status("NUMPY")
        assert s1 == s2

    def test_caching_consistent(self) -> None:
        # Same call twice should return same result
        s1 = get_engine_status("math")
        s2 = get_engine_status("math")
        assert s1 == s2


# ---------------------------------------------------------------------------
# is_engine_available
# ---------------------------------------------------------------------------


class TestIsEngineAvailable:
    def test_engine_availability_numpy_available(self) -> None:
        assert is_engine_available("numpy") is True

    def test_missing_engine_not_available(self) -> None:
        assert is_engine_available("_no_such_engine_xyz") is False


# ---------------------------------------------------------------------------
# get_engine_error
# ---------------------------------------------------------------------------


class TestGetEngineError:
    def test_available_engine_error_is_none(self) -> None:
        # For an available module, no error
        assert get_engine_error("numpy") is None

    def test_missing_engine_error_is_exception_or_none(self) -> None:
        # Unavailable engine may or may not store the error, but shouldn't raise
        result = get_engine_error("_no_such_engine_xyz")
        assert result is None or isinstance(result, Exception)


# ---------------------------------------------------------------------------
# get_available_engines / get_unavailable_engines
# ---------------------------------------------------------------------------


class TestGetEngineCollections:
    def test_available_engines_is_list(self) -> None:
        result = get_available_engines()
        assert isinstance(result, list)

    def test_unavailable_engines_is_list(self) -> None:
        result = get_unavailable_engines()
        assert isinstance(result, list)

    def test_no_overlap(self) -> None:
        available = set(get_available_engines())
        unavailable = set(get_unavailable_engines())
        assert available.isdisjoint(unavailable)

    def test_engine_availability_all_are_strings(self) -> None:
        for name in get_available_engines():
            assert isinstance(name, str)
        for name in get_unavailable_engines():
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# require_engine decorator
# ---------------------------------------------------------------------------


class TestRequireEngine:
    def test_available_engine_runs_function(self) -> None:
        @require_engine("numpy")
        def my_fn() -> str:
            return "ran"

        assert my_fn() == "ran"

    def test_unavailable_engine_skips(self) -> None:
        @require_engine("_no_such_engine_xyz")
        def my_fn() -> str:
            return "should not run"

        with pytest.raises(pytest.skip.Exception):
            my_fn()

    def test_custom_reason_in_skip_message(self) -> None:
        @require_engine("_no_such_engine_xyz", reason="Custom reason for skip")
        def my_fn() -> None:
            pass

        with pytest.raises(pytest.skip.Exception, match="Custom reason"):
            my_fn()

    def test_decorated_function_preserves_name(self) -> None:
        @require_engine("numpy")
        def well_named_function() -> None:
            pass

        assert well_named_function.__name__ == "well_named_function"
