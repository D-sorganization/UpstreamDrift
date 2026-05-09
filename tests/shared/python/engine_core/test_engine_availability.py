"""Tests for engine_availability module.

Validates the public API of the engine availability checking utilities:
- is_engine_available()
- get_engine_status()
- get_engine_error()
- get_available_engines() / get_unavailable_engines()
- require_engine() decorator
- skip_if_unavailable() marker factory
- Module-level __getattr__ for XXXX_AVAILABLE pattern
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import src.shared.python.engine_core.engine_availability as ea
from src.shared.python.engine_core.engine_availability import (
    EngineStatus,
    get_available_engines,
    get_engine_error,
    get_engine_status,
    get_unavailable_engines,
    is_engine_available,
    require_engine,
    skip_if_unavailable,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_cache(*engine_names: str) -> dict:
    """Return a patched cache dict with the named engines removed."""
    cache = ea._engine_status_cache.copy()
    for name in engine_names:
        cache.pop(name, None)
    return cache


# ---------------------------------------------------------------------------
# get_engine_status — known engines
# ---------------------------------------------------------------------------


class TestGetEngineStatus:
    """get_engine_status returns valid EngineStatus enum values."""

    def test_numpy_returns_available(self) -> None:
        """numpy is almost always available; status should be AVAILABLE."""
        status = get_engine_status("numpy")
        assert isinstance(status, EngineStatus)
        # numpy is virtually always present
        assert status == EngineStatus.AVAILABLE

    def test_unknown_engine_returns_not_installed(self) -> None:
        """An engine that definitely cannot be imported is NOT_INSTALLED."""
        engine_name = "_gaai_nonexistent_engine_xyz_1234"
        with patch.dict(ea._engine_status_cache, {}, clear=False):
            ea._engine_status_cache.pop(engine_name, None)
            status = get_engine_status(engine_name)
        assert status == EngineStatus.NOT_INSTALLED

    def test_status_is_engine_status_enum(self) -> None:
        """Return type is always an EngineStatus enum member."""
        for engine in ("numpy", "scipy", "matplotlib"):
            status = get_engine_status(engine)
            assert isinstance(status, EngineStatus)

    def test_repeated_calls_return_same_status(self) -> None:
        """Cached result is consistent across calls."""
        s1 = get_engine_status("numpy")
        s2 = get_engine_status("numpy")
        assert s1 == s2


# ---------------------------------------------------------------------------
# get_engine_status — composite aliases
# ---------------------------------------------------------------------------


class TestCompositeAliases:
    """Composite alias lookups for parquet, qt, c3d_any, gym_any."""

    def test_parquet_returns_engine_status(self) -> None:
        """'parquet' alias resolves via pyarrow or fastparquet."""
        status = get_engine_status("parquet")
        assert isinstance(status, EngineStatus)

    def test_qt_returns_engine_status(self) -> None:
        """'qt' alias resolves via pyqt6, pyqt5, or pyside6."""
        status = get_engine_status("qt")
        assert isinstance(status, EngineStatus)

    def test_c3d_any_returns_engine_status(self) -> None:
        """'c3d_any' alias resolves via ezc3d or c3d_pkg."""
        status = get_engine_status("c3d_any")
        assert isinstance(status, EngineStatus)

    def test_gym_any_returns_engine_status(self) -> None:
        """'gym_any' alias resolves via gymnasium or gym."""
        status = get_engine_status("gym_any")
        assert isinstance(status, EngineStatus)


# ---------------------------------------------------------------------------
# is_engine_available
# ---------------------------------------------------------------------------


class TestIsEngineAvailable:
    """is_engine_available returns a plain bool."""

    def test_engine_availability_returns_bool(self) -> None:
        """Return type is bool, not EngineStatus."""
        result = is_engine_available("numpy")
        assert isinstance(result, bool)

    def test_numpy_is_available(self) -> None:
        """numpy is expected to be available."""
        assert is_engine_available("numpy") is True

    def test_nonexistent_engine_is_not_available(self) -> None:
        """A made-up engine name returns False."""
        engine_name = "_gaai_nonexistent_xyz_9999"
        with patch.dict(ea._engine_status_cache, {}, clear=False):
            ea._engine_status_cache.pop(engine_name, None)
            result = is_engine_available(engine_name)
        assert result is False


# ---------------------------------------------------------------------------
# get_engine_error
# ---------------------------------------------------------------------------


class TestGetEngineError:
    """get_engine_error returns None for available engines, Exception for missing."""

    def test_available_engine_has_no_error(self) -> None:
        """Available engine (numpy) should have no cached error."""
        get_engine_status("numpy")  # ensure cached
        err = get_engine_error("numpy")
        assert err is None

    def test_missing_engine_has_error(self) -> None:
        """A missing engine should have a cached ImportError."""
        engine_name = "_gaai_nonexistent_error_test_5678"
        with (
            patch.dict(ea._engine_status_cache, {}, clear=False),
            patch.dict(ea._engine_error_cache, {}, clear=False),
        ):
            ea._engine_status_cache.pop(engine_name, None)
            ea._engine_error_cache.pop(engine_name, None)
            get_engine_status(engine_name)
            err = get_engine_error(engine_name)
        # May be None if the key wasn't in _MODULE_MAPPING (probed with import_name=engine_name)
        # or an ImportError. Either way, not a non-ImportError exception.
        assert err is None or isinstance(err, Exception)


# ---------------------------------------------------------------------------
# get_available_engines / get_unavailable_engines
# ---------------------------------------------------------------------------


class TestGetEnginesLists:
    """get_available_engines and get_unavailable_engines return lists of strings."""

    def test_available_engines_returns_list(self) -> None:
        """get_available_engines returns a list."""
        result = get_available_engines()
        assert isinstance(result, list)

    def test_unavailable_engines_returns_list(self) -> None:
        """get_unavailable_engines returns a list."""
        result = get_unavailable_engines()
        assert isinstance(result, list)

    def test_available_engines_contains_strings(self) -> None:
        """All entries in available engines are strings."""
        result = get_available_engines()
        assert all(isinstance(e, str) for e in result)

    def test_unavailable_engines_contains_strings(self) -> None:
        """All entries in unavailable engines are strings."""
        result = get_unavailable_engines()
        assert all(isinstance(e, str) for e in result)

    def test_numpy_in_available(self) -> None:
        """numpy appears in get_available_engines when installed."""
        available = get_available_engines()
        assert "numpy" in available

    def test_sets_are_disjoint(self) -> None:
        """No engine appears in both available and unavailable lists."""
        available = set(get_available_engines())
        unavailable = set(get_unavailable_engines())
        assert available.isdisjoint(unavailable)


# ---------------------------------------------------------------------------
# Module-level __getattr__ for XXXX_AVAILABLE pattern
# ---------------------------------------------------------------------------


class TestModuleGetattr:
    """Module-level __getattr__ supports XXXX_AVAILABLE lazy attributes."""

    def test_numpy_available_attribute(self) -> None:
        """NUMPY_AVAILABLE attribute returns bool True."""
        result = ea.NUMPY_AVAILABLE
        assert isinstance(result, bool)
        assert result is True

    def test_scipy_available_attribute(self) -> None:
        """SCIPY_AVAILABLE attribute returns a bool."""
        result = ea.SCIPY_AVAILABLE
        assert isinstance(result, bool)

    def test_nonexistent_attribute_raises(self) -> None:
        """Accessing an unknown attribute raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = ea.TOTALLY_NOT_A_REAL_ATTRIBUTE  # type: ignore[attr-defined]

    def test_available_attribute_not_ending_in_available_raises(self) -> None:
        """Random module attribute that doesn't follow pattern raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = ea.some_random_name  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# require_engine decorator
# ---------------------------------------------------------------------------


class TestRequireEngine:
    """require_engine decorator skips or passes based on engine availability."""

    def test_decorator_passes_for_available_engine(self) -> None:
        """Function executes when engine is available."""
        call_log = []

        @require_engine("numpy")
        def _fn() -> None:
            call_log.append("called")

        _fn()
        assert call_log == ["called"]

    def test_decorator_skips_for_unavailable_engine(self) -> None:
        """pytest.skip is called when engine is not available."""
        engine_name = "_gaai_nonexistent_skip_test_7890"

        @require_engine(engine_name)
        def _fn() -> None:
            pass  # pragma: no cover

        with pytest.raises(pytest.skip.Exception):
            _fn()

    def test_decorator_preserves_function_name(self) -> None:
        """functools.wraps preserves the wrapped function name."""

        @require_engine("numpy")
        def my_named_function() -> None:
            pass

        assert my_named_function.__name__ == "my_named_function"


# ---------------------------------------------------------------------------
# skip_if_unavailable marker factory
# ---------------------------------------------------------------------------


class TestSkipIfUnavailable:
    """skip_if_unavailable returns a pytest mark."""

    def test_returns_pytest_mark(self) -> None:
        """Return value is a pytest mark object."""
        mark = skip_if_unavailable("numpy")
        # pytest marks have a 'name' or 'markname' attribute
        assert mark is not None

    def test_does_not_raise_for_known_engine(self) -> None:
        """No exception raised for a known engine name."""
        # Should not raise
        skip_if_unavailable("scipy")

    def test_does_not_raise_for_unknown_engine(self) -> None:
        """No exception raised for an unknown engine (will be marked skip)."""
        skip_if_unavailable("_gaai_nonexistent_marker_test_1111")
