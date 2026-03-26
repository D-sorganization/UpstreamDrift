"""Extended tests for engine_availability module.

Tests the helper functions, decorators, and availability flag
lookup logic in engine_availability.
"""

from __future__ import annotations

import pytest


class TestIsEngineAvailable:
    """Tests for the is_engine_available() function."""

    def test_numpy_always_available(self) -> None:
        """numpy should always be available in the test environment."""
        from src.shared.python.engine_core.engine_availability import (
            NUMPY_AVAILABLE,
            is_engine_available,
        )

        assert NUMPY_AVAILABLE is True
        assert is_engine_available("numpy") is True

    def test_unknown_engine_returns_false(self) -> None:
        """An unknown engine name should return False."""
        from src.shared.python.engine_core.engine_availability import (
            is_engine_available,
        )

        assert is_engine_available("this_engine_does_not_exist_xyz123") is False

    def test_alias_consistency(self) -> None:
        """Aliases should return the same result as the canonical name."""
        from src.shared.python.engine_core.engine_availability import (
            is_engine_available,
        )

        # torch is an alias for pytorch
        assert is_engine_available("torch") == is_engine_available("pytorch")

        # pillow is an alias for pil
        assert is_engine_available("pillow") == is_engine_available("pil")

        # yaml alias
        assert is_engine_available("pyyaml") == is_engine_available("yaml")

    def test_case_insensitive(self) -> None:
        """is_engine_available should handle any-case input."""
        from src.shared.python.engine_core.engine_availability import (
            is_engine_available,
        )

        # Function lowercases the key internally
        result_lower = is_engine_available("numpy")
        result_upper = is_engine_available("NUMPY")
        assert result_lower == result_upper

    def test_returns_bool(self) -> None:
        """is_engine_available should always return a plain bool."""
        from src.shared.python.engine_core.engine_availability import (
            is_engine_available,
        )

        for name in ["numpy", "mujoco", "drake", "unknown_xyz"]:
            result = is_engine_available(name)
            assert isinstance(result, bool), f"Expected bool for {name!r}"


class TestRequireEngineDecorator:
    """Tests for the require_engine() pytest skip decorator."""

    def test_require_numpy_does_not_skip(self) -> None:
        """Requiring numpy should not skip since numpy is always available."""
        from src.shared.python.engine_core.engine_availability import require_engine

        @require_engine("numpy")
        def dummy_test() -> str:
            return "ran"

        # In a regular call context (not pytest), the wrapped function runs
        # The decorator skips pytest tests; here we just verify it wraps correctly
        assert callable(dummy_test)

    def test_require_engine_wraps_function(self) -> None:
        """require_engine should return a callable."""
        from src.shared.python.engine_core.engine_availability import require_engine

        @require_engine("numpy")
        def some_function() -> int:
            return 42

        assert callable(some_function)


class TestSkipIfUnavailable:
    """Tests for the skip_if_unavailable() marker helper."""

    def test_skip_if_unavailable_returns_marker(self) -> None:
        """skip_if_unavailable should return a pytest.mark object."""
        from src.shared.python.engine_core.engine_availability import (
            skip_if_unavailable,
        )

        mark = skip_if_unavailable("numpy")
        # Should be a pytest mark (has the mark decorator info)
        assert mark is not None

    def test_skip_if_unavailable_unknown(self) -> None:
        """skip_if_unavailable for an unknown engine should return a mark."""
        from src.shared.python.engine_core.engine_availability import (
            skip_if_unavailable,
        )

        mark = skip_if_unavailable("definitely_not_installed_xyz999")
        assert mark is not None


class TestAvailabilityFlags:
    """Tests that core availability flags have the correct type."""

    @pytest.mark.parametrize(
        "flag_name",
        [
            "NUMPY_AVAILABLE",
            "SCIPY_AVAILABLE",
            "MATPLOTLIB_AVAILABLE",
            "MUJOCO_AVAILABLE",
            "PINOCCHIO_AVAILABLE",
            "DRAKE_AVAILABLE",
            "PYQT6_AVAILABLE",
            "YAML_AVAILABLE",
        ],
    )
    def test_flag_is_bool(self, flag_name: str) -> None:
        """Each availability flag should be a plain bool."""
        import importlib

        mod = importlib.import_module("src.shared.python.engine_core.engine_availability")
        flag = getattr(mod, flag_name)
        assert isinstance(flag, bool), f"{flag_name} should be bool, got {type(flag)}"

    def test_numpy_is_true(self) -> None:
        """numpy is required; NUMPY_AVAILABLE must be True in test env."""
        from src.shared.python.engine_core.engine_availability import NUMPY_AVAILABLE

        assert NUMPY_AVAILABLE is True

    def test_pyqt6_is_true(self) -> None:
        """PyQt6 is installed in this environment."""
        from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE

        assert PYQT6_AVAILABLE is True


class TestParquetAlias:
    """Tests for the PARQUET_AVAILABLE composite flag."""

    def test_parquet_available_flag_exists(self) -> None:
        """PARQUET_AVAILABLE should be defined."""
        from src.shared.python.engine_core.engine_availability import PARQUET_AVAILABLE

        assert isinstance(PARQUET_AVAILABLE, bool)

    def test_parquet_is_superset(self) -> None:
        """PARQUET_AVAILABLE should be True if either pyarrow or fastparquet is."""
        from src.shared.python.engine_core.engine_availability import (
            FASTPARQUET_AVAILABLE,
            PARQUET_AVAILABLE,
            PYARROW_AVAILABLE,
        )

        # If neither is available, PARQUET_AVAILABLE should be False
        expected = PYARROW_AVAILABLE or FASTPARQUET_AVAILABLE
        assert expected == PARQUET_AVAILABLE
