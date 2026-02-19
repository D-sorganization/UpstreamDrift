"""Tests for src.shared.python.core.type_utils module."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.type_utils import (
    clamp,
    coerce_numeric,
    ensure_list,
    ensure_tuple,
    is_integer,
    is_numeric,
    safe_bool,
    safe_float,
    safe_int,
    safe_str,
    to_numpy_array,
)


class TestSafeInt:
    """Tests for safe_int function."""

    @pytest.mark.parametrize(
        "value, kwargs, expected",
        [
            (42, {}, 42),
            ("42", {}, 42),
            (3.9, {}, 3),
            ("abc", {"default": 0}, 0),
            (None, {"default": -1}, -1),
        ],
        ids=["int-passthrough", "string-conversion", "float-truncation",
             "invalid-default", "none-default"],
    )
    def test_safe_int_conversions(self, value: object, kwargs: dict, expected: int) -> None:
        """Test safe_int handles various input types correctly."""
        assert safe_int(value, **kwargs) == expected

    def test_strict_raises(self) -> None:
        with pytest.raises(ValueError):
            safe_int("abc", strict=True)


class TestSafeFloat:
    """Tests for safe_float function."""

    @pytest.mark.parametrize(
        "value, kwargs, expected",
        [
            (3.14, {}, 3.14),
            ("3.14", {}, 3.14),
            (42, {}, 42.0),
            ("abc", {"default": 0.0}, 0.0),
        ],
        ids=["float-passthrough", "string-conversion", "int-to-float", "invalid-default"],
    )
    def test_safe_float_conversions(
        self, value: object, kwargs: dict, expected: float
    ) -> None:
        """Test safe_float handles various input types correctly."""
        assert safe_float(value, **kwargs) == pytest.approx(expected)

    def test_nan_rejected_by_default(self) -> None:
        result = safe_float(float("nan"), default=0.0)
        assert result == 0.0

    def test_nan_allowed(self) -> None:
        result = safe_float(float("nan"), allow_nan=True)
        assert result is not None

    def test_inf_rejected_by_default(self) -> None:
        result = safe_float(float("inf"), default=0.0)
        assert result == 0.0

    def test_inf_allowed(self) -> None:
        result = safe_float(float("inf"), allow_inf=True)
        assert result is not None

    def test_strict_raises(self) -> None:
        with pytest.raises(ValueError):
            safe_float("bad", strict=True)


class TestSafeStr:
    """Tests for safe_str function."""

    def test_str_passthrough(self) -> None:
        assert safe_str("hello") == "hello"

    def test_int_to_str(self) -> None:
        assert safe_str(42) == "42"

    def test_strip_whitespace(self) -> None:
        assert safe_str("  hello  ", strip=True) == "hello"

    def test_none_returns_default(self) -> None:
        assert safe_str(None, default="N/A") == "N/A"

    def test_empty_as_none(self) -> None:
        result = safe_str("", empty_as_none=True, default="fallback")
        assert result == "fallback"


class TestSafeBool:
    """Tests for safe_bool function."""

    @pytest.mark.parametrize(
        "value", [True, 1, "true", "yes", "on", "1", "True", "YES"],
    )
    def test_true_values(self, value: object) -> None:
        assert safe_bool(value) is True

    @pytest.mark.parametrize(
        "value", [False, 0, "false", "no", "off", "0", "False", "NO"],
    )
    def test_false_values(self, value: object) -> None:
        assert safe_bool(value) is False

    def test_default(self) -> None:
        assert safe_bool("maybe", default=True) is True


class TestToNumpyArray:
    """Tests for to_numpy_array function."""

    def test_list_to_array(self) -> None:
        arr = to_numpy_array([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_default_dtype(self) -> None:
        arr = to_numpy_array([1, 2, 3])
        assert arr.dtype == np.float64

    def test_custom_dtype(self) -> None:
        arr = to_numpy_array([1, 2, 3], dtype=np.int32)
        assert arr.dtype == np.int32

    def test_ndmin(self) -> None:
        arr = to_numpy_array([1, 2], ndmin=2)
        assert arr.ndim >= 2

    def test_empty_allowed(self) -> None:
        arr = to_numpy_array([], allow_empty=True)
        assert len(arr) == 0

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            to_numpy_array([], allow_empty=False)


class TestEnsureList:
    """Tests for ensure_list function."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("item", ["item"]),
            (["a", "b"], ["a", "b"]),
            ((1, 2), [1, 2]),
            (42, [42]),
        ],
        ids=["string-wraps", "list-passthrough", "tuple-converts", "int-wraps"],
    )
    def test_ensure_list_conversions(self, value: object, expected: list) -> None:
        """Test ensure_list wraps/converts various types correctly."""
        assert ensure_list(value) == expected

    def test_none_wraps(self) -> None:
        result = ensure_list(None)
        assert isinstance(result, list)


class TestEnsureTuple:
    """Tests for ensure_tuple function."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("item", ("item",)),
            ([1, 2, 3], (1, 2, 3)),
            ((1, 2), (1, 2)),
        ],
        ids=["string-wraps", "list-converts", "tuple-passthrough"],
    )
    def test_ensure_tuple_conversions(self, value: object, expected: tuple) -> None:
        """Test ensure_tuple wraps/converts various types correctly."""
        assert ensure_tuple(value) == expected


class TestCoerceNumeric:
    """Tests for coerce_numeric function."""

    @pytest.mark.parametrize(
        "value, kwargs, expected",
        [
            ("42", {}, 42),
            ("3.14", {}, 3.14),
            ("abc", {"default": 0}, 0),
        ],
        ids=["int-string", "float-string", "invalid-default"],
    )
    def test_coerce_numeric_cases(
        self, value: str, kwargs: dict, expected: object
    ) -> None:
        """Test coerce_numeric handles various string inputs."""
        result = coerce_numeric(value, **kwargs)
        if isinstance(expected, float):
            assert result == pytest.approx(expected)
        else:
            assert result == expected

    def test_whole_float_prefers_int(self) -> None:
        assert coerce_numeric("3.0", prefer_int=True) == 3
        assert isinstance(coerce_numeric("3.0", prefer_int=True), int)


class TestClamp:
    """Tests for clamp function."""

    @pytest.mark.parametrize(
        "value, kwargs, expected",
        [
            (50, {"min_value": 0, "max_value": 100}, 50),
            (-5, {"min_value": 0}, 0),
            (150, {"max_value": 100}, 100),
            (999, {}, 999),
            (-10, {"min_value": 0, "max_value": 100}, 0),
            (200, {"min_value": 0, "max_value": 100}, 100),
        ],
        ids=["within-range", "below-min", "above-max", "no-bounds",
             "below-with-both", "above-with-both"],
    )
    def test_clamp_cases(self, value: float, kwargs: dict, expected: float) -> None:
        """Test clamp constrains values to bounds correctly."""
        assert clamp(value, **kwargs) == expected


class TestIsNumeric:
    """Tests for is_numeric function."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (42, True),
            (3.14, True),
            ("3.14", True),
            ("hello", False),
            (None, False),
        ],
        ids=["int", "float", "numeric-string", "non-numeric-string", "none"],
    )
    def test_is_numeric_cases(self, value: object, expected: bool) -> None:
        """Test is_numeric correctly identifies numeric values."""
        assert is_numeric(value) is expected


class TestIsInteger:
    """Tests for is_integer function."""

    def test_int(self) -> None:
        assert is_integer(42) is True

    def test_float_whole(self) -> None:
        assert is_integer(3.0, allow_float_like=True) is True

    def test_float_fractional(self) -> None:
        assert is_integer(3.5) is False

    def test_string(self) -> None:
        assert is_integer("hello") is False
