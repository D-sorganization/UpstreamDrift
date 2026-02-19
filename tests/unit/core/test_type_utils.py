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

    def test_int_passthrough(self) -> None:
        assert safe_int(42) == 42

    def test_string_conversion(self) -> None:
        assert safe_int("42") == 42

    def test_float_truncation(self) -> None:
        assert safe_int(3.9) == 3

    def test_invalid_returns_default(self) -> None:
        assert safe_int("abc", default=0) == 0

    def test_none_returns_default(self) -> None:
        assert safe_int(None, default=-1) == -1

    def test_strict_raises(self) -> None:
        with pytest.raises(ValueError):
            safe_int("abc", strict=True)


class TestSafeFloat:
    """Tests for safe_float function."""

    def test_float_passthrough(self) -> None:
        assert safe_float(3.14) == pytest.approx(3.14)

    def test_string_conversion(self) -> None:
        assert safe_float("3.14") == pytest.approx(3.14)

    def test_int_to_float(self) -> None:
        assert safe_float(42) == 42.0

    def test_invalid_returns_default(self) -> None:
        assert safe_float("abc", default=0.0) == 0.0

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

    def test_true_values(self) -> None:
        for v in [True, 1, "true", "yes", "on", "1", "True", "YES"]:
            assert safe_bool(v) is True

    def test_false_values(self) -> None:
        for v in [False, 0, "false", "no", "off", "0", "False", "NO"]:
            assert safe_bool(v) is False

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

    def test_string_wraps(self) -> None:
        assert ensure_list("item") == ["item"]

    def test_list_passthrough(self) -> None:
        assert ensure_list(["a", "b"]) == ["a", "b"]

    def test_tuple_converts(self) -> None:
        assert ensure_list((1, 2)) == [1, 2]

    def test_none_wraps(self) -> None:
        result = ensure_list(None)
        assert isinstance(result, list)

    def test_int_wraps(self) -> None:
        assert ensure_list(42) == [42]


class TestEnsureTuple:
    """Tests for ensure_tuple function."""

    def test_string_wraps(self) -> None:
        assert ensure_tuple("item") == ("item",)

    def test_list_converts(self) -> None:
        assert ensure_tuple([1, 2, 3]) == (1, 2, 3)

    def test_tuple_passthrough(self) -> None:
        assert ensure_tuple((1, 2)) == (1, 2)


class TestCoerceNumeric:
    """Tests for coerce_numeric function."""

    def test_int_string(self) -> None:
        assert coerce_numeric("42") == 42

    def test_float_string(self) -> None:
        assert coerce_numeric("3.14") == pytest.approx(3.14)

    def test_whole_float_prefers_int(self) -> None:
        assert coerce_numeric("3.0", prefer_int=True) == 3
        assert isinstance(coerce_numeric("3.0", prefer_int=True), int)

    def test_invalid_returns_default(self) -> None:
        assert coerce_numeric("abc", default=0) == 0


class TestClamp:
    """Tests for clamp function."""

    def test_within_range(self) -> None:
        assert clamp(50, min_value=0, max_value=100) == 50

    def test_below_min(self) -> None:
        assert clamp(-5, min_value=0) == 0

    def test_above_max(self) -> None:
        assert clamp(150, max_value=100) == 100

    def test_no_bounds(self) -> None:
        assert clamp(999) == 999

    def test_both_bounds(self) -> None:
        assert clamp(-10, min_value=0, max_value=100) == 0
        assert clamp(200, min_value=0, max_value=100) == 100


class TestIsNumeric:
    """Tests for is_numeric function."""

    def test_int(self) -> None:
        assert is_numeric(42) is True

    def test_float(self) -> None:
        assert is_numeric(3.14) is True

    def test_numeric_string(self) -> None:
        assert is_numeric("3.14") is True

    def test_non_numeric_string(self) -> None:
        assert is_numeric("hello") is False

    def test_none(self) -> None:
        assert is_numeric(None) is False


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
