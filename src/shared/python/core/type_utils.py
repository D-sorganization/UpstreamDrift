"""Centralized type conversion utilities for the Golf Modeling Suite.

This module consolidates common type conversion patterns across the codebase,
addressing DRY violations identified in Pragmatic Programmer reviews.

Usage:
    from src.shared.python.type_utils import (
        safe_int,
        safe_float,
        safe_str,
        to_numpy_array,
        ensure_list,
        coerce_numeric,
    )

    # Safe type conversions with defaults
    value = safe_int("42", default=0)  # 42
    value = safe_int("invalid", default=0)  # 0

    # Convert to numpy array
    arr = to_numpy_array([1, 2, 3])

    # Ensure value is a list
    items = ensure_list("single")  # ["single"]
    items = ensure_list(["a", "b"])  # ["a", "b"]
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def safe_int(
    value: Any,
    default: int | None = None,
    *,
    strict: bool = False,
) -> int | None:
    """Safely convert a value to int.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.
        strict: If True, raise ValueError on failure instead of returning default.

    Returns:
        Integer value or default.

    Raises:
        ValueError: If strict=True and conversion fails.

    Example:
        >>> safe_int("42")
        42
        >>> safe_int("invalid", default=0)
        0
        >>> safe_int(None, default=-1)
        -1
    """
    if value is None:
        if strict:
            raise ValueError("Cannot convert None to int")
        return default

    try:
        # Handle string representations
        if isinstance(value, str):
            # Strip whitespace and handle empty string
            value = value.strip()
            if not value:
                if strict:
                    raise ValueError("Cannot convert empty string to int")
                return default
            # Handle float-like strings by truncating
            if "." in value:
                return int(float(value))

        return int(value)
    except (ValueError, TypeError) as e:
        if strict:
            raise ValueError(f"Cannot convert {value!r} to int: {e}") from e
        return default


def safe_float(
    value: Any,
    default: float | None = None,
    *,
    strict: bool = False,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> float | None:
    """Safely convert a value to float.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.
        strict: If True, raise ValueError on failure instead of returning default.
        allow_nan: If False, treat NaN as conversion failure.
        allow_inf: If False, treat infinity as conversion failure.

    Returns:
        Float value or default.

    Raises:
        ValueError: If strict=True and conversion fails.

    Example:
        >>> safe_float("3.14")
        3.14
        >>> safe_float("invalid", default=0.0)
        0.0
    """
    import math

    if value is None:
        if strict:
            raise ValueError("Cannot convert None to float")
        return default

    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                if strict:
                    raise ValueError("Cannot convert empty string to float")
                return default

        result = float(value)

        # Check for NaN
        if not allow_nan and math.isnan(result):
            if strict:
                raise ValueError("NaN values not allowed")
            return default

        # Check for infinity
        if not allow_inf and math.isinf(result):
            if strict:
                raise ValueError("Infinite values not allowed")
            return default

        return result
    except (ValueError, TypeError) as e:
        if strict:
            raise ValueError(f"Cannot convert {value!r} to float: {e}") from e
        return default


def safe_str(
    value: Any,
    default: str | None = None,
    *,
    strip: bool = True,
    empty_as_none: bool = False,
) -> str | None:
    """Safely convert a value to string.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.
        strip: Strip whitespace from result.
        empty_as_none: Treat empty strings as None.

    Returns:
        String value or default.

    Example:
        >>> safe_str(42)
        "42"
        >>> safe_str(None, default="N/A")
        "N/A"
        >>> safe_str("  hello  ", strip=True)
        "hello"
    """
    if value is None:
        return default

    try:
        result = str(value)
        if strip:
            result = result.strip()
        if empty_as_none and not result:
            return default
        return result
    except Exception:
        return default


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    """Safely convert a value to bool.

    Recognizes: true/false, yes/no, 1/0, on/off (case-insensitive for strings).

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Boolean value.

    Example:
        >>> safe_bool("yes")
        True
        >>> safe_bool("0")
        False
        >>> safe_bool(1)
        True
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("true", "yes", "1", "on", "y", "t"):
            return True
        if value in ("false", "no", "0", "off", "n", "f", ""):
            return False

    return default


def to_numpy_array(
    value: Any,
    *,
    dtype: Any = None,
    copy: bool = False,
    ndmin: int = 0,
    allow_empty: bool = True,
) -> Any:
    """Convert a value to a numpy array.

    Args:
        value: Value to convert (list, tuple, array-like).
        dtype: Desired data type (default: float64).
        copy: Force a copy of the data.
        ndmin: Minimum number of dimensions.
        allow_empty: If False, raise ValueError for empty arrays.

    Returns:
        numpy.ndarray.

    Raises:
        ValueError: If conversion fails or empty array when not allowed.

    Example:
        >>> arr = to_numpy_array([1, 2, 3])
        >>> arr.dtype
        dtype('float64')
    """
    import numpy as np

    if dtype is None:
        dtype = np.float64

    try:
        arr = np.asarray(value, dtype=dtype)

        if copy:
            arr = arr.copy()

        if ndmin > 0 and arr.ndim < ndmin:
            arr = np.atleast_1d(arr) if ndmin == 1 else np.atleast_2d(arr)

        if not allow_empty and arr.size == 0:
            raise ValueError("Empty arrays not allowed")

        return arr
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert to numpy array: {e}") from e


def ensure_list(value: Any) -> list[Any]:
    """Ensure a value is a list.

    Args:
        value: Value to convert. Strings are treated as single items.

    Returns:
        List containing the value(s).

    Example:
        >>> ensure_list("item")
        ["item"]
        >>> ensure_list(["a", "b"])
        ["a", "b"]
        >>> ensure_list((1, 2))
        [1, 2]
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        return [value]

    if isinstance(value, (tuple, set, frozenset)):
        return list(value)

    if hasattr(value, "__iter__"):
        return list(value)

    return [value]


def ensure_tuple(value: Any) -> tuple[Any, ...]:
    """Ensure a value is a tuple.

    Args:
        value: Value to convert. Strings are treated as single items.

    Returns:
        Tuple containing the value(s).

    Example:
        >>> ensure_tuple("item")
        ("item",)
        >>> ensure_tuple([1, 2, 3])
        (1, 2, 3)
    """
    return tuple(ensure_list(value))


def coerce_numeric(
    value: Any,
    *,
    prefer_int: bool = True,
    default: int | float | None = None,
) -> int | float | None:
    """Coerce a value to int or float.

    Args:
        value: Value to convert.
        prefer_int: If True, return int when value is a whole number.
        default: Default value if conversion fails.

    Returns:
        Numeric value (int or float) or default.

    Example:
        >>> coerce_numeric("42")
        42
        >>> coerce_numeric("3.14")
        3.14
        >>> coerce_numeric("3.0", prefer_int=True)
        3
    """
    result = safe_float(value, default=None)

    if result is None:
        return default

    if prefer_int and result.is_integer():
        return int(result)

    return result


def clamp(
    value: T,
    min_value: T | None = None,
    max_value: T | None = None,
) -> T:
    """Clamp a value to a range.

    Args:
        value: Value to clamp.
        min_value: Minimum value (inclusive).
        max_value: Maximum value (inclusive).

    Returns:
        Clamped value.

    Example:
        >>> clamp(150, min_value=0, max_value=100)
        100
        >>> clamp(-5, min_value=0)
        0
    """
    if min_value is not None and value < min_value:  # type: ignore[operator]
        return min_value
    if max_value is not None and value > max_value:  # type: ignore[operator]
        return max_value
    return value


def is_numeric(value: Any) -> bool:
    """Check if a value can be converted to a number.

    Args:
        value: Value to check.

    Returns:
        True if value is numeric or can be converted to float.

    Example:
        >>> is_numeric(42)
        True
        >>> is_numeric("3.14")
        True
        >>> is_numeric("hello")
        False
    """
    if isinstance(value, (int, float)):
        return True
    return safe_float(value, default=None) is not None


def is_integer(value: Any, *, allow_float_like: bool = True) -> bool:
    """Check if a value is an integer or integer-like.

    Args:
        value: Value to check.
        allow_float_like: If True, accept floats that are whole numbers.

    Returns:
        True if value is an integer or integer-like.

    Example:
        >>> is_integer(42)
        True
        >>> is_integer(3.0, allow_float_like=True)
        True
        >>> is_integer(3.5)
        False
    """
    if isinstance(value, bool):
        return False

    if isinstance(value, int):
        return True

    if allow_float_like:
        if isinstance(value, float):
            return value.is_integer()
        result = safe_float(value, default=None)
        if result is not None:
            return result.is_integer()

    return False


def parse_slice(
    value: str,
    *,
    default_start: int | None = None,
    default_stop: int | None = None,
    default_step: int | None = None,
) -> slice:
    """Parse a slice from string notation.

    Args:
        value: Slice string (e.g., "1:10", "::2", "5:").
        default_start: Default start value.
        default_stop: Default stop value.
        default_step: Default step value.

    Returns:
        Python slice object.

    Example:
        >>> parse_slice("1:10")
        slice(1, 10, None)
        >>> parse_slice("::2")
        slice(None, None, 2)
    """
    parts = value.split(":")

    if len(parts) == 1:
        # Single index - convert to slice of length 1
        idx = safe_int(parts[0], strict=True)
        return slice(idx, idx + 1 if idx is not None else None)

    if len(parts) == 2:
        start = safe_int(parts[0]) if parts[0] else default_start
        stop = safe_int(parts[1]) if parts[1] else default_stop
        return slice(start, stop)

    if len(parts) == 3:
        start = safe_int(parts[0]) if parts[0] else default_start
        stop = safe_int(parts[1]) if parts[1] else default_stop
        step = safe_int(parts[2]) if parts[2] else default_step
        return slice(start, stop, step)

    raise ValueError(f"Invalid slice format: {value!r}")


def flatten(
    iterable: Iterable[Any],
    *,
    depth: int = -1,
    types: tuple[type, ...] = (list, tuple),
) -> list[Any]:
    """Flatten a nested iterable.

    Args:
        iterable: Iterable to flatten.
        depth: Maximum depth to flatten (-1 for unlimited).
        types: Types to flatten (default: list and tuple).

    Returns:
        Flattened list.

    Example:
        >>> flatten([[1, 2], [3, [4, 5]]])
        [1, 2, 3, 4, 5]
        >>> flatten([[1, 2], [3, [4, 5]]], depth=1)
        [1, 2, 3, [4, 5]]
    """
    result: list[Any] = []

    def _flatten(items: Any, current_depth: int) -> None:
        for item in items:
            if isinstance(item, types) and (depth < 0 or current_depth < depth):
                _flatten(item, current_depth + 1)
            else:
                result.append(item)

    _flatten(iterable, 0)
    return result


def first(
    iterable: Iterable[T],
    default: T | None = None,
    *,
    predicate: Any = None,
) -> T | None:
    """Get the first item from an iterable.

    Args:
        iterable: Iterable to get first item from.
        default: Default value if empty or no match.
        predicate: Optional function to filter items.

    Returns:
        First item (matching predicate if given) or default.

    Example:
        >>> first([1, 2, 3])
        1
        >>> first([], default=0)
        0
        >>> first([1, 2, 3, 4], predicate=lambda x: x > 2)
        3
    """
    if predicate is None:
        for item in iterable:
            return item
    else:
        for item in iterable:
            if predicate(item):
                return item
    return default
