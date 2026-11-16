"""Overflow follows the same fallback/strict contract as other conversion errors."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from src.shared.python.core.type_utils import coerce_numeric, safe_float, safe_int

pytestmark = pytest.mark.unit

INT_OVERFLOWS = [float("inf"), float("-inf"), "1.0e309", "-1.0e309"]
INT_IDS = ["positive-inf", "negative-inf", "positive-string", "negative-string"]


@pytest.mark.parametrize("value", INT_OVERFLOWS, ids=INT_IDS)
@pytest.mark.parametrize("default", [None, -11])
def test_int_overflow_uses_default(value: float | str, default: int | None) -> None:
    assert safe_int(value, default=default) == default


@pytest.mark.parametrize("value", INT_OVERFLOWS, ids=INT_IDS)
def test_strict_int_overflow_is_a_chained_value_error(value: float | str) -> None:
    with pytest.raises(ValueError, match="Cannot convert") as caught:
        safe_int(value, strict=True)

    assert isinstance(caught.value.__cause__, OverflowError)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("default", [None, -11.5])
def test_float_overflow_uses_default(sign: int, default: float | None) -> None:
    assert safe_float(sign * 10**400, default=default) == default


@pytest.mark.parametrize("sign", [1, -1])
def test_strict_float_overflow_is_a_chained_value_error(sign: int) -> None:
    with pytest.raises(ValueError, match="Cannot convert") as caught:
        safe_float(sign * 10**400, strict=True)

    assert isinstance(caught.value.__cause__, OverflowError)


@pytest.mark.parametrize("sign", [1, -1])
def test_allowing_infinity_does_not_invent_an_overflow_result(sign: int) -> None:
    assert safe_float(sign * 10**400, default=-11.5, allow_inf=True) == -11.5
    result = safe_float(sign * math.inf, allow_inf=True)
    assert result == sign * math.inf


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("prefer_int", [False, True])
@pytest.mark.parametrize("default", [None, -19.5])
def test_numeric_coercion_preserves_default_on_overflow(
    sign: int, prefer_int: bool, default: float | None
) -> None:
    assert (
        coerce_numeric(sign * 10**400, prefer_int=prefer_int, default=default)
        == default
    )


def test_large_integers_still_convert_without_float_round_trip() -> None:
    value = 10**400 + 1

    assert safe_int(value) == value
    assert safe_int(str(value)) == value


@pytest.mark.parametrize(
    ("convert", "value"),
    [
        (safe_int, None),
        (safe_int, ""),
        (safe_int, "   "),
        (safe_float, None),
        (safe_float, ""),
        (safe_float, "   "),
        (safe_float, "nan"),
        (safe_float, "inf"),
        (safe_float, "-inf"),
    ],
)
def test_other_strict_conversion_boundaries_still_raise_value_error(
    convert: Callable[..., int | float | None], value: object
) -> None:
    with pytest.raises(ValueError):
        convert(value, strict=True)


@pytest.mark.parametrize("convert", [safe_int, safe_float])
@pytest.mark.parametrize("value", ["", " \t "])
def test_blank_numeric_strings_preserve_default(
    convert: Callable[..., int | float | None], value: str
) -> None:
    assert convert(value, default=-7) == -7


@pytest.mark.parametrize(("value", "expected"), [(" 3.9 ", 3), ("-3.9", -3)])
def test_finite_decimal_strings_still_truncate_toward_zero(
    value: str, expected: int
) -> None:
    assert safe_int(value) == expected
