"""Shared validation helpers of the force-plate contracts (OF-05 review)."""

from __future__ import annotations

import math

import pytest

from src.shared.python.force_plates.contracts import finite_real, require_foot_keys

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_finite_real_accepts_numbers() -> None:
    assert finite_real(2, "x") == 2.0
    assert isinstance(finite_real(2, "x"), float)


def test_finite_real_rejects_bool_and_text() -> None:
    with pytest.raises(TypeError, match="x"):
        finite_real(True, "x")
    with pytest.raises(TypeError, match="x"):
        finite_real("1", "x")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_finite_real_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="x"):
        finite_real(bad, "x")


def test_require_foot_keys_accepts_exact_keys_and_extras() -> None:
    require_foot_keys({"left": 1, "right": 2}, "load_n")
    require_foot_keys({"left": 1, "right": 2, "net": 3}, "cop_m", extra=("net",))


@pytest.mark.parametrize(
    "table", [{"left": 1}, {"left": 1, "right": 2, "net": 3}, {"a": 1, "b": 2}]
)
def test_require_foot_keys_rejects_other_keys(table: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="load_n must be keyed exactly"):
        require_foot_keys(table, "load_n")
