"""Scalar finiteness fast path matches ``contracts.require_finite``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.contracts import (
    ContractLevel,
    PreconditionError,
    get_contract_level,
    require_finite,
    set_contract_level,
)
from src.shared.python.putting_dynamics._checks import require_finite_scalar

pytestmark = pytest.mark.unit

_FINITE = [0.0, -1.5, 3, 1e308, np.float64(2.5)]
_NON_FINITE = [math.nan, math.inf, -math.inf, np.float64("nan")]


@pytest.mark.parametrize("value", _FINITE)
def test_finite_scalars_pass(value: float) -> None:
    require_finite_scalar(value, "value")
    require_finite(value, "value")


@pytest.mark.parametrize("value", _NON_FINITE)
def test_non_finite_scalars_raise_like_the_array_check(value: float) -> None:
    with pytest.raises(PreconditionError, match="speed"):
        require_finite_scalar(value, "speed")
    with pytest.raises(PreconditionError, match="speed"):
        require_finite(value, "speed")


def test_check_is_skipped_when_contracts_are_off() -> None:
    previous = get_contract_level()
    set_contract_level(ContractLevel.OFF)
    try:
        require_finite_scalar(math.nan, "value")
    finally:
        set_contract_level(previous)
