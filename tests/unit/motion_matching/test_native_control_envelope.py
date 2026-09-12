"""Explicit continuation changes only selected numerical control intervals."""

import numpy as np
import pytest

from src.shared.python.motion_matching.native_restart import widen_control_envelope

pytestmark = pytest.mark.unit


def test_selected_bounds_expand_about_fixed_centers_without_mutating_inputs() -> None:
    lower = np.array([[-2.0, -10.0], [-1.0, 0.0]])
    upper = np.array([[2.0, 10.0], [3.0, 1.0]])
    selected = np.array([[True, False], [True, False]])
    lo, hi = widen_control_envelope(lower, upper, selected, factor=2.0)
    np.testing.assert_array_equal(lo, [[-4.0, -10.0], [-3.0, 0.0]])
    np.testing.assert_array_equal(hi, [[4.0, 10.0], [5.0, 1.0]])
    np.testing.assert_array_equal(lower, [[-2.0, -10.0], [-1.0, 0.0]])
    np.testing.assert_array_equal(upper, [[2.0, 10.0], [3.0, 1.0]])
    assert not lo.flags.writeable and not hi.flags.writeable
    assert not np.shares_memory(lo, lower) and not np.shares_memory(hi, upper)


@pytest.mark.parametrize("factor", [0.0, 1.0, -2.0, np.nan, np.inf, True])
def test_invalid_expansion_factor_rejected(factor: float) -> None:
    with pytest.raises(ValueError, match="factor"):
        widen_control_envelope(
            np.array([-2.0]), np.array([2.0]), np.array([True]), factor=factor
        )


@pytest.mark.parametrize(
    "lower,upper,selected",
    [
        ([], [], []),
        ([-2.0], [2.0, 2.0], [True]),
        ([-2.0], [2.0], [True, False]),
        ([-2.0], [2.0], [1]),
        ([-2.0], [2.0], [False]),
        ([np.nan], [2.0], [True]),
        ([-2.0], [np.inf], [True]),
        ([2.0], [2.0], [True]),
    ],
)
def test_invalid_or_empty_selection_rejected(
    lower: list, upper: list, selected: list
) -> None:
    with pytest.raises(ValueError):
        widen_control_envelope(
            np.asarray(lower), np.asarray(upper), np.asarray(selected), factor=2.0
        )


def test_overflowing_continuation_fails_without_infinite_bounds() -> None:
    with pytest.raises(ValueError, match="finite"):
        widen_control_envelope(
            np.array([-1e308]), np.array([1e308]), np.array([True]), factor=3.0
        )
