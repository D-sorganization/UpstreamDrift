"""The offline reference's future support must be explicit at a seam."""

import numpy as np
import pytest

from src.shared.python.motion_matching.impact_force import (
    reference_rates,
    sample_rate_table,
)
from src.shared.python.motion_matching.reference_support import reference_support

pytestmark = pytest.mark.unit


def test_unsplit_derivatives_include_future_rows():
    times = np.arange(9.0)
    support = reference_support(times, 3.0)
    assert 4 in support.velocity and 5 in support.acceleration
    with pytest.raises(ValueError, match="future"):
        support.require_past_only()
    final = reference_support(times, 8.0)
    final.require_past_only()


def test_impact_split_separates_rates_but_position_still_interpolates():
    times = np.arange(9.0)
    q = np.zeros((9, 1))
    changed = q.copy()
    changed[4] = 1.0
    split_time = 3.5
    support = reference_support(times, 3.25, split_time_s=split_time)
    assert 4 in support.position
    assert max(support.velocity) <= 3 and max(support.acceleration) <= 3
    v0, a0, last = reference_rates(times, q, split_time)
    v1, a1, _ = reference_rates(times, changed, split_time)
    assert sample_rate_table(times, changed, 3.25, None)[0] > 0
    np.testing.assert_array_equal(
        sample_rate_table(times, v0, 3.25, (split_time, last)),
        sample_rate_table(times, v1, 3.25, (split_time, last)),
    )
    np.testing.assert_array_equal(
        sample_rate_table(times, a0, 3.25, (split_time, last)),
        sample_rate_table(times, a1, 3.25, (split_time, last)),
    )
    with pytest.raises(ValueError, match="future"):
        support.require_past_only()


def test_support_contains_every_observed_basis_dependency():
    times = np.array([0.0, 0.2, 0.5, 0.8, 1.1, 1.5, 2.0])
    for split in (None, 0.9):
        velocity, acceleration, last = reference_rates(times, np.eye(len(times)), split)
        split_tuple = None if last is None else (split, last)
        for t in (0.0, 0.3, 0.8, 0.95, 2.0):
            support = reference_support(times, t, split_time_s=split)
            vectors = (
                sample_rate_table(times, np.eye(len(times)), t, None),
                sample_rate_table(times, velocity, t, split_tuple),
                sample_rate_table(times, acceleration, t, split_tuple),
            )
            for observed, predicted in zip(
                vectors,
                (support.position, support.velocity, support.acceleration),
                strict=True,
            ):
                assert set(np.flatnonzero(observed)) <= set(predicted)


@pytest.mark.parametrize("times", [[0, 1, np.nan], [0, 0, 1], [0], [-1e308, 1e308]])
def test_invalid_clocks_fail_closed(times):
    with pytest.raises(ValueError):
        reference_support(times, 0.0)
