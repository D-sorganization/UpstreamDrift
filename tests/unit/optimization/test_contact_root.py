"""Contracts for the bracketed, approaching signed-distance contact root (#15493 Slice 1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.optimization.contact_root import (
    ContactBracketError,
    ContactRoot,
    closest_approach_penalty,
    find_contact_root,
)

pytestmark = pytest.mark.unit


def _approach(t: float) -> float:
    """Clearance shrinking at 2 m/s from 1 m: root at t = 0.5 s."""
    return 1.0 - 2.0 * t


def test_converges_to_analytic_root_within_tolerance():
    root = find_contact_root(_approach, 0.0, 1.0, xtol=1e-12)
    assert isinstance(root, ContactRoot)
    assert root.time_s == pytest.approx(0.5, abs=1e-10)
    assert abs(root.distance_m) < 1e-9
    assert root.closing_speed_mps == pytest.approx(2.0, rel=1e-6)


def test_nonlinear_root_matches_reference():
    def dist(t: float) -> float:
        return math.cos(t) - 0.3

    root = find_contact_root(dist, 0.0, 2.0, xtol=1e-12)
    assert root.time_s == pytest.approx(math.acos(0.3), abs=1e-9)
    assert root.bracket_width_s <= 1e-9


def test_is_deterministic():
    a = find_contact_root(_approach, 0.0, 1.0)
    b = find_contact_root(_approach, 0.0, 1.0)
    assert a == b


def test_touching_at_upper_end_is_a_valid_bracket():
    root = find_contact_root(lambda t: 1.0 - t, 0.0, 1.0)
    assert root.time_s == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "dist",
    [
        lambda t: 1.0 + t,  # never reaches contact: no sign change
        lambda t: -1.0 - t,  # already penetrating at the start
        lambda t: 0.0,  # touching at the start is not an approach
    ],
)
def test_rejects_unbracketed_intervals(dist):
    with pytest.raises(ContactBracketError):
        find_contact_root(dist, 0.0, 1.0)


def test_rejects_separating_crossing():
    # Negative to positive: the club is leaving, not approaching.
    with pytest.raises(ContactBracketError):
        find_contact_root(lambda t: t - 0.5, 0.0, 1.0)


def test_rejects_non_finite_distance_and_bad_interval():
    with pytest.raises(ValueError, match="finite"):
        find_contact_root(lambda t: float("nan"), 0.0, 1.0)
    with pytest.raises(ValueError, match="t1_s"):
        find_contact_root(_approach, 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        find_contact_root(_approach, 0.0, float("inf"))
    with pytest.raises(ValueError, match="xtol"):
        find_contact_root(_approach, 0.0, 1.0, xtol=0.0)


def test_miss_penalty_is_zero_on_contact():
    assert closest_approach_penalty(_approach, 0.0, 1.0) == 0.0


def test_miss_penalty_is_finite_smooth_and_monotone_in_clearance():
    def miss(gap: float):
        return lambda t: gap + (t - 0.5) ** 2

    penalties = [
        closest_approach_penalty(miss(g), 0.0, 1.0) for g in (0.01, 0.02, 0.05)
    ]
    assert all(np.isfinite(penalties))
    assert 0.0 < penalties[0] < penalties[1] < penalties[2]
    # Closest approach is the minimum on the sampled grid, here the gap itself.
    assert penalties[0] == pytest.approx(0.01**2, rel=1e-6)


def test_miss_penalty_rejects_bad_inputs():
    with pytest.raises(ValueError, match="samples"):
        closest_approach_penalty(_approach, 0.0, 1.0, samples=1)
    with pytest.raises(ValueError, match="finite"):
        closest_approach_penalty(lambda t: float("nan"), 0.0, 1.0)
