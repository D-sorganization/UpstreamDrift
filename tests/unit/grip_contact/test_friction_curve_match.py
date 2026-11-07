"""Derived OpenSim static friction: Hollars curve fit to the shared law (#13560)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.grip_contact.friction_curve_match import (
    hollars_curve,
    matched_static_friction,
    shared_law_curve,
)

pytestmark = pytest.mark.unit


def test_matches_reference_values_for_default_law() -> None:
    """us=0.9, ud=0.7 gives a fitted static value near 0.664, below ud."""
    mu_s = matched_static_friction(0.9, 0.7)
    assert mu_s == pytest.approx(0.664, abs=0.005)


def test_result_is_bounded_near_the_dynamic_value() -> None:
    """The fit stays within 20 % of mu_d when the Stribeck peak is absent."""
    assert matched_static_friction(0.7, 0.7) == pytest.approx(0.7, rel=0.2)


def test_fit_beats_the_old_static_equals_dynamic_mapping() -> None:
    r = np.linspace(1e-6, 10.0, 2000)
    ref = shared_law_curve(r, 0.9, 0.7)
    fit = hollars_curve(r, 0.7, matched_static_friction(0.9, 0.7))
    old = hollars_curve(r, 0.7, 0.7)
    assert np.sqrt(np.mean((fit - ref) ** 2)) < np.sqrt(np.mean((old - ref) ** 2))


@pytest.mark.parametrize("bad", [(-0.1, 0.7), (0.9, 0.0), (float("nan"), 0.7)])
def test_rejects_invalid_coefficients(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="friction"):
        matched_static_friction(*bad)


def test_deterministic_and_finite() -> None:
    a = matched_static_friction(1.1, 0.8)
    assert math.isfinite(a)
    assert a == matched_static_friction(1.1, 0.8)
