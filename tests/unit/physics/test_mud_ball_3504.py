"""Tests for the mud-ball aerodynamic adjustment model (issue #3504)."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from src.shared.python.physics.mud_ball import (
    MAX_CD_CLAMP,
    MAX_MUD_MASS_G,
    MudBallAdjustment,
    mud_ball_aero_adjustments,
)

pytestmark = pytest.mark.unit


# -- Baseline / zero-input behavior ------------------------------------------


def test_zero_coverage_zero_mud_returns_baseline_cd_cl() -> None:
    """With no mud and no coverage, Cd and Cl must equal their baselines."""
    res = mud_ball_aero_adjustments(mud_mass_g=0.0, mud_coverage=0.0)
    assert isinstance(res, MudBallAdjustment)
    assert res.cd_eff == pytest.approx(0.21)
    assert res.cl_eff == pytest.approx(0.18)
    # Mass equals clean ball mass (in kg).
    assert res.mass_total_kg == pytest.approx(0.04593)
    assert res.cd_increase_factor == pytest.approx(1.0)
    assert res.cl_decrease_factor == pytest.approx(1.0)


def test_full_coverage_increases_cd_decreases_cl_and_mass() -> None:
    """Full coverage with mud raises Cd, lowers Cl, and increases mass."""
    res = mud_ball_aero_adjustments(mud_mass_g=10.0, mud_coverage=1.0)
    assert res.cd_eff > 0.21
    assert res.cl_eff < 0.18
    assert res.mass_total_kg > 0.04593
    assert res.cd_increase_factor > 1.0
    assert res.cl_decrease_factor < 1.0


# -- Monotonicity / model structure ------------------------------------------


def test_cd_monotonic_non_decreasing_in_coverage() -> None:
    """Cd_eff must be non-decreasing as coverage rises (fixed mud mass)."""
    coverages = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    cds = [
        mud_ball_aero_adjustments(mud_mass_g=5.0, mud_coverage=c).cd_eff
        for c in coverages
    ]
    for prev, curr in pairwise(cds):
        assert curr >= prev, f"Cd dropped from {prev} to {curr}"


def test_cl_monotonic_non_increasing_in_coverage() -> None:
    """Cl_eff must be non-increasing as coverage rises."""
    coverages = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cls = [
        mud_ball_aero_adjustments(mud_mass_g=2.0, mud_coverage=c).cl_eff
        for c in coverages
    ]
    for prev, curr in pairwise(cls):
        assert curr <= prev, f"Cl rose from {prev} to {curr}"


def test_mass_only_contribution_does_not_change_cd_cl() -> None:
    """coverage=0 with mud mass changes only mass_total_kg, not Cd or Cl."""
    base = mud_ball_aero_adjustments(mud_mass_g=0.0, mud_coverage=0.0)
    mud_only = mud_ball_aero_adjustments(mud_mass_g=12.0, mud_coverage=0.0)
    assert mud_only.cd_eff == pytest.approx(base.cd_eff)
    assert mud_only.cl_eff == pytest.approx(base.cl_eff)
    assert mud_only.mass_total_kg > base.mass_total_kg
    assert mud_only.mass_total_kg == pytest.approx((45.93 + 12.0) / 1000.0)


def test_mass_factor_amplifies_cd_at_fixed_coverage() -> None:
    """At fixed nonzero coverage, more mud mass should increase Cd further."""
    light = mud_ball_aero_adjustments(mud_mass_g=1.0, mud_coverage=0.5)
    heavy = mud_ball_aero_adjustments(mud_mass_g=20.0, mud_coverage=0.5)
    assert heavy.cd_eff > light.cd_eff


def test_total_mass_formula() -> None:
    """Total mass equals (ball_mass_g + mud_mass_g) / 1000 kg."""
    res = mud_ball_aero_adjustments(
        mud_mass_g=7.5,
        mud_coverage=0.3,
        ball_mass_g=46.0,
    )
    assert res.mass_total_kg == pytest.approx((46.0 + 7.5) / 1000.0)


# -- Clamps ------------------------------------------------------------------


def test_cd_clamped_at_055_for_extreme_inputs() -> None:
    """Extreme coverage and mud must clamp Cd at MAX_CD_CLAMP (0.55)."""
    res = mud_ball_aero_adjustments(
        mud_mass_g=MAX_MUD_MASS_G,
        mud_coverage=1.0,
        base_cd=0.5,
    )
    assert res.cd_eff == pytest.approx(MAX_CD_CLAMP)
    assert res.cd_eff <= MAX_CD_CLAMP


def test_cl_clamped_at_zero_when_base_low_and_coverage_high() -> None:
    """Cl never goes negative; coverage=1.0 with the model floors it."""
    # With coverage=1.0, cl_raw = base_cl * (1 - 0.7) = 0.3 * base_cl, still > 0.
    # Force the clamp via a pathologically low base_cl combined with the
    # default model: choose base_cl that drives the multiplicand to <= 0.
    # The model multiplier minimum is (1 - 0.7) = 0.3 at coverage=1.0, so
    # any non-negative base_cl is reduced but never negative. Verify the
    # invariant cl_eff >= 0 across grid, including base_cl=0.
    res_zero_base = mud_ball_aero_adjustments(
        mud_mass_g=5.0,
        mud_coverage=1.0,
        base_cl=0.0,
    )
    assert res_zero_base.cl_eff == pytest.approx(0.0)
    assert res_zero_base.cl_decrease_factor == 0.0

    for c in (0.0, 0.5, 1.0):
        r = mud_ball_aero_adjustments(mud_mass_g=3.0, mud_coverage=c)
        assert r.cl_eff >= 0.0


# -- Invalid inputs ----------------------------------------------------------


def test_negative_mud_mass_raises_value_error() -> None:
    with pytest.raises(ValueError, match="mud_mass_g"):
        mud_ball_aero_adjustments(mud_mass_g=-0.1, mud_coverage=0.5)


def test_mud_mass_above_limit_raises_value_error() -> None:
    with pytest.raises(ValueError, match="mud_mass_g"):
        mud_ball_aero_adjustments(
            mud_mass_g=MAX_MUD_MASS_G + 0.1,
            mud_coverage=0.5,
        )


def test_coverage_above_one_raises_value_error() -> None:
    with pytest.raises(ValueError, match="mud_coverage"):
        mud_ball_aero_adjustments(mud_mass_g=5.0, mud_coverage=1.1)


def test_negative_coverage_raises_value_error() -> None:
    with pytest.raises(ValueError, match="mud_coverage"):
        mud_ball_aero_adjustments(mud_mass_g=5.0, mud_coverage=-0.01)


def test_nan_mud_mass_raises_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        mud_ball_aero_adjustments(mud_mass_g=math.nan, mud_coverage=0.5)


def test_inf_coverage_raises_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        mud_ball_aero_adjustments(mud_mass_g=5.0, mud_coverage=math.inf)


def test_none_required_arg_raises_type_error() -> None:
    with pytest.raises(TypeError):
        mud_ball_aero_adjustments(mud_mass_g=None, mud_coverage=0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        mud_ball_aero_adjustments(mud_mass_g=5.0, mud_coverage=None)  # type: ignore[arg-type]


def test_non_numeric_mud_mass_raises_type_error() -> None:
    with pytest.raises(TypeError):
        mud_ball_aero_adjustments(mud_mass_g="5.0", mud_coverage=0.5)  # type: ignore[arg-type]


def test_invalid_base_cd_raises_value_error() -> None:
    with pytest.raises(ValueError, match="base_cd"):
        mud_ball_aero_adjustments(
            mud_mass_g=1.0,
            mud_coverage=0.1,
            base_cd=0.0,
        )


def test_invalid_ball_mass_raises_value_error() -> None:
    with pytest.raises(ValueError, match="ball_mass_g"):
        mud_ball_aero_adjustments(
            mud_mass_g=1.0,
            mud_coverage=0.1,
            ball_mass_g=0.0,
        )


# -- Dataclass behavior ------------------------------------------------------


def test_result_is_frozen_dataclass() -> None:
    """The returned dataclass must be immutable."""
    res = mud_ball_aero_adjustments(mud_mass_g=2.0, mud_coverage=0.2)
    with pytest.raises((AttributeError, Exception)):
        res.cd_eff = 0.99  # type: ignore[misc]
