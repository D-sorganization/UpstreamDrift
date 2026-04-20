"""Characterization tests for the golf-ball drag-crisis transition.

The drag crisis is the sharp drop in sphere Cd that occurs as the boundary
layer transitions from laminar to turbulent. For a smooth sphere this
occurs near Re ~= 3e5, but for a dimpled golf ball dimples trip turbulence
earlier, around Re ~= 4e4 - 1e5.

Context: UpstreamDrift #2803 is a salvage task for the closed-unmerged
PR #2732, which attempted a Bearman-Harvey / Smits-Ogg calibration of
Cd(Re) but was not merged and not independently verified. Until a
reviewed calibration is landed, this file pins the CURRENT coarse
3-segment approximation used in both ``aerodynamics.DragModel`` and
``engines.common.physics.AerodynamicsCalculator``.

These tests are intentionally characterization tests: they exist so that
a future real calibration does not silently change behavior. When the
calibration lands, update the expected values here alongside that change
and document the source (Bearman & Harvey 1976; Smits & Ogg 2004; etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    AIR_VISCOSITY_KG_M_S,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.physics.aerodynamics import DragModel


def _speed_for_reynolds(target_re: float) -> float:
    """Return the ball speed [m/s] that yields ``target_re`` at sea level."""
    diameter = 2.0 * float(GOLF_BALL_RADIUS_M)
    density = float(AIR_DENSITY_SEA_LEVEL_KG_M3)
    viscosity = float(AIR_VISCOSITY_KG_M_S)
    return target_re * viscosity / (density * diameter)


@pytest.mark.parametrize(
    ("target_re", "expected_cd"),
    [
        # Well below the transition: pre-crisis laminar value.
        (5.0e4, 0.5),
        # Right at the lower edge of the modeled transition.
        (8.0e4, 0.5),
        # Midpoint of the modeled transition region.
        (1.4e5, 0.5 - 0.5 * (0.5 - 0.25)),
        # Well above the transition: post-crisis turbulent value (default Cd=0.25).
        (3.0e5, 0.25),
    ],
)
def test_drag_coefficient_pinned_across_crisis(
    target_re: float,
    expected_cd: float,
) -> None:
    """Pin Cd at representative Reynolds numbers across the drag crisis.

    See module docstring and TRACKED(#2803) in aerodynamics/_models.py.
    """
    model = DragModel(base_coefficient=0.25, reynolds_correction=True)
    speed = _speed_for_reynolds(target_re)
    velocity = np.array([speed, 0.0, 0.0])

    cd = model.get_effective_coefficient(
        velocity,
        air_density=float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    )

    assert cd == pytest.approx(expected_cd, rel=1e-9, abs=1e-9)


def test_drag_coefficient_monotonic_non_increasing_through_crisis() -> None:
    """Cd must never increase as Re increases through the crisis region.

    This invariant should survive any future recalibration: real golf-ball
    Cd(Re) curves are non-increasing across the drag crisis (the whole
    point of the crisis is that Cd drops).
    """
    model = DragModel(base_coefficient=0.25, reynolds_correction=True)
    reynolds_samples = np.linspace(5.0e4, 3.5e5, 40)

    previous = float("inf")
    for re in reynolds_samples:
        speed = _speed_for_reynolds(float(re))
        velocity = np.array([speed, 0.0, 0.0])
        cd = model.get_effective_coefficient(
            velocity,
            air_density=float(AIR_DENSITY_SEA_LEVEL_KG_M3),
        )
        assert cd <= previous + 1e-12, (
            f"Cd must be non-increasing across the drag crisis; "
            f"Re={re:.3e} gave Cd={cd} after previous Cd={previous}"
        )
        previous = cd


def test_post_crisis_matches_base_coefficient() -> None:
    """Well above the crisis, Cd should equal the user-supplied base value."""
    base = 0.23
    model = DragModel(base_coefficient=base, reynolds_correction=True)
    velocity = np.array([_speed_for_reynolds(5.0e5), 0.0, 0.0])
    cd = model.get_effective_coefficient(
        velocity,
        air_density=float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    )
    assert cd == pytest.approx(base)


def test_reynolds_correction_disabled_returns_base() -> None:
    """With correction disabled, Cd is constant regardless of Re."""
    base = 0.25
    model = DragModel(base_coefficient=base, reynolds_correction=False)
    for target_re in (1.0e4, 1.0e5, 1.0e6):
        velocity = np.array([_speed_for_reynolds(target_re), 0.0, 0.0])
        cd = model.get_effective_coefficient(
            velocity,
            air_density=float(AIR_DENSITY_SEA_LEVEL_KG_M3),
        )
        assert cd == pytest.approx(base)
