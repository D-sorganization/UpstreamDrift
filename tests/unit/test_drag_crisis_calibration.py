"""Characterization tests for the golf-ball drag-crisis transition.

The drag crisis is the sharp drop in sphere Cd that occurs as the boundary
layer transitions from laminar to turbulent.  For a dimpled golf ball,
dimples trip turbulence earlier than a smooth sphere, so the crisis occurs
around Re ≈ 6e4–8e4.

These tests validate the Bearman-Harvey 1976 calibrated Cd(Re) curve
implemented in ``DragModel.get_effective_coefficient``.  The calibration
uses a natural cubic spline over empirical data from:

  Bearman, P. W. & Harvey, J. K. (1976). Golf ball aerodynamics.
  Aeronautical Quarterly, 27(2), 112-122.

Key behaviours tested:
- Pre-crisis (Re < 2e4): Cd = 0.50 (laminar plateau).
- Crisis region (Re 6e4–8e4): sharp Cd drop from ~0.50 toward ~0.22.
- Post-crisis region (Re > 8e4): Cd settles toward the turbulent plateau
  (~0.23–0.27).  Real data shows slight overshoot/recovery after the minimum
  (physically correct — the monotone-decrease property belonged only to the
  old piecewise linear model, not the real aerodynamics).
- Above data range (Re > 5e5): returns user-supplied base_coefficient.
- With reynolds_correction=False: always returns base_coefficient.
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

pytestmark = pytest.mark.unit


def _speed_for_reynolds(target_re: float) -> float:
    """Return the ball speed [m/s] that yields ``target_re`` at sea level."""
    diameter = 2.0 * float(GOLF_BALL_RADIUS_M)
    density = float(AIR_DENSITY_SEA_LEVEL_KG_M3)
    viscosity = float(AIR_VISCOSITY_KG_M_S)
    return target_re * viscosity / (density * diameter)


@pytest.mark.parametrize(
    ("target_re", "expected_cd", "atol"),
    [
        # Well below the data range: pre-crisis laminar plateau.
        (1.0e4, 0.50, 1e-9),
        # Mid-crisis: Cd should be well below the pre-crisis value.
        # Bearman-Harvey spline gives ~0.465 at Re=5e4 (still on the drop).
        (5.0e4, 0.465, 0.02),
        # At the crisis minimum region (~Re=7.2e4 in the data): Cd ≈ 0.22.
        (7.2e4, 0.22, 0.01),
        # Post-crisis but within range: turbulent plateau around 0.24–0.27.
        (2.0e5, 0.25, 0.04),
        # Above data range: must return base_coefficient exactly.
        (6.0e5, 0.25, 1e-9),
    ],
)
def test_drag_coefficient_calibrated_values(
    target_re: float,
    expected_cd: float,
    atol: float,
) -> None:
    """Pin Cd at representative Re values across the drag crisis.

    Tolerances are relaxed in the transition region (Re 5e4–2e5) because
    the spline interpolates between discretely sampled data points.
    """
    model = DragModel(base_coefficient=0.25, reynolds_correction=True)
    speed = _speed_for_reynolds(target_re)
    velocity = np.array([speed, 0.0, 0.0])

    cd = model.get_effective_coefficient(
        velocity,
        air_density=float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    )

    assert cd == pytest.approx(expected_cd, abs=atol), (
        f"Re={target_re:.2e}: expected Cd≈{expected_cd}, got {cd:.4f}"
    )


def test_pre_crisis_returns_laminar_plateau() -> None:
    """Well below the crisis, Cd must be the laminar plateau value (0.50)."""
    model = DragModel(base_coefficient=0.25, reynolds_correction=True)
    for target_re in (1.0e3, 1.0e4, 1.5e4):
        velocity = np.array([_speed_for_reynolds(target_re), 0.0, 0.0])
        cd = model.get_effective_coefficient(
            velocity,
            air_density=float(AIR_DENSITY_SEA_LEVEL_KG_M3),
        )
        assert cd == pytest.approx(0.50, abs=1e-9), (
            f"Re={target_re:.2e}: expected laminar Cd=0.50, got {cd:.4f}"
        )


def test_crisis_drops_below_laminar_value() -> None:
    """Through the crisis, Cd must fall substantially below the laminar value."""
    model = DragModel(base_coefficient=0.25, reynolds_correction=True)
    # At the crisis minimum (Re ~ 7e4), Cd should be well below 0.50.
    velocity = np.array([_speed_for_reynolds(7.2e4), 0.0, 0.0])
    cd = model.get_effective_coefficient(
        velocity,
        air_density=float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    )
    assert cd < 0.30, f"Crisis minimum Cd should be < 0.30, got {cd:.4f}"
    assert cd > 0.10, f"Cd should remain positive, got {cd:.4f}"


def test_above_data_range_returns_base_coefficient() -> None:
    """Well above the data range (Re > 5e5), Cd must equal base_coefficient."""
    base = 0.23
    model = DragModel(base_coefficient=base, reynolds_correction=True)
    velocity = np.array([_speed_for_reynolds(6.0e5), 0.0, 0.0])
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


def test_cd_strictly_less_than_laminar_after_crisis() -> None:
    """Once past the crisis, Cd must remain below the pre-crisis plateau."""
    model = DragModel(base_coefficient=0.25, reynolds_correction=True)
    # After Re=1e5, the ball is firmly in the turbulent regime.
    for target_re in (1.0e5, 1.5e5, 2.0e5, 3.0e5, 4.0e5, 5.0e5):
        velocity = np.array([_speed_for_reynolds(target_re), 0.0, 0.0])
        cd = model.get_effective_coefficient(
            velocity,
            air_density=float(AIR_DENSITY_SEA_LEVEL_KG_M3),
        )
        assert cd < 0.45, (
            f"Re={target_re:.2e}: post-crisis Cd={cd:.4f} should be < 0.45"
        )
