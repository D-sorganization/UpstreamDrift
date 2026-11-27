"""Spin-dependent drag law of the ball-flight kernel (#15443).

The kernel drag coefficient was a function of Reynolds number only, so one
base value could not fit low-spin drivers and high-spin irons together.  The
law is now ``Cd(Re, S) = Cd_Re(Re; floor) + k * S``: the Reynolds drag-crisis
curve on a production-ball floor plus a linear spin term.  The Python
fallback must stay identical to the Rust kernel.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.shared.python.core.physics_constants import (
    GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT,
    GOLF_BALL_SPIN_DRAG_SLOPE,
)
from src.shared.python.physics.aerodynamics import _rust_facade
from src.shared.python.physics.aerodynamics._rust_facade import (
    AerodynamicsSpec,
    _python_fallback_total,
)
from src.shared.python.physics.atmosphere import cd_dimpled_sphere, cd_spinning_ball

pytestmark = pytest.mark.unit

# Published dCd/dS estimates (see GOLF_BALL_SPIN_DRAG_SLOPE.source): Bearman &
# Harvey (1976) read by Crabill et al. (2018) give about 0.13; the summary in
# US 10118696 B1 (Cd 0.27 -> 0.32 for S 0.1 -> 0.3) gives 0.25.
SPIN_SLOPE_PUBLISHED_RANGE = (0.13, 0.25)
# Lyu et al. (2018): non-spinning production balls, Cd close to 0.2 at Re 1e5,
# ball-to-ball spread below 0.05 above Re 1e5.
PRODUCTION_FLOOR_MEASURED = 0.20
PRODUCTION_FLOOR_SPREAD = 0.025
REYNOLDS_GRID = (2.0e4, 5.0e4, 7.0e4, 1.0e5, 1.5e5, 2.2e5, 6.0e5)
SPIN_RATIO_PROBE = 0.2
REYNOLDS_PROBE = 1.5e5


def _spec(*, magnus: bool = False) -> AerodynamicsSpec:
    return AerodynamicsSpec(
        mass=0.04593,
        radius=0.021335,
        drag_coefficient=float(GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT),
        spin_decay_rate=0.05,
        air_density=1.225,
        lift_enabled=magnus,
        magnus_enabled=magnus,
    )


def test_drag_law_constants_carry_sources() -> None:
    for constant in (GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT, GOLF_BALL_SPIN_DRAG_SLOPE):
        assert constant.source.strip() and constant.description.strip()


def test_floor_matches_production_ball_measurement() -> None:
    floor = float(GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT)
    assert abs(floor - PRODUCTION_FLOOR_MEASURED) <= PRODUCTION_FLOOR_SPREAD


def test_spin_slope_within_published_range() -> None:
    low, high = SPIN_SLOPE_PUBLISHED_RANGE
    assert low <= float(GOLF_BALL_SPIN_DRAG_SLOPE) <= high


@pytest.mark.parametrize("reynolds", REYNOLDS_GRID)
def test_zero_spin_reduces_to_reynolds_curve(reynolds: float) -> None:
    floor = float(GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT)
    assert cd_spinning_ball(reynolds, 0.0, floor) == cd_dimpled_sphere(reynolds, floor)


def test_spin_adds_linear_term() -> None:
    floor = float(GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT)
    added = cd_spinning_ball(
        REYNOLDS_PROBE, SPIN_RATIO_PROBE, floor
    ) - cd_dimpled_sphere(REYNOLDS_PROBE, floor)
    assert added == pytest.approx(float(GOLF_BALL_SPIN_DRAG_SLOPE) * SPIN_RATIO_PROBE)


def test_disabled_drag_stays_zero_with_spin() -> None:
    assert cd_spinning_ball(REYNOLDS_PROBE, SPIN_RATIO_PROBE, 0.0) == 0.0


@pytest.mark.parametrize(
    ("spin_ratio", "error"),
    [(-0.1, ValueError), (math.nan, ValueError), ("0.1", TypeError)],
)
def test_rejects_invalid_spin_ratio(spin_ratio: object, error: type) -> None:
    with pytest.raises(error):
        cd_spinning_ball(REYNOLDS_PROBE, spin_ratio, 0.2)  # type: ignore[arg-type]


def test_python_fallback_drag_uses_spin_term() -> None:
    spec = _spec()
    speed, omega = 60.0, 300.0
    velocity = np.array([speed, 0.0, 0.0])
    force = _python_fallback_total(spec, velocity, np.array([0.0, -omega, 0.0]))
    reynolds = spec.air_density * speed * 2.0 * spec.radius / spec.air_viscosity
    cd = cd_spinning_ball(reynolds, spec.radius * omega / speed, spec.drag_coefficient)
    area = math.pi * spec.radius**2
    assert force[0] == pytest.approx(-0.5 * spec.air_density * cd * area * speed**2)


def test_production_force_path_drag_depends_on_spin() -> None:
    """The routed path (Rust when installed) sees spin in drag, like the fallback."""
    velocity = np.array([60.0, 0.0, 5.0])
    spin = np.array([0.0, -300.0, 0.0])
    with_spin = _rust_facade.compute_total_force(_spec(), velocity, spin)
    no_spin = _rust_facade.compute_total_force(_spec(), velocity, np.zeros(3))
    assert np.linalg.norm(with_spin) > np.linalg.norm(no_spin)
    np.testing.assert_allclose(
        with_spin, _python_fallback_total(_spec(), velocity, spin), rtol=1e-12
    )
