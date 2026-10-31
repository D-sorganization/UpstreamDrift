"""Offset-aware hole capture (epic #14141, step 6).

Holmes (1991; equation reproduced by Long 2020) bounds capture by the
time the ball spends over the hole: it must fall half a diameter while
crossing the chord it travels. A ball entering at perpendicular offset
``b`` from the hole centre crosses a chord ``2 sqrt(R^2 - b^2)``, so its
bound shrinks from the dead-centre value toward zero at the rim. Every
expected value here is that closed form, never a model run.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.core.physics_constants import GOLF_BALL_RADIUS_M, GRAVITY_M_S2
from src.shared.python.putting_dynamics import (
    HOLE_RADIUS_M,
    BallState,
    SurfaceSpec,
    capture_speed_mps,
    simulate_ball,
)

pytestmark = pytest.mark.unit

_R = float(GOLF_BALL_RADIUS_M)


def _closed_form(offset_m: float) -> float:
    chord = 2.0 * math.sqrt(HOLE_RADIUS_M**2 - offset_m**2)
    return chord * math.sqrt(float(GRAVITY_M_S2) / (2.0 * _R))


def test_centre_bound_is_the_full_chord_value() -> None:
    assert capture_speed_mps() == pytest.approx(_closed_form(0.0), rel=1e-12)
    assert capture_speed_mps(0.0) == capture_speed_mps()


@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8, 0.95])
def test_offset_bound_matches_the_chord_closed_form(fraction: float) -> None:
    offset = fraction * HOLE_RADIUS_M
    assert capture_speed_mps(offset) == pytest.approx(_closed_form(offset), rel=1e-12)
    assert capture_speed_mps(-offset) == capture_speed_mps(offset)


def test_bound_decreases_monotonically_toward_the_rim() -> None:
    bounds = [capture_speed_mps(f * HOLE_RADIUS_M) for f in (0.0, 0.3, 0.6, 0.9)]
    assert all(a > b for a, b in zip(bounds, bounds[1:], strict=False))


def test_offset_outside_the_hole_is_rejected() -> None:
    with pytest.raises(ValueError):
        capture_speed_mps(HOLE_RADIUS_M)
    with pytest.raises(ValueError):
        capture_speed_mps(math.nan)


def _roll_into_hole(speed_mps: float, offset_m: float) -> bool:
    """Start a pure-rolling ball just short of the hole on a straight line."""
    start_x = -math.sqrt(HOLE_RADIUS_M**2 - offset_m**2) - 0.002
    ball = BallState(
        x_m=start_x, y_m=offset_m, vx_mps=speed_mps, spin_rad_s=speed_mps / _R
    )
    surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
    return simulate_ball(ball, surface, hole_x_m=0.0, hole_y_m=0.0).holed


def test_centred_ball_below_the_centre_bound_drops() -> None:
    assert _roll_into_hole(0.9 * capture_speed_mps(), 0.0)


def test_same_speed_lips_out_near_the_rim() -> None:
    offset = 0.8 * HOLE_RADIUS_M
    speed = 0.9 * capture_speed_mps()
    assert speed > capture_speed_mps(offset)
    assert not _roll_into_hole(speed, offset)


def test_slow_ball_near_the_rim_still_drops() -> None:
    offset = 0.8 * HOLE_RADIUS_M
    assert _roll_into_hole(0.8 * capture_speed_mps(offset), offset)


def test_lip_out_records_entry_speed_and_rolls_past() -> None:
    offset = 0.8 * HOLE_RADIUS_M
    speed = 0.9 * capture_speed_mps()
    start_x = -math.sqrt(HOLE_RADIUS_M**2 - offset**2) - 0.002
    ball = BallState(x_m=start_x, y_m=offset, vx_mps=speed, spin_rad_s=speed / _R)
    surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
    result = simulate_ball(ball, surface, hole_x_m=0.0, hole_y_m=0.0)
    assert not result.holed
    assert result.speed_at_hole_mps == pytest.approx(speed, rel=0.01)
    assert result.rest_x_m > HOLE_RADIUS_M


def test_ball_that_misses_the_hole_records_no_entry() -> None:
    ball = BallState(x_m=-1.0, y_m=1.2 * HOLE_RADIUS_M, vx_mps=1.0, spin_rad_s=1.0 / _R)
    surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
    result = simulate_ball(ball, surface, hole_x_m=0.0, hole_y_m=0.0)
    assert not result.holed
    assert result.speed_at_hole_mps is None
