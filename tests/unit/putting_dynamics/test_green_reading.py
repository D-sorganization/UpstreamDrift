"""Simulation-based green reading (epic #14141, step 7b).

Expected values are geometric facts (a plane's elevation and gradient,
mirror symmetry, the direct line on a flat green) or the pure-roll
stopping law, never a value read back from a previous solve.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.putting_dynamics import (
    FrictionField,
    FrictionParams,
    HeightField,
    PutterState,
    SurfaceSpec,
    read_line,
    simulate_strike,
    solve_aim,
)

pytestmark = pytest.mark.unit

_PUTTER = PutterState(head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=3.0, speed_mps=1.0)


def _planar(grade_percent: float, aspect_deg: float) -> SurfaceSpec:
    flat = SurfaceSpec.flat_uniform(stimp_ft=10.0)
    return SurfaceSpec(
        height=HeightField.planar(grade_percent, aspect_deg, extent_m=20.0),
        friction_field=FrictionField.uniform(extent_m=20.0),
        friction=FrictionParams(mu_roll0=flat.friction.mu_roll0),
    )


def test_read_line_samples_a_plane_exactly() -> None:
    grade, aspect = 2.0, 30.0
    reading = read_line(_planar(grade, aspect), (-3.0, -1.0), (3.0, 2.0), 7)
    assert reading.distance_m == pytest.approx(math.hypot(6.0, 3.0))
    assert reading.positions_m[0] == (-3.0, -1.0)
    assert reading.positions_m[-1] == pytest.approx((3.0, 2.0))
    # A plane falling toward `aspect` has gradient -grade/100 along it.
    expected = (
        -grade / 100 * math.cos(math.radians(aspect)),
        -grade / 100 * math.sin(math.radians(aspect)),
    )
    for slope in reading.slopes:
        assert slope == pytest.approx(expected, abs=1e-9)
    steps = [
        b - a
        for a, b in zip(reading.elevations_m, reading.elevations_m[1:], strict=False)
    ]
    assert steps == pytest.approx([steps[0]] * len(steps), abs=1e-9)


def test_read_line_rejects_bad_inputs() -> None:
    surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
    with pytest.raises(ValueError):
        read_line(surface, (0.0, 0.0), (1.0, 0.0), num_samples=1)
    with pytest.raises(ValueError):
        read_line(surface, (math.nan, 0.0), (1.0, 0.0))


def test_flat_green_aims_straight_at_the_hole() -> None:
    surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
    start, hole = (-2.0, -1.0), (1.0, 3.0)
    solution = solve_aim(surface, start, hole, _PUTTER, past_hole_m=0.3)
    assert solution.converged
    assert solution.heading_rad == pytest.approx(math.atan2(4.0, 3.0), abs=1e-3)
    assert solution.break_m == pytest.approx(0.0, abs=5e-3)
    assert solution.holed  # a dead-straight putt at dying pace drops


def test_flat_green_pace_finishes_past_the_hole_without_the_cup() -> None:
    surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
    solution = solve_aim(surface, (-2.0, 0.0), (1.0, 0.0), _PUTTER, past_hole_m=0.3)
    free = simulate_strike(
        PutterState(
            head_mass_kg=0.35,
            moi_kg_m2=4.5e-4,
            loft_deg=3.0,
            speed_mps=solution.putter_speed_mps,
        ),
        surface,
        start_x_m=-2.0,
        heading_rad=solution.heading_rad,
    )
    assert free.rest_x_m == pytest.approx(1.3, abs=0.01)


@pytest.mark.parametrize("aspect_deg", [90.0, -90.0])
def test_cross_slope_aims_uphill(aspect_deg: float) -> None:
    # The line runs along +x; aspect +90 falls toward +y (left of the line).
    solution = solve_aim(_planar(2.0, aspect_deg), (-2.0, 0.0), (1.5, 0.0), _PUTTER)
    assert solution.converged
    downhill_left = aspect_deg > 0.0
    assert (solution.break_m < 0.0) == downhill_left


def test_cross_slope_break_is_mirror_symmetric() -> None:
    left = solve_aim(_planar(2.0, 90.0), (-2.0, 0.0), (1.5, 0.0), _PUTTER)
    right = solve_aim(_planar(2.0, -90.0), (-2.0, 0.0), (1.5, 0.0), _PUTTER)
    assert left.break_m == pytest.approx(-right.break_m, abs=0.01)
    assert left.putter_speed_mps == pytest.approx(right.putter_speed_mps, rel=0.01)


def test_solve_aim_rejects_a_start_on_the_hole() -> None:
    surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
    with pytest.raises(ValueError):
        solve_aim(surface, (1.0, 1.0), (1.0, 1.0), _PUTTER)
    with pytest.raises(ValueError):
        solve_aim(surface, (0.0, 0.0), (1.0, 0.0), _PUTTER, past_hole_m=0.0)
