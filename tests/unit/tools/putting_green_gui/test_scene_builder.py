"""Tests for the headless putting-green scene builder.

These exercise the shared ``putting_dynamics`` wiring with no Qt
dependency. Holing configs are derived with :func:`solve_aim` on the
builder's own surface, never tuned to a model's output (epic #14141 7d).
"""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.putting_dynamics import PutterState, solve_aim
from src.tools.putting_green_gui import _scene_builder
from src.tools.putting_green_gui._scene_builder import (
    _FT_TO_M,
    PuttConfig,
    PuttScene,
    _setup_grid_and_surface,
    build_putt_scene,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Structure / invariants
# ---------------------------------------------------------------------------


def test_scene_has_consistent_grid_and_trajectory_shapes() -> None:
    scene = build_putt_scene(PuttConfig())
    nx, ny = scene.grid_x.size, scene.grid_y.size
    assert scene.grid_z.shape == (ny, nx)
    assert scene.trajectory_xyz.ndim == 2 and scene.trajectory_xyz.shape[1] == 3
    assert scene.trajectory_xyz.shape[0] == len(scene.roll_modes)
    assert scene.trajectory_xyz.shape[0] >= 1


def test_scene_metrics_are_finite_and_sane() -> None:
    scene = build_putt_scene(
        PuttConfig(putter_speed_ms=2.5, cup_distance_m=12.0 * _FT_TO_M)
    )
    assert np.isfinite(scene.total_roll_m) and scene.total_roll_m > 0.0
    assert np.isfinite(scene.duration_s) and scene.duration_s > 0.0
    assert np.isfinite(scene.final_distance_to_cup_m)
    assert scene.peak_break_m >= 0.0
    assert scene.launch_speed_ms > 0.0
    assert isinstance(scene.holed, bool)


def test_scene_lines_and_geometry_shapes() -> None:
    scene = build_putt_scene(PuttConfig())
    assert scene.aim_line_xyz.shape == (2, 3)
    assert scene.target_line_xyz.shape == (2, 3)
    assert scene.start_xyz.shape == (3,)
    assert scene.cup_xyz.shape == (3,)
    assert scene.hole_radius_m == pytest.approx(0.054, abs=1e-3)
    width, height = scene.green_size
    assert width > 0 and height > 0


def test_cup_lies_within_green_bounds() -> None:
    scene = build_putt_scene(PuttConfig(cup_distance_m=20.0 * _FT_TO_M))
    width, height = scene.green_size
    assert 0.0 <= scene.cup_xyz[0] <= width
    assert 0.0 <= scene.cup_xyz[1] <= height


def test_trajectory_rides_the_terrain() -> None:
    scene = build_putt_scene(PuttConfig(slope_deg=2.0, cup_distance_m=15.0 * _FT_TO_M))
    # Path z should track terrain elevation (within the ball-lift offset band).
    assert scene.trajectory_xyz[:, 2].min() >= scene.grid_z.min() - 0.05
    assert scene.trajectory_xyz[:, 2].max() <= scene.grid_z.max() + 0.1


# ---------------------------------------------------------------------------
# Physics behaviour
# ---------------------------------------------------------------------------


def _solved(distance_ft: float, slope_deg: float) -> tuple[PuttConfig, float]:
    """The config ``solve_aim`` prescribes on the builder's surface."""
    base = PuttConfig(cup_distance_m=distance_ft * _FT_TO_M, slope_deg=slope_deg)
    ball_x, cup_x, *_, mid_y, _gx, _gy, _gz, surface = _setup_grid_and_surface(
        base, base.cup_distance_m
    )
    aim = solve_aim(surface, (ball_x, mid_y), (cup_x, mid_y), PutterState.typical(1))
    assert aim.holed and aim.converged
    config = replace(
        base,
        putter_speed_ms=aim.putter_speed_mps,
        aim_deg=math.degrees(aim.heading_rad),
    )
    return config, aim.heading_rad


def test_solved_flat_straight_putt_holes() -> None:
    config, heading = _solved(12.0, 0.0)
    assert heading == pytest.approx(0.0, abs=1e-3)
    assert build_putt_scene(config).holed is True


def test_half_the_solved_pace_finishes_short() -> None:
    config, _ = _solved(12.0, 0.0)
    scene = build_putt_scene(
        replace(config, putter_speed_ms=config.putter_speed_ms / 2)
    )
    assert scene.holed is False
    assert scene.trajectory_xyz[-1, 0] < scene.cup_xyz[0]


def test_cross_slope_straight_putt_breaks_and_misses() -> None:
    scene = build_putt_scene(
        PuttConfig(
            putter_speed_ms=2.6,
            aim_deg=0.0,
            cup_distance_m=15.0 * _FT_TO_M,
            slope_deg=3.0,
        )
    )
    # A straight stroke on a 3-degree side-slope must break offline.
    assert scene.peak_break_m > 0.05
    assert scene.holed is False


def test_playing_the_break_aims_uphill_and_holes() -> None:
    # The cross-slope rises toward +y, so the read must aim above the hole.
    config, heading = _solved(15.0, 1.5)
    assert heading > 0.0
    assert build_putt_scene(config).holed is True


def test_scene_names_the_shared_roll_model() -> None:
    assert build_putt_scene(PuttConfig()).roll_model == "usga-stimp-roll/1"


def test_scene_builder_imports_nothing_from_the_legacy_engine() -> None:
    tree = ast.parse(inspect.getsource(_scene_builder))
    modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in modules if m.startswith("src.engines")]


def test_roll_modes_are_rollmode_labels() -> None:
    scene = build_putt_scene(PuttConfig())
    assert all(m.startswith("RollMode.") for m in scene.roll_modes)
    # A real putt slides briefly, then rolls, then stops.
    assert "RollMode.ROLLING" in scene.roll_modes


def test_build_is_deterministic() -> None:
    cfg = PuttConfig(
        putter_speed_ms=2.7, aim_deg=3.0, cup_distance_m=14.0 * _FT_TO_M, slope_deg=2.0
    )
    a = build_putt_scene(cfg)
    b = build_putt_scene(cfg)
    np.testing.assert_array_equal(a.trajectory_xyz, b.trajectory_xyz)
    assert a.holed == b.holed


def test_higher_stimp_rolls_farther() -> None:
    # A gentle, slightly-offline lag putt on a long flat green: the ball stops
    # short of the cup (no hole capture, no boundary clip), so total roll
    # reflects pure turf friction -- a faster green (higher stimp) rolls farther.
    common = {
        "putter_speed_ms": 0.9,
        "aim_deg": 6.0,
        "cup_distance_m": 30.0 * _FT_TO_M,
        "slope_deg": 0.0,
    }
    slow = build_putt_scene(PuttConfig(stimp=8.0, **common))
    fast = build_putt_scene(PuttConfig(stimp=13.0, **common))
    assert not slow.holed and not fast.holed
    assert fast.total_roll_m > slow.total_roll_m


# ---------------------------------------------------------------------------
# Design by Contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"putter_speed_ms": 0.0},
        {"putter_speed_ms": 99.0},
        {"aim_deg": -90.0},
        {"aim_deg": 90.0},
        {"cup_distance_m": 0.0},
        {"cup_distance_m": 100.0},
        {"stimp": 1.0},
        {"stimp": 20.0},
        {"slope_deg": -1.0},
        {"slope_deg": 10.0},
        {"timestep_s": 0.0},
        {"timestep_s": 0.1},
    ],
)
def test_out_of_range_controls_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        build_putt_scene(PuttConfig(**kwargs))


def test_non_finite_control_raises() -> None:
    with pytest.raises(ValueError):
        build_putt_scene(PuttConfig(putter_speed_ms=float("nan")))


def test_degenerate_grid_resolution_raises() -> None:
    with pytest.raises(ValueError):
        build_putt_scene(PuttConfig(grid_resolution=1))


def test_returns_putt_scene_instance() -> None:
    assert isinstance(build_putt_scene(PuttConfig()), PuttScene)
