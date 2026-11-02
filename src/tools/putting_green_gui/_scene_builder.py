"""Headless scene builder for the putting green simulator.

This module owns *all* of the putting-green domain logic for the GUI: it
turns user-facing controls (putter speed, aim, cup distance, stimp, slope)
into a real strike-and-roll simulation on the shared
:mod:`src.shared.python.putting_dynamics` core (epic #14141 step 7d;
ADR-0058) and packages the result as a render-ready :class:`PuttScene`.
Keeping it free of any Qt/OpenGL import means the physics wiring is
exhaustively unit-testable in a headless environment, and the GUI layer
stays a thin renderer (DRY / Law of Demeter).

Design by Contract:
    * :func:`build_putt_scene` validates every control against its documented
      range and raises ``ValueError`` with a descriptive message.
    * Postcondition: the returned scene always carries a non-empty trajectory
      whose points lie on the rendered terrain grid, and always names the roll
      model that produced it (ADR-0045 F1, issue #9343).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.putting_dynamics import (
    HOLE_RADIUS_M,
    FrictionField,
    FrictionParams,
    HeightField,
    Mode,
    PutterState,
    USGA_STIMP_ROLL_MODEL,
    PuttResult,
    SurfaceSpec,
    simulate_strike,
    stimp_to_rolling_mu,
)

_FT_TO_M = 0.3048
_BALL_LIFT_M = 0.021  # half a golf-ball diameter, so the path rides the surface

#: Shared-solver mode -> legacy ``RollMode``-style label (export/colour code
#: dependency). The shared solver's brief post-impact AIRBORNE hop has no
#: legacy analogue; it is rendered with the SLIDING colour since the ball
#: has not yet settled into a pure roll.
_ROLL_MODE_LABELS = {
    Mode.AIRBORNE: "RollMode.SLIDING",
    Mode.SLIDE: "RollMode.SLIDING",
    Mode.ROLL: "RollMode.ROLLING",
    Mode.REST: "RollMode.STOPPED",
}

# Documented control ranges (mirrored by the GUI spin-box bounds).  All
# internal state is SI (issue #8886); the historical 1-30 ft distance range
# is preserved exactly by converting its bounds through ``_FT_TO_M``.
SPEED_RANGE_MS = (0.5, 8.0)
AIM_RANGE_DEG = (-45.0, 45.0)
DISTANCE_RANGE_M = (1.0 * _FT_TO_M, 30.0 * _FT_TO_M)
STIMP_RANGE = (6.0, 14.0)
SLOPE_RANGE_DEG = (0.0, 5.0)


@dataclass(frozen=True)
class PuttConfig:
    """User-facing putt configuration (matches the GUI controls).

    All fields are SI (metres, m/s, degrees) per the repository's unit
    policy (issue #8886) -- there is no mixed-unit-system field here.

    ``integrator`` and ``timestep_s`` are kept for API compatibility with
    existing callers; the shared solver (epic #14141 step 7d) always
    integrates with its own fixed-step RK4 (``putting_dynamics.solver.DT_S``)
    and does not consult either field.
    """

    putter_speed_ms: float = 2.5
    aim_deg: float = 0.0
    cup_distance_m: float = 10.0 * _FT_TO_M
    stimp: float = 10.0
    slope_deg: float = 1.0
    integrator: str = "rk4"
    grid_resolution: int = 48
    timestep_s: float = 0.002


@dataclass(frozen=True)
class PuttScene:
    """Render-ready result of a putt simulation.

    All coordinates are in metres in the green's local frame (x = down the
    nominal ball→cup line, y = lateral, z = elevation).
    """

    grid_x: np.ndarray  # (nx,)
    grid_y: np.ndarray  # (ny,)
    grid_z: np.ndarray  # (ny, nx)
    trajectory_xyz: np.ndarray  # (N, 3)
    roll_modes: list[str]  # length N
    start_xyz: np.ndarray  # (3,)
    cup_xyz: np.ndarray  # (3,)
    aim_line_xyz: np.ndarray  # (2, 3) ball -> aim direction
    target_line_xyz: np.ndarray  # (2, 3) ball -> cup (the straight read)
    hole_radius_m: float
    green_size: tuple[float, float]  # (width_x, height_y)
    holed: bool
    final_distance_to_cup_m: float
    total_roll_m: float
    duration_s: float
    peak_break_m: float
    launch_speed_ms: float
    #: Roll model that produced this scene (ADR-0045 F1); scenes from
    #: different models must never be compared without it.
    roll_model: str


def _validate(config: PuttConfig) -> None:
    checks = (
        ("putter_speed_ms", config.putter_speed_ms, SPEED_RANGE_MS),
        ("aim_deg", config.aim_deg, AIM_RANGE_DEG),
        ("cup_distance_m", config.cup_distance_m, DISTANCE_RANGE_M),
        ("stimp", config.stimp, STIMP_RANGE),
        ("slope_deg", config.slope_deg, SLOPE_RANGE_DEG),
    )
    for name, value, (lo, hi) in checks:
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value!r}")
        if not (lo <= value <= hi):
            raise ValueError(f"{name} must be within [{lo}, {hi}]; got {value}")
    if config.grid_resolution < 2:
        raise ValueError(f"grid_resolution must be >= 2; got {config.grid_resolution}")
    if not (0.0 < config.timestep_s <= 0.05):
        raise ValueError(f"timestep_s must be in (0, 0.05]; got {config.timestep_s}")


def _cross_slope_heightmap(
    grid_x: np.ndarray, grid_y: np.ndarray, ball_y: float, slope_deg: float
) -> np.ndarray:
    """Elevation grid for a uniform cross-slope (a side-tilt that breaks putts).

    The ball→cup line sits on the zero-elevation contour so the break is read
    purely as a lateral tilt, the way a player reads a green.
    """
    tan_slope = np.tan(np.radians(slope_deg))
    _, yy = np.meshgrid(grid_x, grid_y)
    return (yy - ball_y) * tan_slope


def _perpendicular_break(
    trajectory_xy: np.ndarray, start: np.ndarray, cup: np.ndarray
) -> float:
    """Peak lateral deviation of the path from the straight ball→cup line."""
    line = cup - start
    length = float(np.hypot(line[0], line[1]))
    if length < 1e-9:
        return 0.0
    rel = trajectory_xy - start
    cross = np.abs(rel[:, 0] * line[1] - rel[:, 1] * line[0]) / length
    return float(np.max(cross)) if cross.size else 0.0


def _setup_grid_and_surface(
    config: PuttConfig, dist_m: float
) -> tuple[
    float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, SurfaceSpec
]:
    margin_back, margin_front = 0.9, 1.4
    ball_x = margin_back
    cup_x = ball_x + dist_m
    width_x = cup_x + margin_front
    height_y = max(4.0, dist_m * 0.6)
    mid_y = height_y / 2.0

    res = config.grid_resolution
    max_dim = max(width_x, height_y)
    spacing = min(max(max_dim / (res - 1), 0.001), 10.0)
    nx = max(2, int(round(width_x / spacing)) + 1)
    ny = max(2, int(round(height_y / spacing)) + 1)
    grid_x = np.arange(nx) * spacing
    grid_y = np.arange(ny) * spacing
    grid_z = _cross_slope_heightmap(grid_x, grid_y, mid_y, config.slope_deg)

    surface = SurfaceSpec(
        height=HeightField(heights_m=grid_z, spacing_m=spacing, origin_m=(0.0, 0.0)),
        friction_field=FrictionField.uniform(),
        friction=FrictionParams(mu_roll0=stimp_to_rolling_mu(config.stimp)),
    )
    return ball_x, cup_x, width_x, height_y, mid_y, grid_x, grid_y, grid_z, surface


def _compute_scene_trajectory(
    result: PuttResult,
    height_field: HeightField,
) -> tuple[np.ndarray, list[str]]:
    samples = result.samples
    positions = np.array([(s.x_m, s.y_m) for s in samples], dtype=float).reshape(-1, 2)
    floor = np.array([height_field.elevation(x, y) for x, y in positions])
    airborne = np.array([s.height_m for s in samples], dtype=float)
    trajectory_xyz = np.column_stack([positions, floor + airborne + _BALL_LIFT_M])
    roll_modes = [_ROLL_MODE_LABELS[s.mode] for s in samples]
    return trajectory_xyz, roll_modes


def _compute_scene_lines(
    height_field: HeightField,
    ball_xy: np.ndarray,
    ball_x: float,
    mid_y: float,
    cup_x: float,
    direction: np.ndarray,
    dist_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cup_z = height_field.elevation(cup_x, mid_y)
    start_z = height_field.elevation(ball_xy[0], ball_xy[1])
    start_xyz = np.array([ball_x, mid_y, start_z + _BALL_LIFT_M])
    cup_xyz = np.array([cup_x, mid_y, cup_z])

    aim_end = ball_xy + direction * dist_m
    aim_line_xyz = np.array(
        [
            [ball_x, mid_y, start_z + _BALL_LIFT_M],
            [
                aim_end[0],
                aim_end[1],
                height_field.elevation(aim_end[0], aim_end[1]) + _BALL_LIFT_M,
            ],
        ]
    )
    target_line_xyz = np.array(
        [[ball_x, mid_y, start_z + _BALL_LIFT_M], [cup_x, mid_y, cup_z + _BALL_LIFT_M]]
    )
    return start_xyz, cup_xyz, aim_line_xyz, target_line_xyz


def build_putt_scene(config: PuttConfig) -> PuttScene:
    """Strike and roll the ball on the shared putting core and package a scene.

    Args:
        config: Validated putt configuration.

    Returns:
        A :class:`PuttScene` with terrain, trajectory, and summary metrics.

    Raises:
        ValueError: If any control is outside its documented range.
    """
    _validate(config)

    dist_m = config.cup_distance_m
    (
        ball_x,
        cup_x,
        width_x,
        height_y,
        mid_y,
        grid_x,
        grid_y,
        grid_z,
        surface,
    ) = _setup_grid_and_surface(config, dist_m)
    height_field = surface.height

    aim_rad = math.radians(config.aim_deg)
    direction = np.array([math.cos(aim_rad), math.sin(aim_rad)])
    putter = PutterState.typical(config.putter_speed_ms)
    ball_xy = np.array([ball_x, mid_y])
    result = simulate_strike(
        putter,
        surface,
        hole_x_m=cup_x,
        hole_y_m=mid_y,
        start_x_m=ball_x,
        start_y_m=mid_y,
        heading_rad=aim_rad,
    )

    trajectory_xyz, roll_modes = _compute_scene_trajectory(result, height_field)
    positions = trajectory_xyz[:, :2]

    start_xyz, cup_xyz, aim_line_xyz, target_line_xyz = _compute_scene_lines(
        height_field, ball_xy, ball_x, mid_y, cup_x, direction, dist_m
    )

    cup_xy = np.array([cup_x, mid_y])
    final_distance = float(math.hypot(result.rest_x_m - cup_x, result.rest_y_m - mid_y))
    launch_speed = result.samples[0].speed_mps

    return PuttScene(
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        trajectory_xyz=trajectory_xyz,
        roll_modes=roll_modes,
        start_xyz=start_xyz,
        cup_xyz=cup_xyz,
        aim_line_xyz=aim_line_xyz,
        target_line_xyz=target_line_xyz,
        hole_radius_m=HOLE_RADIUS_M,
        green_size=(width_x, height_y),
        holed=bool(result.holed),
        final_distance_to_cup_m=final_distance,
        total_roll_m=float(result.total_distance_m),
        duration_s=float(result.time_s),
        peak_break_m=_perpendicular_break(positions, ball_xy, cup_xy),
        launch_speed_ms=launch_speed,
        roll_model=USGA_STIMP_ROLL_MODEL,
    )
