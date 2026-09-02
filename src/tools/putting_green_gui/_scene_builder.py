"""Headless scene builder for the putting green simulator.

This module owns *all* of the putting-green domain logic for the GUI: it
turns user-facing controls (putter speed, aim, cup distance, stimp, slope)
into a real :class:`PuttingGreenSimulator` run and packages the result as a
render-ready :class:`PuttScene`.  Keeping it free of any Qt/OpenGL import means
the physics wiring is exhaustively unit-testable in a headless environment,
and the GUI layer stays a thin renderer (DRY / Law of Demeter).

Design by Contract:
    * :func:`build_putt_scene` validates every control against its documented
      range and raises ``ValueError`` with a descriptive message.
    * Postcondition: the returned scene always carries a non-empty trajectory
      whose points lie on the rendered terrain grid, and always names the roll
      model that produced it (ADR-0045 F1, issue #9343).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.engines.physics_engines.putting_green.python.green_surface import GreenSurface
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.simulator import (
    PuttingGreenSimulator,
    SimulationConfig,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)

_FT_TO_M = 0.3048
_BALL_LIFT_M = 0.021  # half a golf-ball diameter, so the path rides the surface

# Documented control ranges (mirrored by the GUI spin-box bounds).
SPEED_RANGE_MS = (0.5, 8.0)
AIM_RANGE_DEG = (-45.0, 45.0)
DISTANCE_RANGE_FT = (1.0, 30.0)
STIMP_RANGE = (6.0, 14.0)
SLOPE_RANGE_DEG = (0.0, 5.0)


@dataclass(frozen=True)
class PuttConfig:
    """User-facing putt configuration (matches the GUI controls)."""

    putter_speed_ms: float = 2.5
    aim_deg: float = 0.0
    cup_distance_ft: float = 10.0
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
        ("cup_distance_ft", config.cup_distance_ft, DISTANCE_RANGE_FT),
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


def _setup_grid_and_green(
    config: PuttConfig, dist_m: float
) -> tuple[
    float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, GreenSurface
]:
    margin_back, margin_front = 0.9, 1.4
    ball_x = margin_back
    cup_x = ball_x + dist_m
    width_x = cup_x + margin_front
    height_y = max(4.0, dist_m * 0.6)
    mid_y = height_y / 2.0

    res = config.grid_resolution
    nx = max(2, int(round(res * width_x / max(width_x, height_y))))
    ny = max(2, int(round(res * height_y / max(width_x, height_y))))
    grid_x = np.linspace(0.0, width_x, nx)
    grid_y = np.linspace(0.0, height_y, ny)
    grid_z = _cross_slope_heightmap(grid_x, grid_y, mid_y, config.slope_deg)

    turf = TurfProperties(stimp_rating=config.stimp)
    green = GreenSurface(width=width_x, height=height_y, turf=turf)
    green.set_heightmap(grid_z, smooth=False)
    green.set_hole_position(np.array([cup_x, mid_y]))
    return ball_x, cup_x, width_x, height_y, mid_y, grid_x, grid_y, grid_z, green


def _compute_scene_trajectory(
    result: object,
    green: GreenSurface,
    ball_xy: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    positions = np.asarray(getattr(result, "positions", []), dtype=float).reshape(-1, 2)
    if positions.shape[0] == 0:
        positions = ball_xy.reshape(1, 2)
    elevations = np.array([green.get_elevation_at(p) for p in positions])
    trajectory_xyz = np.column_stack([positions, elevations + _BALL_LIFT_M])

    modes = getattr(result, "modes", None) or []
    roll_modes = [str(m) for m in modes]
    if len(roll_modes) < positions.shape[0]:
        roll_modes += ["RollMode.STOPPED"] * (positions.shape[0] - len(roll_modes))
    roll_modes = roll_modes[: positions.shape[0]]
    return trajectory_xyz, roll_modes


def _compute_scene_lines(
    green: GreenSurface,
    ball_xy: np.ndarray,
    ball_x: float,
    mid_y: float,
    cup_x: float,
    direction: np.ndarray,
    dist_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cup_z = float(green.get_elevation_at(np.array([cup_x, mid_y])))
    start_z = float(green.get_elevation_at(ball_xy))
    start_xyz = np.array([ball_x, mid_y, start_z + _BALL_LIFT_M])
    cup_xyz = np.array([cup_x, mid_y, cup_z])

    aim_end = ball_xy + direction * dist_m
    aim_line_xyz = np.array(
        [
            [ball_x, mid_y, start_z + _BALL_LIFT_M],
            [
                aim_end[0],
                aim_end[1],
                float(green.get_elevation_at(aim_end)) + _BALL_LIFT_M,
            ],
        ]
    )
    target_line_xyz = np.array(
        [[ball_x, mid_y, start_z + _BALL_LIFT_M], [cup_x, mid_y, cup_z + _BALL_LIFT_M]]
    )
    return start_xyz, cup_xyz, aim_line_xyz, target_line_xyz


def build_putt_scene(config: PuttConfig) -> PuttScene:
    """Run the real putting-green engine and package a render-ready scene.

    Args:
        config: Validated putt configuration.

    Returns:
        A :class:`PuttScene` with terrain, trajectory, and summary metrics.

    Raises:
        ValueError: If any control is outside its documented range.
    """
    _validate(config)

    dist_m = config.cup_distance_ft * _FT_TO_M
    (
        ball_x,
        cup_x,
        width_x,
        height_y,
        mid_y,
        grid_x,
        grid_y,
        grid_z,
        green,
    ) = _setup_grid_and_green(config, dist_m)

    sim = PuttingGreenSimulator(
        green=green,
        config=SimulationConfig(
            integrator=config.integrator,
            record_trajectory=True,
            timestep=config.timestep_s,
        ),
    )

    aim_rad = np.radians(config.aim_deg)
    direction = np.array([np.cos(aim_rad), np.sin(aim_rad)])
    stroke = StrokeParameters(speed=config.putter_speed_ms, direction=direction)
    ball_xy = np.array([ball_x, mid_y])
    result = sim.simulate_putt(stroke, ball_position=ball_xy)

    trajectory_xyz, roll_modes = _compute_scene_trajectory(result, green, ball_xy)
    positions = trajectory_xyz[:, :2]

    start_xyz, cup_xyz, aim_line_xyz, target_line_xyz = _compute_scene_lines(
        green, ball_xy, ball_x, mid_y, cup_x, direction, dist_m
    )

    final_xy = np.asarray(result.final_position, dtype=float).reshape(-1)[:2]
    final_distance = float(np.hypot(*(final_xy - np.array([cup_x, mid_y]))))
    launch_speed = (
        float(np.hypot(*np.asarray(result.velocities[0], dtype=float).reshape(-1)[:2]))
        if len(result.velocities)
        else config.putter_speed_ms
    )

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
        hole_radius_m=float(green.hole_radius),
        green_size=(width_x, height_y),
        holed=bool(result.holed),
        final_distance_to_cup_m=final_distance,
        total_roll_m=float(result.total_distance),
        duration_s=float(result.duration),
        peak_break_m=_perpendicular_break(positions, ball_xy, np.array([cup_x, mid_y])),
        launch_speed_ms=launch_speed,
        roll_model=result.roll_model,
    )
