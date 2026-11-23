"""Bunker release to flight and an exploratory flat-green rollout."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from bunkershot3d.ball import BallLaunchResult
from src.shared.python.physics.ball_flight_validity import assess_trajectory_validity
from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.ball_simulator import BallFlightSimulator
from src.shared.python.putting_dynamics import (
    BallState,
    Mode,
    SurfaceSpec,
    simulate_ball,
)


@dataclass(frozen=True)
class BunkerFlight:
    """SI coordinates, x forward, y left, z up; rollout is an assumed model."""

    flight_m: tuple[tuple[float, float, float], ...]
    rollout_m: tuple[tuple[float, float, float], ...]
    carry_m: float
    assumption: str = (
        "Uncalibrated launch; flat 10 ft stimp green, simplified bounce and spin"
    )


def bunker_flight(
    launch: BallLaunchResult, *, refused: bool, dt: float = 0.01
) -> BunkerFlight | None:
    """Fly a valid release and use the existing green solver; fail closed on invalid data."""
    if refused:
        return None
    values = (
        launch.ball_speed_m_s,
        launch.launch_angle_rad,
        launch.azimuth_rad,
        launch.spin_rate_rpm,
        *launch.spin_axis,
    )
    if (
        not all(math.isfinite(v) for v in values)
        or launch.ball_speed_m_s <= 0
        or launch.spin_rate_rpm < 0
    ):
        return None
    if (
        not 0 < launch.launch_angle_rad <= math.pi / 2
        or abs(launch.azimuth_rad) > math.pi
    ):
        return None
    axis = np.asarray(launch.spin_axis, dtype=float)
    if axis.shape != (3,) or not math.isclose(
        float(np.linalg.norm(axis)), 1.0, abs_tol=1e-6
    ):
        return None
    sim = BallFlightSimulator()
    conditions = LaunchConditions(
        launch.ball_speed_m_s,
        launch.launch_angle_rad,
        launch.azimuth_rad,
        launch.spin_rate_rpm,
        axis,
    )
    points = sim.simulate_trajectory(conditions, max_time=10.0, dt=dt)
    if len(points) < 2 or points[-1].position[2] > 0 or points[-2].position[2] <= 0:
        return None
    a, b = points[-2:]
    fraction = float(a.position[2] / (a.position[2] - b.position[2]))
    landing = a.position + fraction * (b.position - a.position)
    velocity = a.velocity + fraction * (b.velocity - a.velocity)
    landing_time = a.time + fraction * (b.time - a.time)
    path_array = np.asarray([p.position for p in points[:-1]] + [landing])
    if (
        not np.isfinite(path_array).all()
        or not assess_trajectory_validity(path_array).valid
    ):
        return None
    if (
        not np.isfinite(landing).all()
        or not np.isfinite(velocity).all()
        or velocity[2] >= 0
    ):
        return None
    heading = math.atan2(float(velocity[1]), float(velocity[0]))
    spin = (
        launch.spin_rate_rpm
        * 2
        * math.pi
        / 60
        * math.exp(-sim.ball.spin_decay_rate * landing_time)
    )
    transverse = -math.sin(heading) * axis[0] + math.cos(heading) * axis[1]
    try:
        state = BallState(
            float(landing[0]),
            float(landing[1]),
            height_m=1e-9,
            vx_mps=float(velocity[0]),
            vy_mps=float(velocity[1]),
            vz_mps=float(velocity[2]),
            spin_rad_s=float(spin * transverse),
        )
        result = simulate_ball(state, SurfaceSpec.flat_uniform(stimp_ft=10.0))
    except ValueError:
        return None
    if result.final_mode is not Mode.REST:
        return None
    landing[2] = 0.0
    path = tuple(
        (float(p.position[0]), float(p.position[1]), float(p.position[2]))
        for p in points[:-1]
    ) + ((float(landing[0]), float(landing[1]), 0.0),)
    rollout = ((float(landing[0]), float(landing[1]), 0.0),) + tuple(
        (s.x_m, s.y_m, s.height_m) for s in result.samples[1:]
    )
    return BunkerFlight(path, rollout, math.hypot(float(landing[0]), float(landing[1])))
