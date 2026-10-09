"""Face and path impact mapping through the existing impact and flight solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.ball_simulator import BallFlightSimulator
from src.shared.python.physics.impact_model import ImpactSolverAPI, PreImpactState

if TYPE_CHECKING:
    from .core import AnalysisConfig


@dataclass(frozen=True)
class ShotOutcome:
    """Carry location and launch state in the target frame, metres and degrees."""

    carry_x_m: float
    carry_y_m: float
    carry_m: float
    launch_azimuth_deg: float
    spin_rpm: float
    spin_axis_tilt_deg: float
    trajectory_xy_m: tuple[tuple[float, float], ...] = ()


class ShotPhysics:
    """Adapter: one impact and one aerodynamic trajectory per shot.

    Target-frame right is positive in this API. The physics engine uses +y
    left, so the sign conversion occurs at the input and output boundary.
    """

    def __init__(self) -> None:
        self._impact = ImpactSolverAPI()
        self._flight = BallFlightSimulator()

    def simulate(
        self,
        *,
        face_deg: float,
        path_deg: float,
        config: AnalysisConfig,
        sample_trajectory: bool = False,
    ) -> ShotOutcome:
        if not all(math.isfinite(v) for v in (face_deg, path_deg)):
            raise ValueError("face_deg and path_deg must be finite")
        if abs(face_deg) >= 30 or abs(path_deg) >= 30:
            raise ValueError("face and path must be within 30 degrees of target")

        path = math.radians(path_deg)
        face = math.radians(face_deg)
        loft = math.radians(config.loft_deg)
        # Golf-positive right -> world-negative y.
        club_v = config.club_speed_mps * np.array(
            [math.cos(path), -math.sin(path), 0.0]
        )
        normal = np.array(
            [
                math.cos(loft) * math.cos(face),
                -math.cos(loft) * math.sin(face),
                math.sin(loft),
            ]
        )
        pre = PreImpactState(
            clubhead_velocity=club_v,
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=normal,
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_loft=loft,
            impact_offset=None,
        )
        post = self._impact.solve_pre_impact_state(
            timestamp=0.0, pre_state=pre, record=False
        )
        vx, vy, vz = (float(v) for v in post.ball_velocity)
        wx, wy, wz = (float(v) for v in post.ball_angular_velocity)
        speed = math.hypot(vx, vy, vz)
        spin_rad_s = math.hypot(wx, wy, wz)
        if speed <= 0 or spin_rad_s <= 0:
            raise RuntimeError("impact produced invalid launch speed or spin")
        launch = LaunchConditions(
            velocity=speed,
            launch_angle=math.atan2(vz, math.hypot(vx, vy)),
            azimuth_angle=math.atan2(vy, vx),
            spin_rate=spin_rad_s * 60 / (2 * math.pi),
            spin_axis=np.asarray(post.ball_angular_velocity, dtype=float) / spin_rad_s,
        )
        trajectory = self._flight.simulate_trajectory(
            launch, max_time=config.max_time_s, dt=config.dt_s
        )
        if len(trajectory) < 2:
            raise RuntimeError("flight did not produce a landing trajectory")
        end = trajectory[-1].position
        prev = trajectory[-2].position
        if float(end[2]) > 0:
            raise RuntimeError("flight exceeded max_time_s before landing")
        fraction = float(prev[2] / (prev[2] - end[2])) if prev[2] > 0 else 1.0
        landing = prev + fraction * (end - prev)
        x = float(landing[0])
        y = -float(landing[1])
        sampled: tuple[tuple[float, float], ...] = ()
        if sample_trajectory:
            indices = np.linspace(0, len(trajectory) - 2, 40, dtype=int)
            sampled = tuple(
                (float(trajectory[i].position[0]), -float(trajectory[i].position[1]))
                for i in indices
            ) + ((x, y),)
        return ShotOutcome(
            carry_x_m=x,
            carry_y_m=y,
            carry_m=math.hypot(x, y),
            launch_azimuth_deg=-math.degrees(launch.azimuth_angle),
            spin_rpm=launch.spin_rate,
            spin_axis_tilt_deg=math.degrees(math.atan2(wz, -wy)),
            trajectory_xy_m=sampled,
        )
