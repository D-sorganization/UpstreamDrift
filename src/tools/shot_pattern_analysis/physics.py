"""Face and path impact mapping through the existing impact and flight solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.ball_simulator import BallFlightSimulator
from src.shared.python.physics.impact_model import ImpactSolverAPI, PreImpactState

from .delivery_geometry import delivery_from_face_angle

if TYPE_CHECKING:
    from .core import AnalysisConfig


@dataclass(frozen=True)
class ShotOutcome:
    """Carry location and launch state in metres/degrees.

    Face, path, launch azimuth, and lateral carry are positive right. The
    native spin-axis tilt is positive for draw/left curvature (+world z);
    its sign is opposite a right-positive spin-tilt convention.
    """

    carry_x_m: float
    carry_y_m: float
    carry_m: float
    launch_azimuth_deg: float
    spin_rpm: float
    spin_axis_tilt_deg: float
    trajectory_xy_m: tuple[tuple[float, float], ...] = ()
    launch_elevation_deg: float = 0.0
    ball_speed_mps: float = 0.0


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
        nominal_face_deg: float = 0.0,
    ) -> ShotOutcome:
        if not all(math.isfinite(v) for v in (face_deg, path_deg, nominal_face_deg)):
            raise ValueError("face, path, and nominal face must be finite")
        if any(abs(v) >= 30 for v in (face_deg, path_deg, nominal_face_deg)):
            raise ValueError("face, path, and nominal face must be within 30 degrees")

        path = math.radians(path_deg)
        face = math.radians(face_deg)
        attack = math.radians(config.attack_angle_deg)
        if config.delivery_mode == "shaft_rotation":
            delivered = delivery_from_face_angle(
                face_deg - nominal_face_deg,
                base_loft_deg=config.loft_deg,
                lie_deg=config.lie_deg,
                shaft_lean_deg=config.shaft_lean_deg,
            )
            loft = math.radians(delivered.dynamic_loft_deg)
        else:
            loft = math.radians(config.loft_deg)
        # Golf-positive right -> world-negative y.
        club_v = config.club_speed_mps * np.array(
            [
                math.cos(attack) * math.cos(path),
                -math.cos(attack) * math.sin(path),
                math.sin(attack),
            ]
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
            clubhead_mass=config.clubhead_mass_kg,
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
        sampled: tuple[tuple[float, float], ...] = ()
        if sample_trajectory:
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
            indices = np.linspace(0, len(trajectory) - 2, 40, dtype=int)
            sampled = tuple(
                (float(trajectory[i].position[0]), -float(trajectory[i].position[1]))
                for i in indices
            ) + ((float(landing[0]), -float(landing[1])),)
        else:
            try:
                landing = self._flight.simulate_landing(
                    launch, max_time=config.max_time_s, dt=config.dt_s
                )
            except RuntimeError as exc:
                if "exceeded max_time" not in str(exc):
                    raise
                raise RuntimeError("flight exceeded max_time_s before landing") from exc
        x = float(landing[0])
        y = -float(landing[1])
        return ShotOutcome(
            carry_x_m=x,
            carry_y_m=y,
            carry_m=math.hypot(x, y),
            launch_azimuth_deg=-math.degrees(launch.azimuth_angle),
            spin_rpm=launch.spin_rate,
            spin_axis_tilt_deg=math.degrees(math.atan2(wz, -wy)),
            trajectory_xy_m=sampled,
            launch_elevation_deg=math.degrees(launch.launch_angle),
            ball_speed_mps=speed,
        )
