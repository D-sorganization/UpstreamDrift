"""Normalised launch-monitor shot dataclass."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.shared.python.physics.ball_flight_physics import LaunchConditions

# Conversion factors
_MPH_TO_MPS = 0.44704
_YARDS_TO_METERS = 0.9144
_RPM_TO_RAD_S = 2.0 * math.pi / 60.0


@dataclass
class LaunchMonitorShot:
    """Normalised representation of a single launch-monitor shot.

    All velocities are in m/s, distances in metres, angles in degrees,
    and spin rates in rpm — the units used internally throughout the
    simulation suite.  Adapters are responsible for unit conversion.

    Attributes:
        club: Club name (e.g. "Driver", "7-Iron", "PW").
        ball_speed_mps: Ball speed immediately post-impact [m/s].
        club_speed_mps: Clubhead speed at impact [m/s].
        smash_factor: ball_speed / club_speed (dimensionless).
        launch_angle_deg: Vertical launch angle [degrees].
        launch_direction_deg: Horizontal launch direction relative to
            target line [degrees]; positive = right for right-handed player.
        back_spin_rpm: Backspin component [rpm]; positive = backspin.
        side_spin_rpm: Sidespin component [rpm]; positive = fade (right).
        spin_axis_deg: Spin axis tilt [degrees]; convention matches TrackMan
            (positive = draw / right-to-left axis for RH player).
        total_spin_rpm: Resultant spin magnitude [rpm].
        carry_m: Carry distance [metres].
        total_m: Total distance including run [metres]; ``None`` if not recorded.
        max_height_m: Apex height [metres]; ``None`` if not recorded.
        landing_angle_deg: Descent angle at landing [degrees]; ``None`` if not recorded.
        flight_time_s: Ball flight time [seconds]; ``None`` if not recorded.
        attack_angle_deg: Club attack angle at impact [degrees]; negative = down.
        dynamic_loft_deg: Dynamic loft at impact [degrees]; ``None`` if not recorded.
        club_path_deg: Club path direction [degrees]; ``None`` if not recorded.
        face_angle_deg: Face angle at impact [degrees]; ``None`` if not recorded.
        source: Name of the launch monitor that produced this shot.
        shot_id: Optional shot identifier from the source file.
        extra: Any extra fields from the source file, keyed by column name.
    """

    club: str
    ball_speed_mps: float
    club_speed_mps: float
    smash_factor: float
    launch_angle_deg: float
    launch_direction_deg: float
    back_spin_rpm: float
    side_spin_rpm: float
    spin_axis_deg: float
    total_spin_rpm: float
    carry_m: float
    total_m: float | None = None
    max_height_m: float | None = None
    landing_angle_deg: float | None = None
    flight_time_s: float | None = None
    attack_angle_deg: float = 0.0
    dynamic_loft_deg: float | None = None
    club_path_deg: float | None = None
    face_angle_deg: float | None = None
    source: str = ""
    shot_id: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ball_speed_mps < 0:
            raise ValueError("ball_speed_mps must be non-negative")
        if self.club_speed_mps < 0:
            raise ValueError("club_speed_mps must be non-negative")
        if self.carry_m < 0:
            raise ValueError("carry_m must be non-negative")
        if self.total_spin_rpm < 0:
            raise ValueError("total_spin_rpm must be non-negative")

    def to_launch_conditions(self) -> LaunchConditions:
        """Convert to :class:`~src.shared.python.physics.ball_flight_physics.LaunchConditions`.

        The spin axis is decomposed into a unit vector in the XZ plane
        (ball travels in +X, vertical is +Z) rotated by ``spin_axis_deg``
        around the +X axis.  Back-spin (positive) creates a spin vector
        pointing in the -Y direction (matches the convention in the Rust
        kernel).

        Returns:
            LaunchConditions compatible with BallFlightSimulator.
        """
        from src.shared.python.physics.ball_flight_physics import LaunchConditions

        axis_rad = math.radians(self.spin_axis_deg)
        spin_axis = np.array([0.0, -math.cos(axis_rad), math.sin(axis_rad)])
        norm = float(np.linalg.norm(spin_axis))
        if norm > 1e-10:
            spin_axis /= norm

        return LaunchConditions(
            velocity=self.ball_speed_mps,
            launch_angle=math.radians(self.launch_angle_deg),
            azimuth_angle=math.radians(self.launch_direction_deg),
            spin_rate=self.total_spin_rpm,
            spin_axis=spin_axis,
        )
