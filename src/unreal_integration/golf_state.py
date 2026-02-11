"""Golf-specific state data models for Unreal Engine integration.

Provides ClubState, SwingMetrics, BallState, TrajectoryPoint, and
EnvironmentState data classes for golf swing simulation and visualization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .geometry import Vector3


@dataclass
class ClubState:
    """State of the golf club during swing.

    Attributes:
        head_position: Position of club head.
        head_velocity: Velocity of club head.
        head_acceleration: Acceleration of club head (optional).
        shaft_flex: List of shaft deflection values along shaft (optional).
        face_angle: Club face angle in degrees (optional).
        loft_angle: Dynamic loft angle in degrees (optional).
        lie_angle: Dynamic lie angle in degrees (optional).
        shaft_lean: Shaft lean angle in degrees (optional).

    Example:
        >>> cs = ClubState(
        ...     head_position=Vector3(x=0.5, y=0.8, z=0.1),
        ...     head_velocity=Vector3(x=25.0, y=10.0, z=5.0),
        ... )
        >>> cs.head_speed
        27.386...
    """

    head_position: Vector3
    head_velocity: Vector3
    head_acceleration: Vector3 | None = None
    shaft_flex: list[float] | None = None
    face_angle: float | None = None
    loft_angle: float | None = None
    lie_angle: float | None = None
    shaft_lean: float | None = None

    @property
    def head_speed(self) -> float:
        """Calculate club head speed (magnitude of velocity).

        Returns:
            Club head speed in m/s.
        """
        return self.head_velocity.magnitude

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "head_position": self.head_position.to_dict(),
            "head_velocity": self.head_velocity.to_dict(),
            "head_speed": self.head_speed,
        }
        if self.head_acceleration is not None:
            result["head_acceleration"] = self.head_acceleration.to_dict()
        if self.shaft_flex is not None:
            result["shaft_flex"] = self.shaft_flex
        if self.face_angle is not None:
            result["face_angle"] = self.face_angle
        if self.loft_angle is not None:
            result["loft_angle"] = self.loft_angle
        if self.lie_angle is not None:
            result["lie_angle"] = self.lie_angle
        if self.shaft_lean is not None:
            result["shaft_lean"] = self.shaft_lean
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClubState:
        """Create ClubState from dictionary.

        Args:
            d: Dictionary representation.

        Returns:
            New ClubState instance.
        """
        return cls(
            head_position=Vector3.from_dict(d["head_position"]),
            head_velocity=Vector3.from_dict(d["head_velocity"]),
            head_acceleration=(
                Vector3.from_dict(d["head_acceleration"])
                if "head_acceleration" in d
                else None
            ),
            shaft_flex=d.get("shaft_flex"),
            face_angle=d.get("face_angle"),
            loft_angle=d.get("loft_angle"),
            lie_angle=d.get("lie_angle"),
            shaft_lean=d.get("shaft_lean"),
        )


@dataclass
class SwingMetrics:
    """Real-time swing analysis metrics.

    Attributes:
        club_head_speed: Club head speed in m/s.
        x_factor: X-factor (hip-shoulder separation) in degrees.
        kinetic_energy: Total kinetic energy in Joules.
        smash_factor: Ball speed / club head speed ratio.
        attack_angle: Attack angle in degrees (negative = down).
        swing_path: Swing path in degrees (positive = out-to-in).
        face_to_path: Face angle relative to path in degrees.
        tempo: Backswing to downswing time ratio.
        hip_speed: Peak hip rotational speed in deg/s.
        shoulder_speed: Peak shoulder rotational speed in deg/s.
        wrist_release_angle: Wrist release angle in degrees.

    Example:
        >>> sm = SwingMetrics(club_head_speed=45.0, smash_factor=1.5)
        >>> sm.estimated_ball_speed
        67.5
    """

    club_head_speed: float | None = None
    x_factor: float | None = None
    kinetic_energy: float | None = None
    smash_factor: float | None = None
    attack_angle: float | None = None
    swing_path: float | None = None
    face_to_path: float | None = None
    tempo: float | None = None
    hip_speed: float | None = None
    shoulder_speed: float | None = None
    wrist_release_angle: float | None = None

    @property
    def estimated_ball_speed(self) -> float | None:
        """Calculate estimated ball speed from club head speed and smash factor.

        Returns:
            Estimated ball speed in m/s, or None if data unavailable.
        """
        if self.club_head_speed is not None and self.smash_factor is not None:
            return self.club_head_speed * self.smash_factor
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes None values).

        Returns:
            Dictionary representation with only non-None values.
        """
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        if self.estimated_ball_speed is not None:
            result["estimated_ball_speed"] = self.estimated_ball_speed
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SwingMetrics:
        """Create SwingMetrics from dictionary.

        Args:
            d: Dictionary representation.

        Returns:
            New SwingMetrics instance.
        """
        return cls(
            club_head_speed=d.get("club_head_speed"),
            x_factor=d.get("x_factor"),
            kinetic_energy=d.get("kinetic_energy"),
            smash_factor=d.get("smash_factor"),
            attack_angle=d.get("attack_angle"),
            swing_path=d.get("swing_path"),
            face_to_path=d.get("face_to_path"),
            tempo=d.get("tempo"),
            hip_speed=d.get("hip_speed"),
            shoulder_speed=d.get("shoulder_speed"),
            wrist_release_angle=d.get("wrist_release_angle"),
        )


@dataclass
class BallState:
    """State of the golf ball.

    Attributes:
        position: Ball position in world space.
        velocity: Ball velocity.
        spin_rate: Spin rate in RPM.
        spin_axis: Spin axis direction.
        is_in_flight: Whether ball is currently airborne.

    Example:
        >>> bs = BallState(
        ...     position=Vector3.zero(),
        ...     velocity=Vector3(x=100.0, y=0.0, z=100.0),
        ... )
        >>> bs.launch_angle
        45.0
    """

    position: Vector3
    velocity: Vector3
    spin_rate: float = 0.0
    spin_axis: Vector3 | None = None
    is_in_flight: bool = False

    @property
    def launch_angle(self) -> float:
        """Calculate launch angle in degrees.

        Returns:
            Launch angle (angle from horizontal).
        """
        horizontal_speed = math.sqrt(self.velocity.x**2 + self.velocity.y**2)
        if horizontal_speed == 0:
            return 90.0 if self.velocity.z > 0 else -90.0
        return math.degrees(math.atan2(self.velocity.z, horizontal_speed))

    @property
    def ball_speed(self) -> float:
        """Calculate ball speed.

        Returns:
            Ball speed in m/s.
        """
        return self.velocity.magnitude

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "spin_rate": self.spin_rate,
            "is_in_flight": self.is_in_flight,
            "launch_angle": self.launch_angle,
            "ball_speed": self.ball_speed,
        }
        if self.spin_axis is not None:
            result["spin_axis"] = self.spin_axis.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BallState:
        """Create BallState from dictionary."""
        return cls(
            position=Vector3.from_dict(d["position"]),
            velocity=Vector3.from_dict(d["velocity"]),
            spin_rate=d.get("spin_rate", 0.0),
            spin_axis=Vector3.from_dict(d["spin_axis"]) if "spin_axis" in d else None,
            is_in_flight=d.get("is_in_flight", False),
        )


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory.

    Attributes:
        time: Time since trajectory start.
        position: 3D position.
        velocity: Velocity at this point (optional).
        color: RGBA color for rendering (optional).

    Example:
        >>> tp = TrajectoryPoint(time=0.5, position=Vector3(x=10.0, y=0.0, z=5.0))
    """

    time: float
    position: Vector3
    velocity: Vector3 | None = None
    color: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "time": self.time,
            "position": self.position.to_dict(),
        }
        if self.velocity is not None:
            result["velocity"] = self.velocity.to_dict()
        if self.color is not None:
            result["color"] = list(self.color)
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrajectoryPoint:
        """Create TrajectoryPoint from dictionary."""
        color = tuple(d["color"]) if "color" in d else None
        return cls(
            time=float(d["time"]),
            position=Vector3.from_dict(d["position"]),
            velocity=Vector3.from_dict(d["velocity"]) if "velocity" in d else None,
            color=color,  # type: ignore
        )


@dataclass
class EnvironmentState:
    """Environmental conditions for simulation.

    Attributes:
        wind_velocity: Wind velocity vector.
        temperature: Air temperature in Celsius.
        humidity: Relative humidity (0-1).
        altitude: Altitude in meters.
        air_density: Air density in kg/m^3.
        pressure: Atmospheric pressure in hPa.

    Example:
        >>> env = EnvironmentState.default()
        >>> env.temperature
        20.0
    """

    wind_velocity: Vector3 = field(default_factory=Vector3.zero)
    temperature: float = 20.0
    humidity: float = 0.5
    altitude: float = 0.0
    air_density: float = 1.225
    pressure: float = 1013.25

    @classmethod
    def default(cls) -> EnvironmentState:
        """Create default environment (sea level, no wind).

        Returns:
            EnvironmentState with standard conditions.
        """
        return cls(
            wind_velocity=Vector3.zero(),
            temperature=20.0,
            humidity=0.5,
            altitude=0.0,
            air_density=1.225,
            pressure=1013.25,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "wind_velocity": self.wind_velocity.to_dict(),
            "temperature": self.temperature,
            "humidity": self.humidity,
            "altitude": self.altitude,
            "air_density": self.air_density,
            "pressure": self.pressure,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvironmentState:
        """Create EnvironmentState from dictionary."""
        return cls(
            wind_velocity=Vector3.from_dict(d["wind_velocity"]),
            temperature=d.get("temperature", 20.0),
            humidity=d.get("humidity", 0.5),
            altitude=d.get("altitude", 0.0),
            air_density=d.get("air_density", 1.225),
            pressure=d.get("pressure", 1013.25),
        )
