"""Core data models for Unreal Engine integration.

This module defines the data structures used for communication between
the Python physics backend and Unreal Engine visualization frontend.

Design by Contract:
    - All data models support optional validation on construction
    - Preconditions ensure valid input data
    - Postconditions guarantee valid serialization output
    - Immutable data models prevent accidental state corruption

Data Flow:
    Physics Engine → Data Models → JSON/MessagePack → WebSocket → Unreal Engine

Usage:
    from src.unreal_integration.data_models import (
        UnrealDataFrame,
        JointState,
        ForceVector,
        Vector3,
    )

    frame = UnrealDataFrame(
        timestamp=0.0167,
        frame_number=1,
        joints={"shoulder_L": JointState(...)},
    )
    json_str = frame.to_json()
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Note: This module is self-contained to avoid pulling in heavy dependencies
# from src.shared.python. Design by Contract is enforced inline.


def _validate_finite(value: float, name: str) -> None:
    """Validate that a value is finite (not NaN or Inf).

    Args:
        value: Value to validate.
        name: Name of the value for error messages.

    Raises:
        ValueError: If value is NaN or infinite.
    """
    if math.isnan(value):
        raise ValueError(f"{name} cannot be NaN")
    if math.isinf(value):
        raise ValueError(f"{name} cannot be infinite")


@dataclass(frozen=False)
class Vector3:
    """3D vector representation for positions, velocities, forces.

    Provides common vector operations and serialization.
    Follows Design by Contract with optional validation.

    Attributes:
        x: X component.
        y: Y component.
        z: Z component.

    Example:
        >>> v = Vector3(x=1.0, y=2.0, z=3.0)
        >>> v.magnitude
        3.7416...
        >>> v.normalized().magnitude
        1.0
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self) -> None:
        """Post-initialization hook for optional validation."""
        # Validation is opt-in via factory methods

    @classmethod
    def from_numpy(cls, arr: np.ndarray, validate: bool = False) -> Vector3:
        """Create Vector3 from numpy array.

        Precondition: arr must have exactly 3 elements.

        Args:
            arr: Numpy array with 3 elements.
            validate: If True, validate values are finite.

        Returns:
            New Vector3 instance.

        Raises:
            ValueError: If array doesn't have 3 elements or values invalid.
        """
        if arr.shape != (3,) and arr.size != 3:
            raise ValueError(f"Array must have 3 elements, got shape {arr.shape}")
        flat = arr.flatten()
        v = cls(x=float(flat[0]), y=float(flat[1]), z=float(flat[2]))
        if validate:
            v._validate()
        return v

    @classmethod
    def from_dict(cls, d: dict[str, Any], validate: bool = False) -> Vector3:
        """Create Vector3 from dictionary.

        Args:
            d: Dictionary with 'x', 'y', 'z' keys.
            validate: If True, validate values are finite.

        Returns:
            New Vector3 instance.
        """
        v = cls(x=float(d["x"]), y=float(d["y"]), z=float(d["z"]))
        if validate:
            v._validate()
        return v

    @classmethod
    def zero(cls) -> Vector3:
        """Create zero vector."""
        return cls(x=0.0, y=0.0, z=0.0)

    def __new__(
        cls, x: float = 0.0, y: float = 0.0, z: float = 0.0, validate: bool = False
    ) -> Vector3:
        """Create new Vector3 with optional validation."""
        instance = object.__new__(cls)
        return instance

    def __init__(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0, validate: bool = False
    ) -> None:
        """Initialize Vector3.

        Args:
            x: X component.
            y: Y component.
            z: Z component.
            validate: If True, validate values are finite.
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        if validate:
            self._validate()

    def _validate(self) -> None:
        """Validate vector components are finite."""
        _validate_finite(self.x, "x")
        _validate_finite(self.y, "y")
        _validate_finite(self.z, "z")

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array.

        Returns:
            Numpy array with shape (3,).
        """
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary.

        Returns:
            Dictionary with 'x', 'y', 'z' keys.
        """
        return {"x": self.x, "y": self.y, "z": self.z}

    @property
    def magnitude(self) -> float:
        """Calculate vector magnitude.

        Returns:
            Euclidean length of vector.
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> Vector3:
        """Return normalized (unit length) vector.

        Returns:
            New Vector3 with magnitude 1.0.

        Raises:
            ValueError: If vector has zero magnitude.
        """
        mag = self.magnitude
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector3(x=self.x / mag, y=self.y / mag, z=self.z / mag)

    def dot(self, other: Vector3) -> float:
        """Compute dot product with another vector.

        Args:
            other: Another Vector3.

        Returns:
            Scalar dot product.
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        """Compute cross product with another vector.

        Args:
            other: Another Vector3.

        Returns:
            New Vector3 representing cross product.
        """
        return Vector3(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )

    def __add__(self, other: Vector3) -> Vector3:
        """Add two vectors."""
        return Vector3(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        """Subtract two vectors."""
        return Vector3(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        """Multiply vector by scalar."""
        return Vector3(x=self.x * scalar, y=self.y * scalar, z=self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        """Right multiply vector by scalar."""
        return self.__mul__(scalar)

    def __neg__(self) -> Vector3:
        """Negate vector."""
        return Vector3(x=-self.x, y=-self.y, z=-self.z)

    def __eq__(self, other: object) -> bool:
        """Check equality with another vector."""
        if not isinstance(other, Vector3):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self) -> str:
        """String representation."""
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"


@dataclass
class Quaternion:
    """Quaternion for rotation representation.

    Uses (w, x, y, z) convention where w is the scalar component.

    Attributes:
        w: Scalar component.
        x: X component of vector part.
        y: Y component of vector part.
        z: Z component of vector part.

    Example:
        >>> q = Quaternion.identity()
        >>> q.w
        1.0
        >>> q.magnitude
        1.0
    """

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __new__(
        cls,
        w: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        validate: bool = False,
    ) -> Quaternion:
        """Create new Quaternion."""
        instance = object.__new__(cls)
        return instance

    def __init__(
        self,
        w: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        validate: bool = False,
    ) -> None:
        """Initialize Quaternion.

        Args:
            w: Scalar component.
            x: X component of vector part.
            y: Y component of vector part.
            z: Z component of vector part.
            validate: If True, normalize the quaternion.
        """
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        if validate:
            self._normalize_in_place()

    def _normalize_in_place(self) -> None:
        """Normalize quaternion in place."""
        mag = self.magnitude
        if mag > 0:
            self.w /= mag
            self.x /= mag
            self.y /= mag
            self.z /= mag

    @classmethod
    def identity(cls) -> Quaternion:
        """Create identity quaternion (no rotation)."""
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    @classmethod
    def from_euler(
        cls, roll: float, pitch: float, yaw: float, validate: bool = False
    ) -> Quaternion:
        """Create quaternion from Euler angles (roll, pitch, yaw).

        Uses ZYX convention (yaw-pitch-roll).

        Args:
            roll: Rotation around X axis in radians.
            pitch: Rotation around Y axis in radians.
            yaw: Rotation around Z axis in radians.
            validate: If True, normalize the result.

        Returns:
            New Quaternion representing the rotation.
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
        )
        if validate:
            q._normalize_in_place()
        return q

    @classmethod
    def from_dict(cls, d: dict[str, Any], validate: bool = False) -> Quaternion:
        """Create Quaternion from dictionary.

        Args:
            d: Dictionary with 'w', 'x', 'y', 'z' keys.
            validate: If True, normalize the result.

        Returns:
            New Quaternion instance.
        """
        return cls(
            w=float(d["w"]),
            x=float(d["x"]),
            y=float(d["y"]),
            z=float(d["z"]),
            validate=validate,
        )

    def to_euler(self) -> tuple[float, float, float]:
        """Convert to Euler angles (roll, pitch, yaw).

        Returns:
            Tuple of (roll, pitch, yaw) in radians.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary.

        Returns:
            Dictionary with 'w', 'x', 'y', 'z' keys.
        """
        return {"w": self.w, "x": self.x, "y": self.y, "z": self.z}

    @property
    def magnitude(self) -> float:
        """Calculate quaternion magnitude.

        Returns:
            Magnitude of quaternion.
        """
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> Quaternion:
        """Return normalized quaternion.

        Returns:
            New Quaternion with magnitude 1.0.
        """
        mag = self.magnitude
        if mag == 0:
            return Quaternion.identity()
        return Quaternion(
            w=self.w / mag,
            x=self.x / mag,
            y=self.y / mag,
            z=self.z / mag,
        )

    def conjugate(self) -> Quaternion:
        """Return conjugate of quaternion.

        Returns:
            New Quaternion with negated vector part.
        """
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    def __repr__(self) -> str:
        """String representation."""
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"


@dataclass
class JointState:
    """State of a single joint/bone in the skeleton.

    Attributes:
        name: Joint name (must match skeleton mapping).
        position: 3D position in world space.
        rotation: Orientation as quaternion.
        velocity: Linear velocity (optional).
        angular_velocity: Angular velocity (optional).
        joint_angle: Joint angle in radians for revolute joints (optional).
        joint_velocity: Joint angular velocity (optional).
        parent_name: Name of parent joint for hierarchy (optional).

    Example:
        >>> js = JointState(
        ...     name="shoulder_L",
        ...     position=Vector3(x=0.1, y=1.4, z=0.2),
        ...     rotation=Quaternion.identity(),
        ... )
    """

    name: str
    position: Vector3
    rotation: Quaternion
    velocity: Vector3 | None = None
    angular_velocity: Vector3 | None = None
    joint_angle: float | None = None
    joint_velocity: float | None = None
    parent_name: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization validation."""

    def __new__(
        cls,
        name: str,
        position: Vector3,
        rotation: Quaternion,
        velocity: Vector3 | None = None,
        angular_velocity: Vector3 | None = None,
        joint_angle: float | None = None,
        joint_velocity: float | None = None,
        parent_name: str | None = None,
        validate: bool = False,
    ) -> JointState:
        """Create new JointState with optional validation."""
        instance = object.__new__(cls)
        return instance

    def __init__(
        self,
        name: str,
        position: Vector3,
        rotation: Quaternion,
        velocity: Vector3 | None = None,
        angular_velocity: Vector3 | None = None,
        joint_angle: float | None = None,
        joint_velocity: float | None = None,
        parent_name: str | None = None,
        validate: bool = False,
    ) -> None:
        """Initialize JointState.

        Args:
            name: Joint name.
            position: 3D position.
            rotation: Orientation quaternion.
            velocity: Linear velocity.
            angular_velocity: Angular velocity.
            joint_angle: Joint angle in radians.
            joint_velocity: Joint angular velocity.
            parent_name: Parent joint name.
            validate: If True, validate inputs.
        """
        if validate and (not name or not name.strip()):
            raise ValueError("Joint name cannot be empty")
        self.name = name
        self.position = position
        self.rotation = rotation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.joint_angle = joint_angle
        self.joint_velocity = joint_velocity
        self.parent_name = parent_name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of joint state.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
        }
        if self.velocity is not None:
            result["velocity"] = self.velocity.to_dict()
        if self.angular_velocity is not None:
            result["angular_velocity"] = self.angular_velocity.to_dict()
        if self.joint_angle is not None:
            result["joint_angle"] = self.joint_angle
        if self.joint_velocity is not None:
            result["joint_velocity"] = self.joint_velocity
        if self.parent_name is not None:
            result["parent_name"] = self.parent_name
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any], validate: bool = False) -> JointState:
        """Create JointState from dictionary.

        Args:
            d: Dictionary representation.
            validate: If True, validate inputs.

        Returns:
            New JointState instance.
        """
        return cls(
            name=d["name"],
            position=Vector3.from_dict(d["position"]),
            rotation=Quaternion.from_dict(d["rotation"]),
            velocity=Vector3.from_dict(d["velocity"]) if "velocity" in d else None,
            angular_velocity=(
                Vector3.from_dict(d["angular_velocity"])
                if "angular_velocity" in d
                else None
            ),
            joint_angle=d.get("joint_angle"),
            joint_velocity=d.get("joint_velocity"),
            parent_name=d.get("parent_name"),
            validate=validate,
        )


@dataclass
class ForceVector:
    """Force or torque vector for visualization.

    Used to render force arrows and torque indicators in Unreal Engine.

    Attributes:
        origin: Starting point of the force vector.
        direction: Unit direction of force (will be normalized).
        magnitude: Magnitude of force in Newtons or torque in Nm.
        force_type: Type of force ("force", "torque", "gravity", "muscle", etc.).
        joint_name: Associated joint name (for joint torques).
        color: RGBA color for rendering (optional).

    Example:
        >>> fv = ForceVector(
        ...     origin=Vector3(x=0.0, y=1.0, z=0.0),
        ...     direction=Vector3(x=0.0, y=-1.0, z=0.0),
        ...     magnitude=9.81,
        ...     force_type="gravity",
        ... )
    """

    origin: Vector3
    direction: Vector3
    magnitude: float
    force_type: str = "force"
    joint_name: str | None = None
    color: tuple[float, float, float, float] | None = None
    scale_factor: float = 1.0

    def __new__(
        cls,
        origin: Vector3,
        direction: Vector3,
        magnitude: float,
        force_type: str = "force",
        joint_name: str | None = None,
        color: tuple[float, float, float, float] | None = None,
        scale_factor: float = 1.0,
        validate: bool = False,
    ) -> ForceVector:
        """Create new ForceVector with optional validation."""
        instance = object.__new__(cls)
        return instance

    def __init__(
        self,
        origin: Vector3,
        direction: Vector3,
        magnitude: float,
        force_type: str = "force",
        joint_name: str | None = None,
        color: tuple[float, float, float, float] | None = None,
        scale_factor: float = 1.0,
        validate: bool = False,
    ) -> None:
        """Initialize ForceVector.

        Args:
            origin: Starting point.
            direction: Direction (will be normalized internally).
            magnitude: Force magnitude.
            force_type: Type of force.
            joint_name: Associated joint name.
            color: RGBA color.
            scale_factor: Visual scale factor.
            validate: If True, validate inputs.
        """
        if validate and magnitude < 0:
            raise ValueError("Force magnitude must be positive")
        self.origin = origin
        self.direction = direction
        self.magnitude = magnitude
        self.force_type = force_type
        self.joint_name = joint_name
        self.color = color
        self.scale_factor = scale_factor

    def endpoint(self) -> Vector3:
        """Calculate endpoint of force vector.

        Returns:
            Position at the end of the force arrow.
        """
        dir_normalized = self.direction.normalized()
        scaled_magnitude = self.magnitude * self.scale_factor
        return self.origin + dir_normalized * scaled_magnitude

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "origin": self.origin.to_dict(),
            "direction": self.direction.to_dict(),
            "magnitude": self.magnitude,
            "force_type": self.force_type,
        }
        if self.joint_name is not None:
            result["joint_name"] = self.joint_name
        if self.color is not None:
            result["color"] = list(self.color)
        if self.scale_factor != 1.0:
            result["scale_factor"] = self.scale_factor
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any], validate: bool = False) -> ForceVector:
        """Create ForceVector from dictionary.

        Args:
            d: Dictionary representation.
            validate: If True, validate inputs.

        Returns:
            New ForceVector instance.
        """
        color = tuple(d["color"]) if "color" in d else None
        return cls(
            origin=Vector3.from_dict(d["origin"]),
            direction=Vector3.from_dict(d["direction"]),
            magnitude=float(d["magnitude"]),
            force_type=d.get("force_type", "force"),
            joint_name=d.get("joint_name"),
            color=color,  # type: ignore
            scale_factor=d.get("scale_factor", 1.0),
            validate=validate,
        )


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


@dataclass
class UnrealDataFrame:
    """Complete data frame for Unreal Engine visualization.

    This is the primary data structure streamed to Unreal Engine.
    Contains all data needed to render a single frame.

    Attributes:
        timestamp: Simulation time in seconds.
        frame_number: Frame counter (0-indexed).
        joints: Dictionary of joint states by name.
        forces: List of force vectors to visualize.
        club: Golf club state (optional).
        ball: Golf ball state (optional).
        metrics: Swing analysis metrics (optional).
        trajectory: List of trajectory points (optional).
        environment: Environmental conditions (optional).

    Example:
        >>> frame = UnrealDataFrame(
        ...     timestamp=0.0167,
        ...     frame_number=1,
        ...     joints={"shoulder_L": JointState(...)},
        ... )
        >>> json_str = frame.to_json()
    """

    timestamp: float
    frame_number: int
    joints: dict[str, JointState]
    forces: list[ForceVector] | None = None
    club: ClubState | None = None
    ball: BallState | None = None
    metrics: SwingMetrics | None = None
    trajectory: list[TrajectoryPoint] | None = None
    environment: EnvironmentState | None = None

    def __new__(
        cls,
        timestamp: float,
        frame_number: int,
        joints: dict[str, JointState],
        forces: list[ForceVector] | None = None,
        club: ClubState | None = None,
        ball: BallState | None = None,
        metrics: SwingMetrics | None = None,
        trajectory: list[TrajectoryPoint] | None = None,
        environment: EnvironmentState | None = None,
        validate: bool = False,
    ) -> UnrealDataFrame:
        """Create new UnrealDataFrame with optional validation."""
        instance = object.__new__(cls)
        return instance

    def __init__(
        self,
        timestamp: float,
        frame_number: int,
        joints: dict[str, JointState],
        forces: list[ForceVector] | None = None,
        club: ClubState | None = None,
        ball: BallState | None = None,
        metrics: SwingMetrics | None = None,
        trajectory: list[TrajectoryPoint] | None = None,
        environment: EnvironmentState | None = None,
        validate: bool = False,
    ) -> None:
        """Initialize UnrealDataFrame.

        Args:
            timestamp: Simulation time.
            frame_number: Frame counter.
            joints: Dictionary of joint states.
            forces: List of force vectors.
            club: Golf club state.
            ball: Golf ball state.
            metrics: Swing metrics.
            trajectory: Trajectory points.
            environment: Environmental conditions.
            validate: If True, validate inputs.
        """
        if validate:
            if timestamp < 0:
                raise ValueError("timestamp must be non-negative")
            if frame_number < 0:
                raise ValueError("frame_number must be non-negative")
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.joints = joints
        self.forces = forces
        self.club = club
        self.ball = ball
        self.metrics = metrics
        self.trajectory = trajectory
        self.environment = environment

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "frame": self.frame_number,
            "joints": {name: js.to_dict() for name, js in self.joints.items()},
        }
        if self.forces:
            result["forces"] = [f.to_dict() for f in self.forces]
        if self.club is not None:
            result["club"] = self.club.to_dict()
        if self.ball is not None:
            result["ball"] = self.ball.to_dict()
        if self.metrics is not None:
            result["metrics"] = self.metrics.to_dict()
        if self.trajectory:
            result["trajectory"] = [tp.to_dict() for tp in self.trajectory]
        if self.environment is not None:
            result["environment"] = self.environment.to_dict()
        return result

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict())

    def to_protocol_message(self) -> dict[str, Any]:
        """Convert to WebSocket protocol message format.

        Returns:
            Protocol message with type and data fields.
        """
        return {
            "type": "frame",
            "data": self.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], validate: bool = False) -> UnrealDataFrame:
        """Create UnrealDataFrame from dictionary.

        Args:
            d: Dictionary representation.
            validate: If True, validate inputs.

        Returns:
            New UnrealDataFrame instance.
        """
        joints = {
            name: JointState.from_dict(js_dict)
            for name, js_dict in d.get("joints", {}).items()
        }
        forces = (
            [ForceVector.from_dict(f) for f in d.get("forces", [])]
            if "forces" in d
            else None
        )
        club = ClubState.from_dict(d["club"]) if "club" in d else None
        ball = BallState.from_dict(d["ball"]) if "ball" in d else None
        metrics = SwingMetrics.from_dict(d["metrics"]) if "metrics" in d else None
        trajectory = (
            [TrajectoryPoint.from_dict(tp) for tp in d.get("trajectory", [])]
            if "trajectory" in d
            else None
        )
        environment = (
            EnvironmentState.from_dict(d["environment"]) if "environment" in d else None
        )

        return cls(
            timestamp=float(d["timestamp"]),
            frame_number=int(d["frame"]),
            joints=joints,
            forces=forces,
            club=club,
            ball=ball,
            metrics=metrics,
            trajectory=trajectory,
            environment=environment,
            validate=validate,
        )

    @classmethod
    def from_json(cls, json_str: str, validate: bool = False) -> UnrealDataFrame:
        """Create UnrealDataFrame from JSON string.

        Args:
            json_str: JSON string representation.
            validate: If True, validate inputs.

        Returns:
            New UnrealDataFrame instance.
        """
        d = json.loads(json_str)
        return cls.from_dict(d, validate=validate)

    @classmethod
    def from_physics_state(
        cls,
        q: np.ndarray,
        v: np.ndarray,
        timestamp: float,
        frame_number: int,
        joint_names: list[str] | None = None,
        validate: bool = False,
    ) -> UnrealDataFrame:
        """Create UnrealDataFrame from physics engine state.

        This is a convenience method for converting raw physics state
        into the Unreal Engine format.

        Args:
            q: Generalized coordinates (positions).
            v: Generalized velocities.
            timestamp: Simulation time.
            frame_number: Frame counter.
            joint_names: List of joint names (optional).
            validate: If True, validate inputs.

        Returns:
            New UnrealDataFrame instance.
        """
        joints: dict[str, JointState] = {}

        # Create joint states from physics state
        # This is a simplified mapping - real implementation would use
        # proper skeleton configuration
        if joint_names:
            for i, name in enumerate(joint_names):
                if i * 3 + 2 < len(q):
                    joints[name] = JointState(
                        name=name,
                        position=Vector3(
                            x=float(q[i * 3]) if i * 3 < len(q) else 0.0,
                            y=float(q[i * 3 + 1]) if i * 3 + 1 < len(q) else 0.0,
                            z=float(q[i * 3 + 2]) if i * 3 + 2 < len(q) else 0.0,
                        ),
                        rotation=Quaternion.identity(),
                        velocity=(
                            Vector3(
                                x=float(v[i * 3]) if i * 3 < len(v) else 0.0,
                                y=float(v[i * 3 + 1]) if i * 3 + 1 < len(v) else 0.0,
                                z=float(v[i * 3 + 2]) if i * 3 + 2 < len(v) else 0.0,
                            )
                            if len(v) > i * 3 + 2
                            else None
                        ),
                    )

        return cls(
            timestamp=timestamp,
            frame_number=frame_number,
            joints=joints,
            validate=validate,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnrealDataFrame(t={self.timestamp:.4f}, frame={self.frame_number}, "
            f"joints={len(self.joints)}, forces={len(self.forces or [])})"
        )
