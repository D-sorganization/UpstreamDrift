"""Skeleton and force data models for Unreal Engine integration.

Provides JointState and ForceVector data classes for representing
articulated body states and force/torque visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .geometry import Quaternion, Vector3


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
