"""Fundamental geometry types for Unreal Engine integration.

Provides Vector3 and Quaternion data classes used throughout the
unreal_integration package for positions, rotations, velocities, and forces.

Design by Contract:
    - All data models support optional validation on construction
    - Preconditions ensure valid input data
    - Postconditions guarantee valid serialization output
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


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
