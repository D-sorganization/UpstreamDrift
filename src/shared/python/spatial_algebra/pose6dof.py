"""
6DOF (Six Degrees of Freedom) positioning module.

Provides intuitive classes for positioning and orienting entities in 3D space:
- Pose6DOF: Position + orientation representation
- Transform6DOF: Rigid body transformation
- EntityPlacement: High-level entity positioning for simulation models
- PlacementGroup: Managing multiple entity placements

Follows pragmatic programmer principles:
- DRY: Reuses existing spatial algebra functions
- KISS: Simple, intuitive API
- Clear separation of concerns
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .spatial_vectors import skew
from .transforms import xtrans

if TYPE_CHECKING:
    from collections.abc import Iterator

# Type alias for clarity
Vec3 = npt.NDArray[np.float64]
Mat3 = npt.NDArray[np.float64]
Mat4 = npt.NDArray[np.float64]
Mat6 = npt.NDArray[np.float64]
Quat = npt.NDArray[np.float64]  # [w, x, y, z]


# =============================================================================
# Rotation Conversion Utilities
# =============================================================================


def euler_to_rotation_matrix(
    euler: Vec3 | list[float] | tuple[float, float, float],
) -> Mat3:
    """
    Convert euler angles (roll, pitch, yaw) to 3x3 rotation matrix.

    Uses ZYX convention (yaw-pitch-roll): R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Args:
        euler: [roll, pitch, yaw] in radians

    Returns:
        3x3 rotation matrix
    """
    euler = np.asarray(euler, dtype=np.float64)
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return R


def rotation_matrix_to_euler(R: Mat3) -> Vec3:
    """
    Convert 3x3 rotation matrix to euler angles (roll, pitch, yaw).

    Uses ZYX convention.

    Args:
        R: 3x3 rotation matrix

    Returns:
        [roll, pitch, yaw] in radians
    """
    R = np.asarray(R, dtype=np.float64)

    # Handle gimbal lock
    if np.abs(R[2, 0]) >= 1.0 - 1e-10:
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        cp = np.cos(pitch)
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_quaternion(
    euler: Vec3 | list[float] | tuple[float, float, float],
) -> Quat:
    """
    Convert euler angles (roll, pitch, yaw) to quaternion [w, x, y, z].

    Args:
        euler: [roll, pitch, yaw] in radians

    Returns:
        Quaternion [w, x, y, z]
    """
    euler = np.asarray(euler, dtype=np.float64)
    roll, pitch, yaw = euler[0] / 2, euler[1] / 2, euler[2] / 2

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_to_euler(quat: Quat | list[float]) -> Vec3:
    """
    Convert quaternion [w, x, y, z] to euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        [roll, pitch, yaw] in radians
    """
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def quaternion_to_rotation_matrix(quat: Quat | list[float]) -> Mat3:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)  # Normalize
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def rotation_matrix_to_quaternion(R: Mat3) -> Quat:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z]
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float64)


def axis_angle_to_rotation_matrix(axis: Vec3 | list[float], angle: float) -> Mat3:
    """
    Convert axis-angle representation to 3x3 rotation matrix (Rodrigues formula).

    Args:
        axis: Unit vector representing rotation axis
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)  # Normalize

    K = skew(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def quaternion_multiply(q1: Quat | list[float], q2: Quat | list[float]) -> Quat:
    """
    Multiply two quaternions [w, x, y, z].

    Args:
        q1: First quaternion
        q2: Second quaternion

    Returns:
        Product quaternion
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_inverse(q: Quat | list[float]) -> Quat:
    """
    Compute inverse of a quaternion.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Inverse quaternion
    """
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64) / np.dot(q, q)


def slerp(q1: Quat, q2: Quat, t: float) -> Quat:
    """
    Spherical linear interpolation between quaternions.

    Args:
        q1: Start quaternion
        q2: End quaternion
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    dot = np.dot(q1, q2)

    # Handle antipodal quaternions
    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        # Linear interpolation for very close quaternions
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return s1 * q1 + s2 * q2


# =============================================================================
# Pose6DOF Class
# =============================================================================


@dataclass
class Pose6DOF:
    """
    Represents a 6DOF pose: position (x, y, z) and orientation (roll, pitch, yaw).

    Provides intuitive methods for positioning and orienting objects in 3D space.
    Uses ZYX Euler angle convention (yaw-pitch-roll).
    """

    _position: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    _euler_angles: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __init__(
        self,
        position: Vec3 | list[float] | None = None,
        euler_angles: Vec3 | list[float] | None = None,
    ) -> None:
        """
        Initialize a 6DOF pose.

        Args:
            position: [x, y, z] position, defaults to origin
            euler_angles: [roll, pitch, yaw] in radians, defaults to no rotation
        """
        if position is None:
            self._position = np.zeros(3, dtype=np.float64)
        else:
            self._position = np.asarray(position, dtype=np.float64).copy()

        if euler_angles is None:
            self._euler_angles = np.zeros(3, dtype=np.float64)
        else:
            self._euler_angles = np.asarray(euler_angles, dtype=np.float64).copy()

    @classmethod
    def from_quaternion(
        cls,
        position: Vec3 | list[float],
        quaternion: Quat | list[float],
    ) -> Pose6DOF:
        """Create pose from position and quaternion [w, x, y, z]."""
        euler = quaternion_to_euler(quaternion)
        return cls(position=position, euler_angles=euler)

    @classmethod
    def from_rotation_matrix(
        cls,
        position: Vec3 | list[float],
        rotation: Mat3,
    ) -> Pose6DOF:
        """Create pose from position and rotation matrix."""
        euler = rotation_matrix_to_euler(rotation)
        return cls(position=position, euler_angles=euler)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def position(self) -> Vec3:
        """Get position [x, y, z]."""
        return self._position

    @position.setter
    def position(self, value: Vec3 | list[float]) -> None:
        """Set position [x, y, z]."""
        self._position = np.asarray(value, dtype=np.float64)

    @property
    def euler_angles(self) -> Vec3:
        """Get euler angles [roll, pitch, yaw] in radians."""
        return self._euler_angles

    @euler_angles.setter
    def euler_angles(self, value: Vec3 | list[float]) -> None:
        """Set euler angles [roll, pitch, yaw] in radians."""
        self._euler_angles = np.asarray(value, dtype=np.float64)

    @property
    def x(self) -> float:
        """Get x position."""
        return float(self._position[0])

    @x.setter
    def x(self, value: float) -> None:
        """Set x position."""
        self._position[0] = value

    @property
    def y(self) -> float:
        """Get y position."""
        return float(self._position[1])

    @y.setter
    def y(self, value: float) -> None:
        """Set y position."""
        self._position[1] = value

    @property
    def z(self) -> float:
        """Get z position."""
        return float(self._position[2])

    @z.setter
    def z(self, value: float) -> None:
        """Set z position."""
        self._position[2] = value

    @property
    def roll(self) -> float:
        """Get roll angle in radians."""
        return float(self._euler_angles[0])

    @roll.setter
    def roll(self, value: float) -> None:
        """Set roll angle in radians."""
        self._euler_angles[0] = value

    @property
    def pitch(self) -> float:
        """Get pitch angle in radians."""
        return float(self._euler_angles[1])

    @pitch.setter
    def pitch(self, value: float) -> None:
        """Set pitch angle in radians."""
        self._euler_angles[1] = value

    @property
    def yaw(self) -> float:
        """Get yaw angle in radians."""
        return float(self._euler_angles[2])

    @yaw.setter
    def yaw(self, value: float) -> None:
        """Set yaw angle in radians."""
        self._euler_angles[2] = value

    @property
    def rotation_matrix(self) -> Mat3:
        """Get 3x3 rotation matrix."""
        return euler_to_rotation_matrix(self._euler_angles)

    @property
    def homogeneous_matrix(self) -> Mat4:
        """Get 4x4 homogeneous transformation matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self._position
        return T

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def to_quaternion(self) -> Quat:
        """Convert orientation to quaternion [w, x, y, z]."""
        return euler_to_quaternion(self._euler_angles)

    def to_spatial_transform(self) -> Mat6:
        """Convert to 6x6 Plücker spatial transformation matrix."""
        R = self.rotation_matrix
        return xtrans(R, self._position)

    # -------------------------------------------------------------------------
    # Transformation Methods
    # -------------------------------------------------------------------------

    def translate(self, offset: Vec3 | list[float]) -> Pose6DOF:
        """Return new pose translated by offset (world frame)."""
        offset = np.asarray(offset, dtype=np.float64)
        return Pose6DOF(
            position=self._position + offset,
            euler_angles=self._euler_angles.copy(),
        )

    def rotate_euler(self, delta_euler: Vec3 | list[float]) -> Pose6DOF:
        """Return new pose with additional euler rotation."""
        # Compose rotations via quaternions for accuracy
        q1 = self.to_quaternion()
        q2 = euler_to_quaternion(delta_euler)
        q3 = quaternion_multiply(q1, q2)
        new_euler = quaternion_to_euler(q3)
        return Pose6DOF(position=self._position.copy(), euler_angles=new_euler)

    def inverse(self) -> Pose6DOF:
        """Return inverse pose."""
        R = self.rotation_matrix
        R_inv = R.T
        p_inv = -R_inv @ self._position
        euler_inv = rotation_matrix_to_euler(R_inv)
        return Pose6DOF(position=p_inv, euler_angles=euler_inv)

    def compose(self, other: Pose6DOF) -> Pose6DOF:
        """Compose this pose with another (this * other)."""
        R1 = self.rotation_matrix
        p1 = self._position
        R2 = other.rotation_matrix
        p2 = other._position

        R = R1 @ R2
        p = R1 @ p2 + p1

        return Pose6DOF(position=p, euler_angles=rotation_matrix_to_euler(R))

    def transform_point(self, point: Vec3 | list[float]) -> Vec3:
        """Transform a point by this pose."""
        point = np.asarray(point, dtype=np.float64)
        return self.rotation_matrix @ point + self._position

    def transform_vector(self, vector: Vec3 | list[float]) -> Vec3:
        """Transform a direction vector (rotation only, no translation)."""
        vector = np.asarray(vector, dtype=np.float64)
        return self.rotation_matrix @ vector

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def copy(self) -> Pose6DOF:
        """Create a deep copy of this pose."""
        return Pose6DOF(
            position=self._position.copy(),
            euler_angles=self._euler_angles.copy(),
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with tolerance."""
        if not isinstance(other, Pose6DOF):
            return False
        return bool(
            np.allclose(self._position, other._position, atol=1e-10)
            and np.allclose(self._euler_angles, other._euler_angles, atol=1e-10)
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Pose6DOF(position=[{self.x:.4f}, {self.y:.4f}, {self.z:.4f}], "
            f"euler=[{self.roll:.4f}, {self.pitch:.4f}, {self.yaw:.4f}])"
        )


# =============================================================================
# Transform6DOF Class
# =============================================================================


@dataclass
class Transform6DOF:
    """
    Rigid body transformation in 3D space.

    Stores rotation matrix and translation vector for efficient composition
    and point transformation.
    """

    _rotation: Mat3 = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    _translation: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __init__(
        self,
        rotation: Mat3 | None = None,
        translation: Vec3 | list[float] | None = None,
    ) -> None:
        """
        Initialize transform.

        Args:
            rotation: 3x3 rotation matrix, defaults to identity
            translation: 3D translation vector, defaults to zero
        """
        if rotation is None:
            self._rotation = np.eye(3, dtype=np.float64)
        else:
            self._rotation = np.asarray(rotation, dtype=np.float64).copy()

        if translation is None:
            self._translation = np.zeros(3, dtype=np.float64)
        else:
            self._translation = np.asarray(translation, dtype=np.float64).copy()

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls) -> Transform6DOF:
        """Create identity transform."""
        return cls()

    @classmethod
    def from_translation(cls, translation: Vec3 | list[float]) -> Transform6DOF:
        """Create pure translation transform."""
        return cls(translation=translation)

    @classmethod
    def from_rotation_x(cls, angle: float) -> Transform6DOF:
        """Create rotation about x-axis."""
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
        return cls(rotation=R)

    @classmethod
    def from_rotation_y(cls, angle: float) -> Transform6DOF:
        """Create rotation about y-axis."""
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        return cls(rotation=R)

    @classmethod
    def from_rotation_z(cls, angle: float) -> Transform6DOF:
        """Create rotation about z-axis."""
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        return cls(rotation=R)

    @classmethod
    def from_axis_angle(cls, axis: Vec3 | list[float], angle: float) -> Transform6DOF:
        """Create rotation about arbitrary axis."""
        R = axis_angle_to_rotation_matrix(axis, angle)
        return cls(rotation=R)

    @classmethod
    def from_rotation_matrix(
        cls,
        rotation: Mat3,
        translation: Vec3 | list[float] | None = None,
    ) -> Transform6DOF:
        """Create transform from rotation matrix and optional translation."""
        if translation is None:
            translation = np.zeros(3)
        return cls(rotation=rotation, translation=translation)

    @classmethod
    def from_homogeneous_matrix(cls, H: Mat4) -> Transform6DOF:
        """Create transform from 4x4 homogeneous matrix."""
        H = np.asarray(H, dtype=np.float64)
        return cls(rotation=H[:3, :3], translation=H[:3, 3])

    @classmethod
    def from_pose(cls, pose: Pose6DOF) -> Transform6DOF:
        """Create transform from Pose6DOF."""
        return cls(rotation=pose.rotation_matrix, translation=pose.position)

    @classmethod
    def interpolate(cls, t1: Transform6DOF, t2: Transform6DOF, alpha: float) -> Transform6DOF:
        """
        Linear interpolation between two transforms.

        Uses SLERP for rotation and linear interpolation for translation.

        Args:
            t1: Start transform
            t2: End transform
            alpha: Interpolation parameter [0, 1]

        Returns:
            Interpolated transform
        """
        # Interpolate translation linearly
        translation = (1 - alpha) * t1._translation + alpha * t2._translation

        # Interpolate rotation with SLERP
        q1 = rotation_matrix_to_quaternion(t1._rotation)
        q2 = rotation_matrix_to_quaternion(t2._rotation)
        q = slerp(q1, q2, alpha)
        rotation = quaternion_to_rotation_matrix(q)

        return cls(rotation=rotation, translation=translation)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def rotation_matrix(self) -> Mat3:
        """Get 3x3 rotation matrix."""
        return self._rotation

    @property
    def translation(self) -> Vec3:
        """Get translation vector."""
        return self._translation

    @property
    def homogeneous_matrix(self) -> Mat4:
        """Get 4x4 homogeneous transformation matrix."""
        H = np.eye(4, dtype=np.float64)
        H[:3, :3] = self._rotation
        H[:3, 3] = self._translation
        return H

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def compose(self, other: Transform6DOF) -> Transform6DOF:
        """Compose this transform with another (this * other)."""
        R = self._rotation @ other._rotation
        t = self._rotation @ other._translation + self._translation
        return Transform6DOF(rotation=R, translation=t)

    def inverse(self) -> Transform6DOF:
        """Return inverse transform."""
        R_inv = self._rotation.T
        t_inv = -R_inv @ self._translation
        return Transform6DOF(rotation=R_inv, translation=t_inv)

    def transform_point(self, point: Vec3 | list[float]) -> Vec3:
        """Transform a point."""
        point = np.asarray(point, dtype=np.float64)
        return self._rotation @ point + self._translation

    def transform_points(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform multiple points (Nx3 array)."""
        points = np.asarray(points, dtype=np.float64)
        return (self._rotation @ points.T).T + self._translation

    def transform_vector(self, vector: Vec3 | list[float]) -> Vec3:
        """Transform a direction vector (rotation only)."""
        vector = np.asarray(vector, dtype=np.float64)
        return self._rotation @ vector

    def to_spatial_transform(self) -> Mat6:
        """Convert to 6x6 Plücker spatial transformation matrix."""
        return xtrans(self._rotation, self._translation)

    def to_pose(self) -> Pose6DOF:
        """Convert to Pose6DOF."""
        euler = rotation_matrix_to_euler(self._rotation)
        return Pose6DOF(position=self._translation.copy(), euler_angles=euler)

    def copy(self) -> Transform6DOF:
        """Create a deep copy."""
        return Transform6DOF(
            rotation=self._rotation.copy(),
            translation=self._translation.copy(),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Transform6DOF(translation=[{self._translation[0]:.4f}, "
            f"{self._translation[1]:.4f}, {self._translation[2]:.4f}])"
        )


# =============================================================================
# EntityPlacement Class
# =============================================================================


@dataclass
class EntityPlacement:
    """
    High-level entity/offense placement for simulation models.

    Provides intuitive methods for positioning entities (offenses, objects, models)
    in 3D simulation space with 6DOF control.
    """

    name: str
    pose: Pose6DOF = field(default_factory=Pose6DOF)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        pose: Pose6DOF | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize entity placement.

        Args:
            name: Unique identifier for the entity
            pose: Initial pose, defaults to origin with no rotation
            metadata: Optional metadata dictionary
        """
        self.name = name
        self.pose = pose if pose is not None else Pose6DOF()
        self.metadata = metadata if metadata is not None else {}

    # -------------------------------------------------------------------------
    # Position Methods
    # -------------------------------------------------------------------------

    def move_to(self, x: float, y: float, z: float) -> None:
        """Move entity to absolute position."""
        self.pose.position = np.array([x, y, z], dtype=np.float64)

    def move_by(self, dx: float = 0, dy: float = 0, dz: float = 0) -> None:
        """Move entity by relative offset."""
        self.pose._position += np.array([dx, dy, dz], dtype=np.float64)

    # -------------------------------------------------------------------------
    # Rotation Methods
    # -------------------------------------------------------------------------

    def rotate_euler(self, roll: float = 0, pitch: float = 0, yaw: float = 0) -> None:
        """Set entity rotation using euler angles."""
        self.pose.euler_angles = np.array([roll, pitch, yaw], dtype=np.float64)

    def set_yaw(self, yaw: float) -> None:
        """Set entity heading (yaw) angle."""
        self.pose.yaw = yaw

    def rotate_axis(self, axis: Vec3 | list[float], angle: float) -> None:
        """Rotate entity about arbitrary axis."""
        R = axis_angle_to_rotation_matrix(axis, angle)
        new_euler = rotation_matrix_to_euler(R @ self.pose.rotation_matrix)
        self.pose.euler_angles = new_euler

    def look_at(self, target: Vec3 | list[float], up: Vec3 | list[float] | None = None) -> None:
        """
        Orient entity to look at a target point.

        Args:
            target: Point to look at
            up: Up vector, defaults to [0, 0, 1]
        """
        target = np.asarray(target, dtype=np.float64)
        if up is None:
            up = np.array([0, 0, 1], dtype=np.float64)
        else:
            up = np.asarray(up, dtype=np.float64)

        forward = target - self.pose.position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-10:
            return  # Target is at entity position
        forward = forward / forward_norm

        # Compute rotation matrix from forward vector
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-10:
            # Forward is parallel to up, choose arbitrary right
            right = np.array([0, 1, 0], dtype=np.float64)
        else:
            right = right / right_norm

        up_corrected = np.cross(right, forward)

        # Build rotation matrix with forward as x-axis convention
        R = np.column_stack([forward, right, up_corrected])
        self.pose.euler_angles = rotation_matrix_to_euler(R)

    # -------------------------------------------------------------------------
    # Frame Vectors
    # -------------------------------------------------------------------------

    @property
    def forward_vector(self) -> Vec3:
        """Get entity's forward direction (local +x in world frame)."""
        return self.pose.rotation_matrix @ np.array([1, 0, 0], dtype=np.float64)

    @property
    def right_vector(self) -> Vec3:
        """Get entity's right direction (local +y in world frame)."""
        return self.pose.rotation_matrix @ np.array([0, 1, 0], dtype=np.float64)

    @property
    def up_vector(self) -> Vec3:
        """Get entity's up direction (local +z in world frame)."""
        return self.pose.rotation_matrix @ np.array([0, 0, 1], dtype=np.float64)

    # -------------------------------------------------------------------------
    # Distance Methods
    # -------------------------------------------------------------------------

    def distance_to(self, point: Vec3 | list[float]) -> float:
        """Calculate distance to a point."""
        point = np.asarray(point, dtype=np.float64)
        return float(np.linalg.norm(self.pose.position - point))

    def distance_to_entity(self, other: EntityPlacement) -> float:
        """Calculate distance to another entity."""
        return float(np.linalg.norm(self.pose.position - other.pose.position))

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def to_transform(self) -> Transform6DOF:
        """Convert to Transform6DOF."""
        return Transform6DOF.from_pose(self.pose)

    @classmethod
    def from_transform(
        cls,
        name: str,
        transform: Transform6DOF,
        metadata: dict[str, Any] | None = None,
    ) -> EntityPlacement:
        """Create entity from Transform6DOF."""
        return cls(name=name, pose=transform.to_pose(), metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "position": self.pose.position.tolist(),
            "euler_angles": self.pose.euler_angles.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityPlacement:
        """Deserialize from dictionary."""
        pose = Pose6DOF(
            position=data["position"],
            euler_angles=data["euler_angles"],
        )
        return cls(
            name=data["name"],
            pose=pose,
            metadata=data.get("metadata", {}),
        )

    def copy(self) -> EntityPlacement:
        """Create a deep copy."""
        return EntityPlacement(
            name=self.name,
            pose=self.pose.copy(),
            metadata=deepcopy(self.metadata),
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"EntityPlacement(name='{self.name}', {self.pose})"


# =============================================================================
# PlacementGroup Class
# =============================================================================


class PlacementGroup:
    """
    Manages a collection of entity placements.

    Provides operations on groups of entities like bulk translation,
    rotation around a point, and spatial queries.
    """

    def __init__(self) -> None:
        """Initialize empty placement group."""
        self._entities: dict[str, EntityPlacement] = {}

    def add(self, entity: EntityPlacement) -> None:
        """Add an entity to the group."""
        self._entities[entity.name] = entity

    def remove(self, name: str) -> None:
        """Remove an entity by name."""
        if name in self._entities:
            del self._entities[name]

    def get(self, name: str) -> EntityPlacement | None:
        """Get entity by name."""
        return self._entities.get(name)

    def __len__(self) -> int:
        """Return number of entities."""
        return len(self._entities)

    def __iter__(self) -> Iterator[EntityPlacement]:
        """Iterate over entities."""
        return iter(self._entities.values())

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    def translate_all(self, offset: Vec3 | list[float]) -> None:
        """Translate all entities by offset."""
        offset = np.asarray(offset, dtype=np.float64)
        for entity in self._entities.values():
            entity.pose._position += offset

    def rotate_around_point(
        self,
        point: Vec3 | list[float],
        axis: Vec3 | list[float],
        angle: float,
    ) -> None:
        """
        Rotate all entities around a point.

        Args:
            point: Center of rotation
            axis: Rotation axis
            angle: Rotation angle in radians
        """
        point = np.asarray(point, dtype=np.float64)
        R = axis_angle_to_rotation_matrix(axis, angle)

        for entity in self._entities.values():
            # Translate to origin, rotate, translate back
            rel_pos = entity.pose.position - point
            new_pos = R @ rel_pos + point
            entity.pose.position = new_pos

            # Also rotate entity's orientation
            new_euler = rotation_matrix_to_euler(R @ entity.pose.rotation_matrix)
            entity.pose.euler_angles = new_euler

    # -------------------------------------------------------------------------
    # Spatial Queries
    # -------------------------------------------------------------------------

    @property
    def centroid(self) -> Vec3:
        """Calculate centroid of all entity positions."""
        if not self._entities:
            return np.zeros(3, dtype=np.float64)
        positions = np.array([e.pose.position for e in self._entities.values()])
        return np.mean(positions, axis=0)

    @property
    def bounding_box(self) -> dict[str, Vec3]:
        """Calculate axis-aligned bounding box."""
        if not self._entities:
            return {"min": np.zeros(3), "max": np.zeros(3)}
        positions = np.array([e.pose.position for e in self._entities.values()])
        return {
            "min": np.min(positions, axis=0),
            "max": np.max(positions, axis=0),
        }

    def copy(self) -> PlacementGroup:
        """Create a deep copy of the group."""
        new_group = PlacementGroup()
        for entity in self._entities.values():
            new_group.add(entity.copy())
        return new_group

    def __repr__(self) -> str:
        """String representation."""
        return f"PlacementGroup({len(self)} entities)"
