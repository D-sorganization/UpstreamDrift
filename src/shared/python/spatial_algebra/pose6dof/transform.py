from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from ..transforms import xtrans
from .pose import Pose6DOF
from .rotations import (
    axis_angle_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
    slerp,
)

Vec3: TypeAlias = npt.NDArray[np.float64]
Mat3: TypeAlias = npt.NDArray[np.float64]
Mat4: TypeAlias = npt.NDArray[np.float64]
Mat6: TypeAlias = npt.NDArray[np.float64]
Quat: TypeAlias = npt.NDArray[np.float64]


@dataclass
class Transform6DOF:
    """
    Rigid body transformation in 3D space.
    """

    _rotation: Mat3 = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    _translation: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __init__(
        self,
        rotation: Mat3 | None = None,
        translation: Vec3 | list[float] | None = None,
    ) -> None:
        if rotation is None:
            self._rotation = np.eye(3, dtype=np.float64)
        else:
            self._rotation = np.asarray(rotation, dtype=np.float64).copy()

        if translation is None:
            self._translation = np.zeros(3, dtype=np.float64)
        else:
            self._translation = np.asarray(translation, dtype=np.float64).copy()

    @classmethod
    def identity(cls) -> "Transform6DOF":
        return cls()

    @classmethod
    def from_translation(cls, translation: Vec3 | list[float]) -> "Transform6DOF":
        return cls(translation=translation)

    @classmethod
    def from_rotation_x(cls, angle: float) -> "Transform6DOF":
        if angle is None:
            raise ValueError("angle must be provided")
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
        return cls(rotation=R)

    @classmethod
    def from_rotation_y(cls, angle: float) -> "Transform6DOF":
        if angle is None:
            raise ValueError("angle must be provided")
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        return cls(rotation=R)

    @classmethod
    def from_rotation_z(cls, angle: float) -> "Transform6DOF":
        if angle is None:
            raise ValueError("angle must be provided")
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        return cls(rotation=R)

    @classmethod
    def from_axis_angle(cls, axis: Vec3 | list[float], angle: float) -> "Transform6DOF":
        if axis is None:
            raise ValueError("axis must be provided")
        R = axis_angle_to_rotation_matrix(axis, angle)
        return cls(rotation=R)

    @classmethod
    def from_rotation_matrix(
        cls,
        rotation: Mat3,
        translation: Vec3 | list[float] | None = None,
    ) -> "Transform6DOF":
        if rotation is None:
            raise ValueError("rotation must be provided")
        if translation is None:
            translation = np.zeros(3)
        return cls(rotation=rotation, translation=translation)

    @classmethod
    def from_homogeneous_matrix(cls, H: Mat4) -> "Transform6DOF":
        if H is None:
            raise ValueError("H must be provided")
        H = np.asarray(H, dtype=np.float64)
        return cls(rotation=H[:3, :3], translation=H[:3, 3])

    @classmethod
    def from_pose(cls, pose: Pose6DOF) -> "Transform6DOF":
        return cls(rotation=pose.rotation_matrix, translation=pose.position)

    @classmethod
    def interpolate(
        cls, t1: "Transform6DOF", t2: "Transform6DOF", alpha: float
    ) -> "Transform6DOF":
        if t1 is None:
            raise ValueError("t1 must be provided")
        translation = (1 - alpha) * t1._translation + alpha * t2._translation

        q1 = rotation_matrix_to_quaternion(t1._rotation)
        q2 = rotation_matrix_to_quaternion(t2._rotation)
        q = slerp(q1, q2, alpha)
        rotation = quaternion_to_rotation_matrix(q)

        return cls(rotation=rotation, translation=translation)

    @property
    def rotation_matrix(self) -> Mat3:
        return self._rotation

    @property
    def translation(self) -> Vec3:
        return self._translation

    @property
    def homogeneous_matrix(self) -> Mat4:
        H = np.eye(4, dtype=np.float64)
        H[:3, :3] = self._rotation
        H[:3, 3] = self._translation
        return H

    def compose(self, other: "Transform6DOF") -> "Transform6DOF":
        if other is None:
            raise ValueError("other must be provided")
        R = other._rotation @ self._rotation
        t = other._rotation @ self._translation + other._translation
        return Transform6DOF(rotation=R, translation=t)

    def inverse(self) -> "Transform6DOF":
        R_inv = self._rotation.T
        t_inv = -R_inv @ self._translation
        return Transform6DOF(rotation=R_inv, translation=t_inv)

    def transform_point(self, point: Vec3 | list[float]) -> Vec3:
        if point is None:
            raise ValueError("point must be provided")
        point = np.asarray(point, dtype=np.float64)
        return self._rotation @ point + self._translation

    def transform_points(
        self, points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if points is None:
            raise ValueError("points must be provided")
        points = np.asarray(points, dtype=np.float64)
        return (self._rotation @ points.T).T + self._translation

    def transform_vector(self, vector: Vec3 | list[float]) -> Vec3:
        if vector is None:
            raise ValueError("vector must be provided")
        vector = np.asarray(vector, dtype=np.float64)
        return self._rotation @ vector

    def to_spatial_transform(self) -> Mat6:
        return xtrans(self._rotation, self._translation)

    def to_pose(self) -> Pose6DOF:
        euler = rotation_matrix_to_euler(self._rotation)
        return Pose6DOF(position=self._translation.copy(), euler_angles=euler)

    def copy(self) -> "Transform6DOF":
        return Transform6DOF(
            rotation=self._rotation.copy(),
            translation=self._translation.copy(),
        )

    def __repr__(self) -> str:
        return (
            f"Transform6DOF(translation=[{self._translation[0]:.4f}, "
            f"{self._translation[1]:.4f}, {self._translation[2]:.4f}])"
        )
