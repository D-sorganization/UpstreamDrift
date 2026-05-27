from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from ..transforms import xtrans
from .rotations import (
    euler_to_quaternion,
    euler_to_rotation_matrix,
    quaternion_multiply,
    quaternion_to_euler,
    rotation_matrix_to_euler,
)

Vec3: TypeAlias = npt.NDArray[np.float64]
Mat3: TypeAlias = npt.NDArray[np.float64]
Mat4: TypeAlias = npt.NDArray[np.float64]
Mat6: TypeAlias = npt.NDArray[np.float64]
Quat: TypeAlias = npt.NDArray[np.float64]


@dataclass
class Pose6DOF:
    """
    Represents a 6DOF pose: position (x, y, z) and orientation (roll, pitch, yaw).
    """

    _position: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    _euler_angles: Vec3 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __init__(
        self,
        position: Vec3 | list[float] | None = None,
        euler_angles: Vec3 | list[float] | None = None,
    ) -> None:
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
    ) -> "Pose6DOF":
        if position is None:
            raise ValueError("position must be provided")
        euler = quaternion_to_euler(quaternion)
        return cls(position=position, euler_angles=euler)

    @classmethod
    def from_rotation_matrix(
        cls,
        position: Vec3 | list[float],
        rotation: Mat3,
    ) -> "Pose6DOF":
        if position is None:
            raise ValueError("position must be provided")
        euler = rotation_matrix_to_euler(rotation)
        return cls(position=position, euler_angles=euler)

    @property
    def position(self) -> Vec3:
        return self._position

    @position.setter
    def position(self, value: Vec3 | list[float]) -> None:
        self._position = np.asarray(value, dtype=np.float64)

    @property
    def euler_angles(self) -> Vec3:
        return self._euler_angles

    @euler_angles.setter
    def euler_angles(self, value: Vec3 | list[float]) -> None:
        self._euler_angles = np.asarray(value, dtype=np.float64)

    @property
    def x(self) -> float:
        return float(self._position[0])

    @x.setter
    def x(self, value: float) -> None:
        self._position[0] = value

    @property
    def y(self) -> float:
        return float(self._position[1])

    @y.setter
    def y(self, value: float) -> None:
        self._position[1] = value

    @property
    def z(self) -> float:
        return float(self._position[2])

    @z.setter
    def z(self, value: float) -> None:
        self._position[2] = value

    @property
    def roll(self) -> float:
        return float(self._euler_angles[0])

    @roll.setter
    def roll(self, value: float) -> None:
        self._euler_angles[0] = value

    @property
    def pitch(self) -> float:
        return float(self._euler_angles[1])

    @pitch.setter
    def pitch(self, value: float) -> None:
        self._euler_angles[1] = value

    @property
    def yaw(self) -> float:
        return float(self._euler_angles[2])

    @yaw.setter
    def yaw(self, value: float) -> None:
        self._euler_angles[2] = value

    @property
    def rotation_matrix(self) -> Mat3:
        return euler_to_rotation_matrix(self._euler_angles)

    @property
    def homogeneous_matrix(self) -> Mat4:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self._position
        return T

    def to_quaternion(self) -> Quat:
        return euler_to_quaternion(self._euler_angles)

    def to_spatial_transform(self) -> Mat6:
        R = self.rotation_matrix
        return xtrans(R, self._position)

    def translate(self, offset: Vec3 | list[float]) -> "Pose6DOF":
        if offset is None:
            raise ValueError("offset must be provided")
        offset = np.asarray(offset, dtype=np.float64)
        return Pose6DOF(
            position=self._position + offset,
            euler_angles=self._euler_angles.copy(),
        )

    def rotate_euler(self, delta_euler: Vec3 | list[float]) -> "Pose6DOF":
        if delta_euler is None:
            raise ValueError("delta_euler must be provided")
        q1 = self.to_quaternion()
        q2 = euler_to_quaternion(delta_euler)
        q3 = quaternion_multiply(q1, q2)
        new_euler = quaternion_to_euler(q3)
        return Pose6DOF(position=self._position.copy(), euler_angles=new_euler)

    def inverse(self) -> "Pose6DOF":
        R = self.rotation_matrix
        R_inv = R.T
        p_inv = -R_inv @ self._position
        euler_inv = rotation_matrix_to_euler(R_inv)
        return Pose6DOF(position=p_inv, euler_angles=euler_inv)

    def compose(self, other: "Pose6DOF") -> "Pose6DOF":
        if other is None:
            raise ValueError("other must be provided")
        R1 = self.rotation_matrix
        p1 = self._position
        R2 = other.rotation_matrix
        p2 = other._position

        R = R1 @ R2
        p = R1 @ p2 + p1

        return Pose6DOF(position=p, euler_angles=rotation_matrix_to_euler(R))

    def transform_point(self, point: Vec3 | list[float]) -> Vec3:
        if point is None:
            raise ValueError("point must be provided")
        point = np.asarray(point, dtype=np.float64)
        return self.rotation_matrix @ point + self._position

    def transform_vector(self, vector: Vec3 | list[float]) -> Vec3:
        if vector is None:
            raise ValueError("vector must be provided")
        vector = np.asarray(vector, dtype=np.float64)
        return self.rotation_matrix @ vector

    def copy(self) -> "Pose6DOF":
        return Pose6DOF(
            position=self._position.copy(),
            euler_angles=self._euler_angles.copy(),
        )

    def __eq__(self, other: object) -> bool:
        if other is None:
            raise ValueError("other must be provided")
        if not isinstance(other, Pose6DOF):
            return False
        return bool(
            np.allclose(self._position, other._position, atol=1e-10)
            and np.allclose(self._euler_angles, other._euler_angles, atol=1e-10)
        )

    def __repr__(self) -> str:
        return (
            f"Pose6DOF(position=[{self.x:.4f}, {self.y:.4f}, {self.z:.4f}], "
            f"euler=[{self.roll:.4f}, {self.pitch:.4f}, {self.yaw:.4f}])"
        )
