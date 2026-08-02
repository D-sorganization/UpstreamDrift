import math
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from .pose import Pose6DOF
from .rotations import axis_angle_to_rotation_matrix, rotation_matrix_to_euler
from .transform import Transform6DOF

Vec3: TypeAlias = npt.NDArray[np.float64]


@dataclass
class EntityPlacement:
    """
    High-level entity/offense placement for simulation models.
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
        if name is None:
            raise ValueError("name must be provided")
        self.name = name
        self.pose = pose if pose is not None else Pose6DOF()
        self.metadata = metadata if metadata is not None else {}

    def move_to(self, x: float, y: float, z: float) -> None:
        self.pose.position = np.array([x, y, z], dtype=np.float64)

    def move_by(self, dx: float = 0, dy: float = 0, dz: float = 0) -> None:
        self.pose._position += np.array([dx, dy, dz], dtype=np.float64)

    def rotate_euler(self, roll: float = 0, pitch: float = 0, yaw: float = 0) -> None:
        if roll is None:
            raise ValueError("roll must be provided")
        self.pose.euler_angles = np.array([roll, pitch, yaw], dtype=np.float64)

    def set_yaw(self, yaw: float) -> None:
        self.pose.yaw = yaw

    def rotate_axis(self, axis: Vec3 | list[float], angle: float) -> None:
        if axis is None:
            raise ValueError("axis must be provided")
        R = axis_angle_to_rotation_matrix(axis, angle)
        new_euler = rotation_matrix_to_euler(R @ self.pose.rotation_matrix)
        self.pose.euler_angles = new_euler

    def look_at(
        self, target: Vec3 | list[float], up: Vec3 | list[float] | None = None
    ) -> None:
        if target is None:
            raise ValueError("target must be provided")
        target = np.asarray(target, dtype=np.float64)
        if up is None:
            up = np.array([0, 0, 1], dtype=np.float64)
        else:
            up = np.asarray(up, dtype=np.float64)

        forward = target - self.pose.position
        forward_norm = math.hypot(forward[0], forward[1], forward[2])
        if forward_norm < 1e-10:
            return
        forward = forward / forward_norm

        right = np.cross(forward, up)
        right_norm = math.hypot(right[0], right[1], right[2])
        if right_norm < 1e-10:
            right = np.array([0, 1, 0], dtype=np.float64)
        else:
            right = right / right_norm

        up_corrected = np.cross(right, forward)

        R = np.column_stack([forward, right, up_corrected])
        self.pose.euler_angles = rotation_matrix_to_euler(R)

    @property
    def forward_vector(self) -> Vec3:
        return self.pose.rotation_matrix @ np.array([1, 0, 0], dtype=np.float64)

    @property
    def right_vector(self) -> Vec3:
        return self.pose.rotation_matrix @ np.array([0, 1, 0], dtype=np.float64)

    @property
    def up_vector(self) -> Vec3:
        return self.pose.rotation_matrix @ np.array([0, 0, 1], dtype=np.float64)

    def distance_to(self, point: Vec3 | list[float]) -> float:
        if point is None:
            raise ValueError("point must be provided")
        point = np.asarray(point, dtype=np.float64)
        arr = np.ravel(self.pose.position - point)
        return 0.0 if arr.size == 0 else math.hypot(*arr)

    def distance_to_entity(self, other: "EntityPlacement") -> float:
        arr = np.ravel(self.pose.position - other.pose.position)
        return 0.0 if arr.size == 0 else math.hypot(*arr)

    def to_transform(self) -> Transform6DOF:
        return Transform6DOF.from_pose(self.pose)

    @classmethod
    def from_transform(
        cls,
        name: str,
        transform: Transform6DOF,
        metadata: dict[str, Any] | None = None,
    ) -> "EntityPlacement":
        return cls(name=name, pose=transform.to_pose(), metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.pose.position.tolist(),
            "euler_angles": self.pose.euler_angles.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityPlacement":
        if data is None:
            raise ValueError("data must be provided")
        pose = Pose6DOF(
            position=data["position"],
            euler_angles=data["euler_angles"],
        )
        return cls(
            name=data["name"],
            pose=pose,
            metadata=data.get("metadata", {}),
        )

    def copy(self) -> "EntityPlacement":
        return EntityPlacement(
            name=self.name,
            pose=self.pose.copy(),
            metadata=deepcopy(self.metadata),
        )

    def __repr__(self) -> str:
        return f"EntityPlacement(name='{self.name}', {self.pose})"


class PlacementGroup:
    """
    Manages a collection of entity placements.
    """

    def __init__(self) -> None:
        self._entities: dict[str, EntityPlacement] = {}

    def add(self, entity: EntityPlacement) -> None:
        self._entities[entity.name] = entity

    def remove(self, name: str) -> None:
        if name in self._entities:
            del self._entities[name]

    def get(self, name: str) -> EntityPlacement | None:
        return self._entities.get(name)

    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self) -> Iterator[EntityPlacement]:
        return iter(self._entities.values())

    def translate_all(self, offset: Vec3 | list[float]) -> None:
        if offset is None:
            raise ValueError("offset must be provided")
        offset = np.asarray(offset, dtype=np.float64)
        for entity in self._entities.values():
            entity.pose._position += offset

    def rotate_around_point(
        self,
        point: Vec3 | list[float],
        axis: Vec3 | list[float],
        angle: float,
    ) -> None:
        if point is None:
            raise ValueError("point must be provided")
        point = np.asarray(point, dtype=np.float64)
        R = axis_angle_to_rotation_matrix(axis, angle)

        for entity in self._entities.values():
            rel_pos = entity.pose.position - point
            new_pos = R @ rel_pos + point
            entity.pose.position = new_pos

            new_euler = rotation_matrix_to_euler(R @ entity.pose.rotation_matrix)
            entity.pose.euler_angles = new_euler

    @property
    def centroid(self) -> Vec3:
        if not self._entities:
            return np.zeros(3, dtype=np.float64)
        positions = np.array([e.pose.position for e in self._entities.values()])
        return np.mean(positions, axis=0)

    @property
    def bounding_box(self) -> dict[str, Vec3]:
        if not self._entities:
            return {"min": np.zeros(3), "max": np.zeros(3)}
        positions = np.array([e.pose.position for e in self._entities.values()])
        return {
            "min": np.min(positions, axis=0),
            "max": np.max(positions, axis=0),
        }

    def copy(self) -> "PlacementGroup":
        new_group = PlacementGroup()
        for entity in self._entities.values():
            new_group.add(entity.copy())
        return new_group

    def __repr__(self) -> str:
        return f"PlacementGroup({len(self)} entities)"
