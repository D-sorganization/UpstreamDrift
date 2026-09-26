"""Rigid poses and power-consistent twist/wrench frame changes (IA-U2, #9703).

Conventions (identical to Tools ``golf_club.types.RigidTransform`` and
``swing_sim.delivery_interchange``):

- A :class:`Pose` maps child coordinates into its parent:
  ``p_parent = rotation @ p_child + translation_m``.
- Quaternions are ``(w, x, y, z)`` and must already be unit length.
- A twist is ``(linear, angular)``: the velocity of the frame's origin point
  and the body angular velocity, both expressed in that frame. A wrench is
  ``(force, moment)`` with the moment taken about that frame's origin.

Forces, moments and velocities are always transformed together through the
Plücker motion transform ``X`` and its dual ``X* = X^-T`` (UpstreamDrift's
``spatial_algebra.transforms``), so wrench power ``F·v + M·ω`` is invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.physics._pre_impact_contracts import (
    FloatArray,
    fail,
    finite_array,
    identifier,
    proper_rotation,
    unit_quaternion,
)
from src.shared.python.spatial_algebra.pose6dof.rotations import (
    quaternion_to_rotation_matrix,
)
from src.shared.python.spatial_algebra.transforms import inv_xtrans, xtrans


@dataclass(frozen=True, eq=False)
class Pose:
    """Proper rigid pose of ``frame_id`` expressed in ``parent_frame_id``."""

    parent_frame_id: str
    frame_id: str
    rotation: FloatArray
    translation_m: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parent_frame_id", identifier(self.parent_frame_id, "parent_frame_id")
        )
        object.__setattr__(self, "frame_id", identifier(self.frame_id, "frame_id"))
        object.__setattr__(
            self,
            "rotation",
            proper_rotation(self.rotation, f"{self.frame_id}.rotation"),
        )
        object.__setattr__(
            self,
            "translation_m",
            finite_array(self.translation_m, (3,), f"{self.frame_id}.translation_m"),
        )

    @classmethod
    def from_quaternion(
        cls,
        parent_frame_id: str,
        frame_id: str,
        quaternion_wxyz: object,
        translation_m: object,
    ) -> Pose:
        """Build from a unit ``(w, x, y, z)`` quaternion; never normalizes."""
        quaternion = unit_quaternion(quaternion_wxyz, f"{frame_id}.quaternion_wxyz")
        rotation = quaternion_to_rotation_matrix(quaternion)
        return cls(parent_frame_id, frame_id, rotation, translation_m)  # type: ignore[arg-type]

    @classmethod
    def identity(cls, frame_id: str) -> Pose:
        return cls(frame_id, frame_id, np.eye(3), np.zeros(3))

    def inverse(self) -> Pose:
        """Pose of the parent expressed in this frame."""
        rotation_t = self.rotation.T
        return Pose(
            self.frame_id,
            self.parent_frame_id,
            rotation_t,
            -rotation_t @ self.translation_m,
        )

    def compose(self, child: Pose) -> Pose:
        """``self`` (A<-B) composed with ``child`` (B<-C) gives A<-C."""
        if child.parent_frame_id != self.frame_id:
            fail(
                "frame_mismatch",
                f"cannot compose {self.frame_id!r} with parent "
                f"{child.parent_frame_id!r}",
            )
        return Pose(
            self.parent_frame_id,
            child.frame_id,
            self.rotation @ child.rotation,
            self.rotation @ child.translation_m + self.translation_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_frame_id": self.parent_frame_id,
            "frame_id": self.frame_id,
            "rotation": self.rotation.tolist(),
            "translation_m": self.translation_m.tolist(),
        }


def _featherstone_arguments(pose: Pose) -> tuple[FloatArray, FloatArray]:
    # Featherstone xtrans(E, r): E rotates child->parent coordinates and r is
    # the parent origin measured from the child origin in child coordinates.
    return pose.rotation, -pose.rotation.T @ pose.translation_m


def twist_to_parent(
    pose: Pose, linear: object, angular: object
) -> tuple[FloatArray, FloatArray]:
    """Express a child-frame twist in the parent, referenced at its origin."""
    rotation, offset = _featherstone_arguments(pose)
    motion = np.concatenate(
        [finite_array(angular, (3,), "angular"), finite_array(linear, (3,), "linear")]
    )
    mapped = xtrans(rotation, offset) @ motion
    return mapped[3:], mapped[:3]


def wrench_to_parent(
    pose: Pose, force: object, moment: object
) -> tuple[FloatArray, FloatArray]:
    """Express a child-frame wrench in the parent, moment about its origin."""
    rotation, offset = _featherstone_arguments(pose)
    wrench = np.concatenate(
        [finite_array(moment, (3,), "moment"), finite_array(force, (3,), "force")]
    )
    mapped = inv_xtrans(rotation, offset).T @ wrench
    return mapped[3:], mapped[:3]


def shift_wrench_origin(
    force: object, moment: object, from_point: object, to_point: object
) -> tuple[FloatArray, FloatArray]:
    """Moment about ``to_point``: ``M' = M + (from - to) x F`` (same frame)."""
    force_array = finite_array(force, (3,), "force")
    lever = finite_array(from_point, (3,), "from_point") - finite_array(
        to_point, (3,), "to_point"
    )
    return force_array, finite_array(moment, (3,), "moment") + np.cross(
        lever, force_array
    )


def shift_twist_reference(
    linear: object, angular: object, from_point: object, to_point: object
) -> tuple[FloatArray, FloatArray]:
    """Point velocity at ``to_point``: ``v' = v + w x (to - from)``."""
    angular_array = finite_array(angular, (3,), "angular")
    lever = finite_array(to_point, (3,), "to_point") - finite_array(
        from_point, (3,), "from_point"
    )
    return finite_array(linear, (3,), "linear") + np.cross(
        angular_array, lever
    ), angular_array


__all__ = [
    "Pose",
    "shift_twist_reference",
    "shift_wrench_origin",
    "twist_to_parent",
    "wrench_to_parent",
]
