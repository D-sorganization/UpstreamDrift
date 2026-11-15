"""Shared types for interchangeable hand visual backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh

HandRole = Literal["lead", "trail"]


@dataclass(frozen=True)
class HandJointAngles:
    """Visual articulation inputs shared by every backend (from :class:`GripSpec`)."""

    palm_finger_depth: float
    thumb_straightness: float
    club_cross_angle_deg: float

    @classmethod
    def from_grip_spec(cls, spec: GripSpec, role: HandRole) -> HandJointAngles:
        hand = spec.lead if role == "lead" else spec.trail
        return cls(
            palm_finger_depth=float(hand.palm_finger_depth),
            thumb_straightness=float(hand.thumb_straightness),
            club_cross_angle_deg=float(hand.club_cross_angle_deg),
        )


@dataclass(frozen=True)
class HandVisualResult:
    """One hand mesh in the grip-hand frame, plus render metadata."""

    backend_id: str
    role: HandRole
    mesh: Mesh
    grip_radius_m: float
    material: str
    palm_arch_angle_deg: float


def hand_mesh_with_shaft_roll(
    mesh: Mesh,
    spec: GripSpec,
    role: HandRole,
) -> Mesh:
    """Rotate vertices about the appearance shaft axis (z in the hand-grip frame)."""
    roll_deg = resolved_shaft_roll_deg(spec, role)
    if abs(roll_deg) <= 1e-15:
        return mesh
    rotation = apply_shaft_roll(np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, roll_deg)[:3, :3]
    return Mesh(mesh.vertices @ rotation.T, mesh.faces.copy())


__all__ = [
    "HandJointAngles",
    "HandRole",
    "HandVisualResult",
    "hand_mesh_with_shaft_roll",
]
