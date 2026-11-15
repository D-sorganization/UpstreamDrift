"""Capsule procedural hand visual backend (legacy :mod:`grip_hand_mesh`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.settings import (
    default_myohand_settings,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandVisualMeshes,
    HandVisualSide,
)


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandVisualSide,
    joint_angles: Mapping[str, float] | None = None,
) -> HandVisualMeshes:
    """Single-shell capsule hand in the grip hand-grip frame."""
    if not isinstance(grip_spec, GripSpec):
        raise TypeError(f"grip_spec must be GripSpec, got {type(grip_spec).__name__}")
    if joint_angles:
        raise NotImplementedError("capsule backend ignores joint_angles overrides")
    pose = grip_spec.to_grip_pose()
    hand = Hand.LEAD if side == "lead" else Hand.TRAIL
    cfg = default_myohand_settings()
    baked = build_grip_hand_mesh(hand, grip_radius_m=cfg.grip_radius_m, pose=pose)
    return {"hand_shell": baked.mesh}


__all__ = ["build_hand_visual"]
