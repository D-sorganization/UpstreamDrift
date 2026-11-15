"""Legacy 16-capsule hand visual (unchanged generator)."""

from __future__ import annotations

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.grip_hand_mesh import (
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    HandRole,
    HandVisualResult,
)
from src.shared.python.model_appearance.hand_backends.metrics import palm_arch_angle_deg
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    HandVisualSettings,
)

BACKEND_ID = "capsule"


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    *,
    grip_radius_m: float,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> HandVisualResult:
    """Delegate to :func:`build_grip_hand_mesh` (legacy baseline, unchanged)."""
    if not isinstance(joint_angles, HandJointAngles):
        raise TypeError("joint_angles must be HandJointAngles")
    hand = Hand.LEAD if side == "lead" else Hand.TRAIL
    pose = grip_spec.to_grip_pose()
    scale = hand_mesh_scale(grip_spec.hand_size.length_m)
    params = HandMeshParams(scale=scale)
    baked = build_grip_hand_mesh(
        hand,
        grip_radius_m=float(grip_radius_m),
        pose=pose,
        params=params,
    )
    arch = palm_arch_angle_deg(baked.mesh)
    return HandVisualResult(
        backend_id=BACKEND_ID,
        role=side,
        mesh=baked.mesh,
        grip_radius_m=float(grip_radius_m),
        material=settings.skin_material,
        palm_arch_angle_deg=arch,
    )


__all__ = ["BACKEND_ID", "build_hand_visual"]
