"""Legacy 16-capsule hand visual: the comparison-harness baseline.

The capsule hand is the default drawn hand and is not a registered backend
(ADR-0057 A3); the harness builds it directly through this module.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import (
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    HandVisualSettings,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandSide,
    HandVisualResult,
)
from src.shared.python.model_appearance.library import MATERIALS

BACKEND_ID = "capsule"


def rolled_about_shaft(mesh: Mesh, spec: GripSpec, side: HandSide) -> Mesh:
    """``mesh`` turned rigidly about the shaft axis by the hand's resolved roll."""
    rotation = apply_shaft_roll(
        np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, resolved_shaft_roll_deg(spec, side)
    )[:3, :3]
    return Mesh(mesh.vertices @ rotation.T, mesh.faces.copy())


def skin_rgb_for(material: str) -> tuple[float, float, float]:
    """RGB of a skin material from the appearance library (``skin_2`` ...)."""
    r, g, b, _ = MATERIALS[material].base_color
    return (float(r), float(g), float(b))


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float,
    skin_rgb: tuple[float, float, float] | None = None,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> HandVisualResult:
    """:func:`build_grip_hand_mesh`, rolled about the shaft like every backend."""
    hand = Hand.LEAD if side == "lead" else Hand.TRAIL
    baked = build_grip_hand_mesh(
        hand,
        grip_radius_m=float(grip_radius_m),
        pose=grip_spec.to_grip_pose(),
        params=HandMeshParams(scale=hand_mesh_scale(grip_spec.hand_size.length_m)),
    )
    return HandVisualResult(
        backend=BACKEND_ID,
        side=side,
        mesh=rolled_about_shaft(baked.mesh, grip_spec, side),
        skin_rgb=skin_rgb or skin_rgb_for(settings.skin_material),
        grip_radius_m=float(grip_radius_m),
    )


__all__ = ["BACKEND_ID", "build_hand_visual", "rolled_about_shaft", "skin_rgb_for"]
