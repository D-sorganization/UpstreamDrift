"""Penetration and clearance metrics for hand visual backends."""

from __future__ import annotations

import pytest

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.metrics import (
    grip_shaft_penetration_fraction,
    inter_finger_penetration_fraction,
)
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.grip_hand_mesh import HandMeshParams
from src.shared.python.model_appearance.hand_backends.comparison import (
    build_for_comparison,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_VISUAL_GRIP_RADIUS_M,
    HandVisualSettings,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend_id", ["capsule", "procedural_skin"])
def test_no_shaft_or_finger_penetration_beyond_tolerance(backend_id: str) -> None:
    spec = GripSpec.legacy()
    settings = HandVisualSettings()
    for role in ("lead", "trail"):
        vis = build_for_comparison(
            backend_id,
            spec,
            role,
            grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
            settings=settings,
        )
        shaft_pct = grip_shaft_penetration_fraction(
            vis.mesh,
            grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
            tolerance_m=settings.shaft_penetration_tolerance_m,
        )
        hand = Hand.LEAD if role == "lead" else Hand.TRAIL
        reference = build_grip_hand_mesh(
            hand,
            grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
            pose=spec.to_grip_pose(),
            params=HandMeshParams(scale=hand_mesh_scale(spec.hand_size.length_m)),
        )
        finger_clouds = [
            (part.finger, part.mesh.vertices)
            for part in reference.components
            if part.finger in FINGER_NAMES and part.name.endswith("_proximal")
        ]
        finger_pct = inter_finger_penetration_fraction(
            finger_clouds,
            tolerance_m=settings.finger_penetration_tolerance_m,
        )
        assert shaft_pct <= settings.max_shaft_penetration_fraction
        assert finger_pct <= settings.max_finger_penetration_fraction
