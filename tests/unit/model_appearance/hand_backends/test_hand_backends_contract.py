"""Comparison harness backends: capsule baseline plus the shared registry."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.model_appearance.hand_backends import capsule
from src.shared.python.model_appearance.hand_backends.comparison import (
    build_for_comparison,
    comparison_backend_names,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    list_hand_visual_backends,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_VISUAL_GRIP_RADIUS_M,
)

pytestmark = pytest.mark.unit


def test_harness_lists_capsule_baseline_then_registered_backends() -> None:
    names = comparison_backend_names()
    assert names[0] == "capsule"
    assert "procedural_skin" in names
    assert "procedural_skin" in list_hand_visual_backends()
    assert "capsule" not in list_hand_visual_backends()  # ADR-0057 A3


def test_build_for_comparison_rejects_unknown_backend() -> None:
    with pytest.raises(KeyError, match="unknown"):
        build_for_comparison(
            "not_a_backend",
            GripSpec.legacy(),
            "lead",
            grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        )


def test_capsule_baseline_matches_legacy_mesh_vertex_count() -> None:
    from src.shared.python.model_appearance.grip_hand_mesh import (
        build_grip_hand_mesh,
    )
    from src.shared.python.model_appearance.grip_pose import Hand

    spec = GripSpec.legacy()
    vis = build_for_comparison(
        "capsule", spec, "lead", grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M
    )
    legacy = build_grip_hand_mesh(
        Hand.LEAD,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        pose=spec.to_grip_pose(),
    )
    assert vis.mesh.vertices.shape == legacy.mesh.vertices.shape
    assert vis.backend == "capsule"


def test_capsule_baseline_rolls_rigidly_about_the_grip_axis() -> None:
    spec = GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=20.0))
    base = capsule.build_hand_visual(
        GripSpec.legacy(), "lead", grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M
    )
    rolled = capsule.build_hand_visual(
        spec, "lead", grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M
    )
    r = apply_shaft_roll(
        np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, resolved_shaft_roll_deg(spec, "lead")
    )[:3, :3]
    np.testing.assert_allclose(
        rolled.mesh.vertices, base.mesh.vertices @ r.T, atol=1e-9
    )


def test_left_handedness_mirrors_lead_trail_x_extent() -> None:
    def width(spec: GripSpec) -> float:
        mesh = capsule.build_hand_visual(
            spec, "lead", grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M
        ).mesh
        return float(mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min())

    assert width(GripSpec.legacy()) == pytest.approx(
        width(GripSpec.legacy(handedness="left")), rel=0.02
    )
