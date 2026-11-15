"""Hand visual backend contract: registry, capsule baseline, shaft roll."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    hand_mesh_with_shaft_roll,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    build_hand_visual,
    iter_registered_backends,
    registered_backend_ids,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_VISUAL_GRIP_RADIUS_M,
    HandVisualSettings,
)

pytestmark = pytest.mark.unit


def _joint_angles(spec: GripSpec, role: str) -> HandJointAngles:
    return HandJointAngles.from_grip_spec(spec, role)  # type: ignore[arg-type]


def test_registry_lists_capsule_and_procedural_skin() -> None:
    ids = registered_backend_ids()
    assert "capsule" in ids
    assert "procedural_skin" in ids
    assert len(list(iter_registered_backends())) >= 2


def test_build_hand_visual_rejects_unknown_backend() -> None:
    spec = GripSpec.legacy()
    with pytest.raises(KeyError, match="unknown"):
        build_hand_visual(
            "not_a_backend",
            spec,
            "lead",
            _joint_angles(spec, "lead"),
            grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        )


def test_capsule_backend_matches_legacy_mesh_vertex_count() -> None:
    from src.shared.python.model_appearance.grip_hand_mesh import (
        build_grip_hand_mesh,
    )
    from src.shared.python.model_appearance.grip_pose import Hand

    spec = GripSpec.legacy()
    angles = _joint_angles(spec, "lead")
    settings = HandVisualSettings()
    vis = build_hand_visual(
        "capsule",
        spec,
        "lead",
        angles,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=settings,
    )
    legacy = build_grip_hand_mesh(
        Hand.LEAD,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        pose=spec.to_grip_pose(),
    )
    assert vis.mesh.vertices.shape == legacy.mesh.vertices.shape
    assert vis.backend_id == "capsule"


def test_shaft_roll_rotates_mesh_about_grip_axis() -> None:
    spec = GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=20.0))
    angles = _joint_angles(spec, "lead")
    base = build_hand_visual(
        "capsule",
        spec,
        "lead",
        angles,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
    )
    rolled = hand_mesh_with_shaft_roll(base.mesh, spec, "lead")
    tip = base.mesh.vertices.mean(axis=0)
    roll_deg = resolved_shaft_roll_deg(spec, "lead")
    r = apply_shaft_roll(np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, roll_deg)[:3, :3]
    expected = (r @ tip).astype(float)
    actual = rolled.vertices.mean(axis=0)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_left_handedness_mirrors_lead_trail_x_extent() -> None:
    right = GripSpec.legacy()
    left = GripSpec.legacy(handedness="left")
    angles_r = _joint_angles(right, "lead")
    angles_l = _joint_angles(left, "lead")
    mesh_r = build_hand_visual(
        "capsule", right, "lead", angles_r, grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M
    ).mesh
    mesh_l = build_hand_visual(
        "capsule", left, "lead", angles_l, grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M
    ).mesh
    width_r = float(mesh_r.vertices[:, 0].max() - mesh_r.vertices[:, 0].min())
    width_l = float(mesh_l.vertices[:, 0].max() - mesh_l.vertices[:, 0].min())
    assert width_r == pytest.approx(width_l, rel=0.02)
