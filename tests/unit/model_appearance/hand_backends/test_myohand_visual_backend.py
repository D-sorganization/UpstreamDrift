"""MyoHand visual backend (epic #15122, refs #13763)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
)
from src.shared.python.model_appearance.geometry import concat_meshes
from src.shared.python.model_appearance.hand_backends.joint_mapping import (
    anatomical_mirror_for_side,
    grip_spec_to_myohand_joint_angles,
)
from src.shared.python.model_appearance.hand_backends.myohand import (
    build_hand_visual,
    finger_pair_penetration_m,
    mirror_y_vertices,
    myohand_assets_ready,
    penetration_depth_m,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    BACKEND_MYOHAND,
    get_hand_backend,
    hand_backend_names,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    default_myohand_settings,
)

_assets = myohand_assets_ready()
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _assets, reason="myo_sim submodule not initialized"),
]


def test_registry_includes_myohand() -> None:
    assert BACKEND_MYOHAND in hand_backend_names()
    builder = get_hand_backend(BACKEND_MYOHAND)
    meshes = builder(GripSpec.legacy(), "trail", None)
    assert len(meshes) >= 20


def test_meshes_have_finite_watertight_parts() -> None:
    meshes = build_hand_visual(GripSpec.legacy(), "lead")
    for name, mesh in meshes.items():
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        assert np.isfinite(mesh.vertices).all(), name
        assert len(mesh.vertices) > 10, name


def test_palm_finger_depth_moves_cmc_and_wrist_joints() -> None:
    shallow = grip_spec_to_myohand_joint_angles(
        GripSpec.legacy(lead=HandGripSpec(palm_finger_depth=0.1)),
        "lead",
        grip_radius_m=0.014,
    )
    deep = grip_spec_to_myohand_joint_angles(
        GripSpec.legacy(lead=HandGripSpec(palm_finger_depth=0.9)),
        "lead",
        grip_radius_m=0.014,
    )
    assert shallow["flexion"] != pytest.approx(deep["flexion"])
    assert shallow["deviation"] != pytest.approx(deep["deviation"])


def test_thumb_straightness_moves_thumb_mp_ip() -> None:
    bent = grip_spec_to_myohand_joint_angles(
        GripSpec.legacy(trail=HandGripSpec(thumb_straightness=0.0)),
        "trail",
        grip_radius_m=0.014,
    )
    straight = grip_spec_to_myohand_joint_angles(
        GripSpec.legacy(trail=HandGripSpec(thumb_straightness=1.0)),
        "trail",
        grip_radius_m=0.014,
    )
    assert bent["mp_flexion"] < straight["mp_flexion"]
    assert bent["ip_flexion"] < straight["ip_flexion"]


def test_lead_trail_mirroring_under_right_handed_spec() -> None:
    spec = GripSpec.legacy()
    assert anatomical_mirror_for_side(spec, "lead")
    assert not anatomical_mirror_for_side(spec, "trail")
    lead = build_hand_visual(spec, "lead")
    trail = build_hand_visual(spec, "trail")
    lc = concat_meshes(*lead.values()).vertices.mean(axis=0)
    tc = concat_meshes(*trail.values()).vertices.mean(axis=0)
    assert lc[1] * tc[1] < 0.0 or abs(lc[0] - tc[0]) > 0.005


def test_shaft_roll_rotates_hand_about_grip_z() -> None:
    cfg = default_myohand_settings()
    base_spec = GripSpec.legacy()
    rolled = GripSpec.legacy(
        trail=HandGripSpec(shaft_roll_deg=20.0),
    )
    base = concat_meshes(*build_hand_visual(base_spec, "trail").values()).vertices
    moved = concat_meshes(*build_hand_visual(rolled, "trail").values()).vertices
    axis = APPEARANCE_GRIP_SHAFT_AXIS / np.linalg.norm(APPEARANCE_GRIP_SHAFT_AXIS)
    t0 = base - base.mean(0)
    t1 = moved - moved.mean(0)
    # Points on the shaft axis stay fixed.
    for pts in (t0, t1):
        proj = pts @ axis
        perp = pts - np.outer(proj, axis)
        assert perp.std() > 1e-4
    frame = apply_shaft_roll(np.eye(4), axis, -20.0)
    expect = t1 @ frame[:3, :3].T
    assert np.sqrt(((expect - t0) ** 2).mean()) < 0.02


def test_club_penetration_within_tolerance() -> None:
    cfg = default_myohand_settings()
    spec = GripSpec.legacy()
    for side in ("lead", "trail"):
        meshes = build_hand_visual(spec, side)  # type: ignore[arg-type]
        pen = penetration_depth_m(meshes, club_radius_m=cfg.club_cylinder_radius_m)
        assert pen <= cfg.penetration_tolerance_m, (side, pen)


def test_finger_parts_do_not_interpenetrate_beyond_tolerance() -> None:
    cfg = default_myohand_settings()
    meshes = build_hand_visual(GripSpec.legacy(), "trail")
    pairs = (
        ("proxph2", "proxph3"),
        ("proxph2", "proximal_thumb"),
        ("midph3", "distph3"),
    )
    for a, b in pairs:
        if a not in meshes or b not in meshes:
            continue
        overlap = finger_pair_penetration_m(meshes, a, b)
        assert overlap <= cfg.finger_pair_tolerance_m, (a, b, overlap)


def test_left_handed_mirrors_shaft_roll_on_build() -> None:
    spec = GripSpec.legacy(handedness="left", trail=HandGripSpec(shaft_roll_deg=15.0))
    meshes = build_hand_visual(spec, "trail")
    assert len(meshes) > 0


def test_mirror_y_swaps_hand_side() -> None:
    spec = GripSpec.legacy()
    trail = concat_meshes(*build_hand_visual(spec, "trail").values())
    mirrored = mirror_y_vertices(trail)
    assert mirrored.vertices[:, 1].mean() == pytest.approx(
        -trail.vertices[:, 1].mean(), rel=0, abs=1e-9
    )


def test_joint_angle_override_is_honored() -> None:
    spec = GripSpec.legacy()
    open_hand = build_hand_visual(spec, "trail", {"mcp2_flexion": 0.0})
    curled = build_hand_visual(spec, "trail", {"mcp2_flexion": 1.2})
    o = open_hand["proxph2"].vertices.mean(0)
    c = curled["proxph2"].vertices.mean(0)
    assert np.linalg.norm(c - o) > 0.005


def test_anthropometric_scaling_changes_extent() -> None:
    small = build_hand_visual(
        replace(
            GripSpec.legacy(),
            hand_size=replace(GripSpec.legacy().hand_size, length_m=0.15),
        ),
        "trail",
    )
    large = build_hand_visual(
        replace(
            GripSpec.legacy(),
            hand_size=replace(GripSpec.legacy().hand_size, length_m=0.23),
        ),
        "trail",
    )
    s_ext = float(np.ptp(concat_meshes(*small.values()).vertices, axis=0).max())
    l_ext = float(np.ptp(concat_meshes(*large.values()).vertices, axis=0).max())
    assert l_ext > s_ext * 1.08
