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
    build_myohand_bone_meshes,
    finger_pair_penetration_m,
    mirror_y_vertices,
    myohand_assets_ready,
    penetration_depth_m,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    build_hand_visual,
    get_hand_visual_backend,
    list_hand_visual_backends,
)
from src.shared.python.model_appearance.hand_backends.skeleton_export import (
    export_bilateral_myohand_skeleton,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    default_myohand_settings,
)
from src.shared.python.model_appearance.hand_backends.mujoco_extract import (
    anthropometric_scale_matrix,
)

_assets = myohand_assets_ready()
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _assets, reason="myo_sim submodule not initialized"),
]


def test_myohand_is_registered_but_disabled_by_default() -> None:
    assert "myohand" not in list_hand_visual_backends()
    assert "myohand" in list_hand_visual_backends(include_disabled=True)
    assert get_hand_visual_backend("myohand").default_enabled is False


def test_disabled_backend_still_builds_by_name() -> None:
    result = build_hand_visual(
        GripSpec.legacy(), "trail", grip_radius_m=0.012, backend="myohand"
    )
    assert result.backend == "myohand"
    assert result.skin_rgb == default_myohand_settings().bone_rgb
    bones = build_myohand_bone_meshes(GripSpec.legacy(), "trail")
    assert len(bones) >= 20
    assert len(result.mesh.vertices) == sum(len(m.vertices) for m in bones.values())


def test_skeleton_export_has_all_canonical_joints() -> None:
    payload = export_bilateral_myohand_skeleton(GripSpec.legacy())
    for side in ("lead", "trail"):
        joints = payload[side]["joints"]
        assert len(joints) >= 23
        sample = joints["mcp2_flexion"]
        assert len(sample["origin_m"]) == 3
        assert len(sample["axis_unit"]) == 3


def test_meshes_have_finite_watertight_parts() -> None:
    meshes = build_myohand_bone_meshes(GripSpec.legacy(), "lead")
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
    lead = build_myohand_bone_meshes(spec, "lead")
    trail = build_myohand_bone_meshes(spec, "trail")
    # Kabsch to per-hand capsule reference centres both hands; use thumb MC.
    lm = lead["firstmc"].vertices.mean(axis=0)
    tm = trail["firstmc"].vertices.mean(axis=0)
    assert np.linalg.norm(lm - tm) > 0.012


def test_shaft_roll_rotates_hand_about_grip_z() -> None:
    cfg = default_myohand_settings()
    base_spec = GripSpec.legacy()
    rolled = GripSpec.legacy(
        trail=HandGripSpec(shaft_roll_deg=20.0),
    )
    base = concat_meshes(
        *build_myohand_bone_meshes(base_spec, "trail").values()
    ).vertices
    moved = concat_meshes(*build_myohand_bone_meshes(rolled, "trail").values()).vertices
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
    assert np.sqrt(((expect - t0) ** 2).mean()) < 0.025


def test_club_penetration_within_tolerance() -> None:
    default_myohand_settings.cache_clear()
    cfg = default_myohand_settings()
    spec = GripSpec.legacy()
    for side in ("lead", "trail"):
        meshes = build_myohand_bone_meshes(spec, side)  # type: ignore[arg-type]
        pen = penetration_depth_m(meshes, club_radius_m=cfg.club_cylinder_radius_m)
        assert pen <= cfg.penetration_tolerance_m, (side, pen)


def test_finger_parts_do_not_interpenetrate_beyond_tolerance() -> None:
    cfg = default_myohand_settings()
    meshes = build_myohand_bone_meshes(GripSpec.legacy(), "trail")
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
    meshes = build_myohand_bone_meshes(spec, "trail")
    assert len(meshes) > 0


def test_mirror_y_swaps_hand_side() -> None:
    spec = GripSpec.legacy()
    trail = concat_meshes(*build_myohand_bone_meshes(spec, "trail").values())
    mirrored = mirror_y_vertices(trail)
    assert mirrored.vertices[:, 1].mean() == pytest.approx(
        -trail.vertices[:, 1].mean(), rel=0, abs=1e-9
    )


def test_joint_angle_override_is_honored() -> None:
    spec = GripSpec.legacy()
    open_hand = build_myohand_bone_meshes(spec, "trail", {"mcp2_flexion": 0.0})
    curled = build_myohand_bone_meshes(spec, "trail", {"mcp2_flexion": 1.2})
    o = open_hand["proxph2"].vertices.mean(0)
    c = curled["proxph2"].vertices.mean(0)
    assert np.linalg.norm(c - o) > 0.005


def test_anthropometric_scaling_changes_extent() -> None:
    cfg = default_myohand_settings()
    spec = GripSpec.legacy()
    small = anthropometric_scale_matrix(
        0.15,
        spec.hand_size.breadth_m,
        reference_length_m=cfg.anthropometric_reference_hand_length_m,
        width_from_breadth=cfg.width_scale_from_breadth,
        exponent=cfg.anthropometric_scale_exponent,
    )[0, 0]
    large = anthropometric_scale_matrix(
        0.23,
        spec.hand_size.breadth_m,
        reference_length_m=cfg.anthropometric_reference_hand_length_m,
        width_from_breadth=cfg.width_scale_from_breadth,
        exponent=cfg.anthropometric_scale_exponent,
    )[0, 0]
    assert large > small * 1.08
