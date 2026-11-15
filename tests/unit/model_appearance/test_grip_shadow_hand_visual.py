"""Shadow Dexterous Hand grip visuals (#15122)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.grip_hands import grip_hand_frame, grip_hand_meshes
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand
from src.shared.python.model_appearance.grip_shadow_hand_assets import (
    load_shadow_hand_provenance,
    verify_shadow_mesh_asset_licences,
)
from src.shared.python.model_appearance.grip_shadow_hand_visual import (
    build_shadow_grip_hand_mesh,
    map_hand_grip_spec_to_wrist,
    penetration_metrics,
    solve_shadow_hand_qpos,
    visual_settings,
)
from src.shared.python.grip_spec.shaft_roll import apply_shaft_roll

pytestmark = pytest.mark.unit

mujoco = pytest.importorskip("mujoco")


def _iron7() -> ClubAssembly:
    return ClubAssembly("iron7", 0.95, 0.25, 0.0065)


def test_shadow_hand_provenance_gate() -> None:
    record = load_shadow_hand_provenance()
    assert record["licence"] == "Apache-2.0"
    assert record["mesh_obj_count"] == 14
    licence_audit = verify_shadow_mesh_asset_licences()
    assert licence_audit["bundle_licence_spdx"] == "Apache-2.0"
    assert licence_audit["runtime_matches_vendored_licence_copy"]
    assert licence_audit["mesh_obj_licence_spdx"] == "Apache-2.0"
    assert licence_audit["mesh_obj_count"] == record["mesh_obj_count"]


def test_shadow_hand_mesh_has_many_vertices_and_fingers() -> None:
    spec = GripSpec.legacy()
    mesh = build_shadow_grip_hand_mesh(
        Hand.TRAIL, grip_radius_m=0.012, pose=spec.to_grip_pose(), grip_spec=spec
    )
    assert len(mesh.vertices) > 20_000
    assert len(mesh.faces) > 20_000
    assert abs(mesh.volume()) > 0.0


def test_club_penetration_within_tolerance() -> None:
    spec = GripSpec.legacy()
    radius = 0.012
    settings = visual_settings()
    for hand in (Hand.LEAD, Hand.TRAIL):
        hand_spec = spec.lead if hand == Hand.LEAD else spec.trail
        solved = solve_shadow_hand_qpos(
            hand, hand_spec, grip_spec=spec, grip_radius_m=radius
        )
        metrics = penetration_metrics(
            hand,
            solved.qpos,
            grip_radius_m=radius,
            mount_matrix=solved.mount_matrix,
        )
        assert metrics["club_penetration_m"] <= settings.club_penetration_tolerance_m


def test_finger_finger_clearance_within_tolerance() -> None:
    spec = GripSpec.legacy()
    settings = visual_settings()
    solved = solve_shadow_hand_qpos(
        Hand.TRAIL, spec.trail, grip_spec=spec, grip_radius_m=0.012
    )
    metrics = penetration_metrics(
        Hand.TRAIL,
        solved.qpos,
        grip_radius_m=0.012,
        mount_matrix=solved.mount_matrix,
    )
    gap = metrics["finger_finger_gap_m"]
    assert gap + settings.finger_finger_penetration_tolerance_m >= 0


def test_shaft_roll_rotates_hand_about_grip_axis() -> None:
    club = _iron7()
    base_pose = GripSpec.legacy().to_grip_pose()
    rolled_pose = replace(
        GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=20.0)).to_grip_pose()
    )
    frame0 = grip_hand_frame(club, Hand.LEAD, base_pose)
    frame1 = grip_hand_frame(club, Hand.LEAD, rolled_pose)
    axis = np.array([0.0, 1.0, 0.0])
    rel = frame1[:3, :3] @ frame0[:3, :3].T
    angle = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    assert angle == pytest.approx(20.0, abs=0.5)
    for t in (-0.03, 0.0, 0.03):
        p0 = frame0[:3, 3] + t * axis
        p1 = frame1[:3, 3] + t * axis
        assert np.allclose(p0, p1, atol=1e-9)
    # appearance_frame_with_shaft_roll identity on fixed axis points
    moved = apply_shaft_roll(frame0, axis, 20.0)
    assert np.allclose(moved[:3, 3], frame0[:3, 3], atol=1e-12)


def test_left_handed_lead_uses_left_hand_xml_mirroring() -> None:
    spec = GripSpec.neutral(handedness="left")
    lead = build_shadow_grip_hand_mesh(
        Hand.LEAD,
        grip_radius_m=0.012,
        pose=spec.to_grip_pose(),
        grip_spec=spec,
    )
    trail = build_shadow_grip_hand_mesh(
        Hand.TRAIL,
        grip_radius_m=0.012,
        pose=spec.to_grip_pose(),
        grip_spec=spec,
    )
    assert lead.vertices.shape == trail.vertices.shape
    assert not np.allclose(lead.vertices, trail.vertices, atol=1e-3)


def test_hand_grip_spec_round_trip_wrist_mapping() -> None:
    spec = GripSpec.legacy(
        lead=HandGripSpec(
            club_cross_angle_deg=42.0,
            shaft_roll_deg=7.0,
            thumb_azimuth_deg=25.0,
        )
    )
    wrist = map_hand_grip_spec_to_wrist(spec.lead, handedness="right")
    restored = GripSpec.from_json(spec.to_json())
    again = map_hand_grip_spec_to_wrist(restored.lead, handedness="right")
    assert again == wrist


def test_grip_hand_meshes_use_shadow_not_capsule_count() -> None:
    club = _iron7()
    meshes = grip_hand_meshes(club, DEFAULT_GRIP_POSE, grip_spec=GripSpec.legacy())
    for label in ("lead_hand", "trail_hand"):
        assert len(meshes[label].vertices) > 10_000