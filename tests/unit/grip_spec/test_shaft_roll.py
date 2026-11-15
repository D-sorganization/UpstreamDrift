"""HRG-GB1b per-hand shaft roll about the grip axis (#15138)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact.binding import bind_rigid_hand_sites
from src.shared.python.grip_contact.contact_topology import contact_topology
from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.grip_spec.shaft_roll import (
    SHARED_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    hand_to_club_relative_rotation,
    resolved_shaft_roll_deg,
)
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_hands import (
    grip_hand_frame,
    grip_hand_meshes,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

pytestmark = pytest.mark.unit


def _iron7_club() -> ClubAssembly:
    return ClubAssembly("iron7", 0.95, 0.25, 0.0065)


_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/grip_shaft_roll"
)


def _palm_azimuth_deg(hand: Hand, pose: GripPose) -> float:
    mesh = build_grip_hand_mesh(hand, grip_radius_m=0.014, pose=pose)
    palm = next(c for c in mesh.components if c.name == "palm")
    c = palm.mesh.vertices.mean(axis=0)
    return math.degrees(math.atan2(float(c[1]), float(c[0])))


def test_shaft_roll_zero_matches_legacy_projection_and_mesh() -> None:
    projected = GripSpec.legacy().to_grip_pose()
    assert projected == DEFAULT_GRIP_POSE
    assert projected.lead_shaft_roll_deg == 0.0
    assert projected.trail_shaft_roll_deg == 0.0
    club = _iron7_club()
    placed = grip_hand_meshes(club, projected)
    baseline = grip_hand_meshes(club, DEFAULT_GRIP_POSE)
    for key in placed:
        assert np.allclose(placed[key].vertices, baseline[key].vertices, atol=1e-9)


def test_plus_minus_theta_is_identity() -> None:
    frame = np.eye(4)
    frame[:3, 0] = [0.0, 0.0, 1.0]
    frame[:3, 1] = [1.0, 0.0, 0.0]
    frame[:3, 2] = [0.0, 1.0, 0.0]
    out = apply_shaft_roll(
        apply_shaft_roll(frame, SHARED_GRIP_SHAFT_AXIS, 17.0),
        SHARED_GRIP_SHAFT_AXIS,
        -17.0,
    )
    assert np.allclose(out, frame, atol=1e-10)


def test_full_rotation_is_identity() -> None:
    frame = np.eye(4)
    out = apply_shaft_roll(frame, SHARED_GRIP_SHAFT_AXIS, 360.0)
    assert np.allclose(out, frame, atol=1e-10)


def test_lead_and_trail_roll_independent() -> None:
    base = GripSpec.legacy().to_grip_pose()
    lead_only = replace(
        GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=12.0)).to_grip_pose()
    )
    assert lead_only.lead_shaft_roll_deg == pytest.approx(12.0)
    club = _iron7_club()
    trail_frame_base = grip_hand_frame(club, Hand.TRAIL, base)
    trail_frame_lead = grip_hand_frame(club, Hand.TRAIL, lead_only)
    assert np.allclose(trail_frame_base, trail_frame_lead, atol=1e-12)
    lead_frame_base = grip_hand_frame(club, Hand.LEAD, base)
    lead_frame_lead = grip_hand_frame(club, Hand.LEAD, lead_only)
    assert not np.allclose(lead_frame_base, lead_frame_lead, atol=1e-6)


def test_roll_axis_fixes_shaft_line_points() -> None:
    pose = GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=20.0)).to_grip_pose()
    club = _iron7_club()
    frame = grip_hand_frame(club, Hand.LEAD, pose)
    axis = np.array([0.0, 1.0, 0.0])
    for t in (-0.05, 0.0, 0.05):
        p = frame[:3, 3] + t * axis
        p2 = apply_shaft_roll(frame, axis, 25.0)[:3, 3] + t * axis
        assert np.allclose(p, p2, atol=1e-12)


def test_left_handed_mirrors_shaft_roll_sign() -> None:
    spec = GripSpec.legacy(handedness="left", lead=HandGripSpec(shaft_roll_deg=10.0))
    assert resolved_shaft_roll_deg(spec, "lead") == pytest.approx(-10.0)


def test_cross_engine_hand_to_club_rotation_matches_binding_base() -> None:
    pose = GripSpec.legacy(
        lead=HandGripSpec(shaft_roll_deg=11.0),
        trail=HandGripSpec(shaft_roll_deg=-7.0),
    ).to_grip_pose()
    topo = contact_topology()
    from src.shared.python.grip_spec.shaft_roll import attachment_base_transform

    rel = hand_to_club_relative_rotation(pose, Hand.LEAD)
    base = attachment_base_transform(pose, Hand.LEAD)
    assert np.allclose(base[:3, :3], rel, atol=1e-10)
    sites = bind_rigid_hand_sites(topo, grip_pose=pose)
    lead_site = next(s for s in sites if s.site_id.startswith("lead_"))
    assert np.allclose(lead_site.body_to_site[:3, :3], rel, atol=1e-10)
    from src.engines.physics_engines.drake.python import grip_contact_sim as drake_grip
    from src.engines.physics_engines.mujoco.python import (
        grip_contact_sim as mujoco_grip,
    )

    mj = mujoco_grip.hand_attachment_rotation_for_pose(pose, Hand.LEAD)
    dr = drake_grip.hand_attachment_rotation_for_pose(pose, Hand.LEAD)
    assert np.allclose(mj, dr, atol=1e-12)
    assert np.allclose(mj, rel, atol=1e-12)


def test_shaft_roll_json_roundtrip() -> None:
    spec = GripSpec.legacy(
        lead=HandGripSpec(shaft_roll_deg=5.0),
        trail=HandGripSpec(shaft_roll_deg=-3.0),
    )
    restored = GripSpec.from_json(spec.to_json())
    assert restored.lead.shaft_roll_deg == pytest.approx(5.0)
    assert restored.trail.shaft_roll_deg == pytest.approx(-3.0)


def test_shaft_roll_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        HandGripSpec(shaft_roll_deg=181.0)


@pytest.mark.slow
def test_shaft_roll_preview_evidence_images() -> None:
    """Save ±15° lead-only preview meshes as committed evidence (#15138)."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    club = _iron7_club()
    cases = {
        "lead_minus_15deg": -15.0,
        "lead_plus_15deg": 15.0,
    }
    manifest: dict[str, object] = {"cases": {}}
    base_pose = GripSpec.legacy().to_grip_pose()
    for name, roll in cases.items():
        pose = GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=roll)).to_grip_pose()
        mesh = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=0.014, pose=pose)
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        v = mesh.mesh.vertices
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, c="tab:blue")
        ax.set_title(f"lead shaft_roll {roll:+.0f}°")
        out = _EVIDENCE_DIR / f"{name}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        manifest["cases"][name] = {
            "shaft_roll_deg": roll,
            "palm_azimuth_deg": _palm_azimuth_deg(Hand.LEAD, pose),
            "baseline_palm_azimuth_deg": _palm_azimuth_deg(Hand.LEAD, base_pose),
            "png": str(out.relative_to(_EVIDENCE_DIR.parents[2])),
        }
    receipt = _EVIDENCE_DIR / "receipt.json"
    receipt.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert receipt.is_file()
