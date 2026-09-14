"""Engine-agnostic visual skeleton derived from a body/joint specification."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import visual_skeleton as module

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FULL_BODY = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)


def _t(translation) -> list[list[float]]:
    m = np.eye(4)
    m[:3, 3] = translation
    return m.tolist()


def _synthetic_spec() -> dict:
    """world -> trunk (joint at trunk origin) -> arm (joint 0.4 m up the trunk)."""
    return {
        "gravity_m_s2": [0.0, 0.0, -9.81],
        "coordinate_order": ["root_rz", "arm_rz"],
        "bodies": [
            {
                "name": "trunk",
                "solids": [
                    {
                        "name": "trunk",
                        "mass_kg": 10.0,
                        "com_m": [0.0, 0.0, 0.2],
                        "inertia_com_kg_m2": np.eye(3).tolist(),
                        "placement": np.eye(4).tolist(),
                    }
                ],
            },
            {
                "name": "arm",
                "solids": [
                    {
                        "name": "arm",
                        "mass_kg": 2.0,
                        "com_m": [0.15, 0.0, 0.0],
                        "inertia_com_kg_m2": (0.01 * np.eye(3)).tolist(),
                        "placement": np.eye(4).tolist(),
                    }
                ],
            },
        ],
        "joints": [
            {
                "name": "root",
                "parent": "world",
                "child": "trunk",
                "parent_to_base": _t([0, 0, 1.0]),
                "child_to_follower": _t([0, 0, 0]),
                "primitives": [{"primitive": "Rz", "coordinate": "root_rz"}],
            },
            {
                "name": "shoulder",
                "parent": "trunk",
                "child": "arm",
                "parent_to_base": _t([0, 0, 0.4]),
                "child_to_follower": _t([0, 0, 0]),
                "primitives": [{"primitive": "Rz", "coordinate": "arm_rz"}],
            },
        ],
        "frames": [{"name": "hand", "body": "arm", "placement": _t([0.3, 0, 0])}],
    }


def test_capsules_follow_joint_origins_and_leaves_reach_their_com() -> None:
    skeleton = module.derive_visual_skeleton(_synthetic_spec())
    by_body = {c.body: c for c in skeleton.capsules}
    trunk = by_body["trunk"]
    np.testing.assert_allclose(trunk.start_m, [0, 0, 0])
    np.testing.assert_allclose(trunk.end_m, [0, 0, 0.4])
    arm = by_body["arm"]
    np.testing.assert_allclose(arm.start_m, [0, 0, 0])
    np.testing.assert_allclose(arm.end_m, [0.15, 0, 0])  # leaf: joint -> COM
    assert 0.015 <= trunk.radius_m <= 0.05 and arm.radius_m < trunk.radius_m
    kinds = {s.kind for s in skeleton.spheres}
    assert kinds == {"com", "frame"}
    assert any(s.kind == "frame" and s.body == "arm" for s in skeleton.spheres)
    assert skeleton.ground.normal == pytest.approx((0.0, 0.0, 1.0))
    assert skeleton.ground.height_m == 0.0 and skeleton.ground.calibrated is False


def test_world_segments_follow_body_poses() -> None:
    skeleton = module.derive_visual_skeleton(_synthetic_spec())
    pose_arm = np.eye(4)
    pose_arm[:3, 3] = [1.0, 2.0, 3.0]
    poses = {"trunk": np.eye(4), "arm": pose_arm}
    segments = module.skeleton_world_segments(skeleton, poses)
    arm_segment = next(s for s in segments if s.body == "arm")
    np.testing.assert_allclose(arm_segment.start_m, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(arm_segment.end_m, [1.15, 2.0, 3.0])
    with pytest.raises(ValueError):
        module.skeleton_world_segments(skeleton, {"trunk": np.eye(4)})
    with pytest.raises(ValueError):
        module.skeleton_world_segments(skeleton, {"trunk": np.eye(3), "arm": np.eye(4)})


def test_full_body_spec_yields_a_complete_skeleton() -> None:
    spec = json.loads(FULL_BODY.read_text())
    skeleton = module.derive_visual_skeleton(spec)
    bodies = {b["name"] for b in spec["bodies"]} - {"world"}
    assert {c.body for c in skeleton.capsules} == bodies
    assert all(
        np.isfinite(c.start_m).all() and np.isfinite(c.end_m).all()
        for c in skeleton.capsules
    )
    assert all(c.length_m() > 0 for c in skeleton.capsules)
    assert len([s for s in skeleton.spheres if s.kind == "frame"]) == len(
        spec["frames"]
    )
    # Ground opposes gravity and is flagged uncalibrated until FB-4.
    assert skeleton.ground.calibrated is False
    assert skeleton.ground.normal == pytest.approx((0.0, 0.0, 1.0))


def test_upper_body_spec_without_contact_block_still_works() -> None:
    spec = json.loads(UPPER.read_text())
    skeleton = module.derive_visual_skeleton(spec)
    assert len(skeleton.capsules) == len(
        [b for b in spec["bodies"] if b["name"] != "world"]
    )


def test_rejects_disconnected_or_degenerate_specs() -> None:
    spec = _synthetic_spec()
    spec["joints"][1]["parent"] = "nowhere"
    with pytest.raises(ValueError):
        module.derive_visual_skeleton(spec)
    spec = _synthetic_spec()
    spec["gravity_m_s2"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        module.derive_visual_skeleton(spec)
