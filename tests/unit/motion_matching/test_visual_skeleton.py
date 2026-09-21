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
                        "mass_kg": 0.5,
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
    by_body: dict[str, list] = {}
    for c in skeleton.capsules:
        by_body.setdefault(c.body, []).append(c)
    (trunk,) = by_body["trunk"]
    np.testing.assert_allclose(trunk.start_m, [0, 0, 0])
    np.testing.assert_allclose(trunk.end_m, [0, 0, 0.4])
    (arm,) = by_body["arm"]
    np.testing.assert_allclose(arm.start_m, [0, 0, 0])
    np.testing.assert_allclose(arm.end_m, [0.15, 0, 0])  # leaf: joint -> COM
    # Radii follow mass and length (uniform-density cylinder), clamped.
    assert trunk.radius_m == pytest.approx(module.capsule_radius(10.0, 0.4))
    assert arm.radius_m == pytest.approx(module.capsule_radius(0.5, 0.15))
    assert 0.006 <= arm.radius_m < trunk.radius_m <= 0.05
    kinds = {s.kind for s in skeleton.spheres}
    assert kinds == {"com", "frame"}
    assert any(s.kind == "frame" and s.body == "arm" for s in skeleton.spheres)
    assert skeleton.ground.normal == pytest.approx((0.0, 0.0, 1.0))
    assert skeleton.ground.height_m == 0.0 and skeleton.ground.calibrated is False


def test_one_capsule_per_child_joint_and_extension_to_a_far_com() -> None:
    spec = _synthetic_spec()
    spec["bodies"].append(
        {
            "name": "arm2",
            "solids": [
                {
                    "name": "arm2",
                    "mass_kg": 2.0,
                    "com_m": [0.15, 0.0, 0.0],
                    "inertia_com_kg_m2": (0.01 * np.eye(3)).tolist(),
                    "placement": np.eye(4).tolist(),
                }
            ],
        }
    )
    spec["joints"].append(
        {
            "name": "shoulder2",
            "parent": "trunk",
            "child": "arm2",
            "parent_to_base": _t([0.1, 0, 0.4]),
            "child_to_follower": _t([0, 0, 0]),
            "primitives": [{"primitive": "Rz", "coordinate": "arm2_rz"}],
        }
    )
    spec["coordinate_order"].append("arm2_rz")
    spec["bodies"][0]["solids"][0]["com_m"] = [0.0, 0.0, 0.9]  # head-like far COM
    skeleton = module.derive_visual_skeleton(spec)
    trunk = [c for c in skeleton.capsules if c.body == "trunk"]
    ends = sorted(tuple(np.round(c.end_m, 6)) for c in trunk)
    assert ends == [(0.0, 0.0, 0.4), (0.0, 0.0, 0.9), (0.1, 0.0, 0.4)]


def test_capsule_radius_policy() -> None:
    assert module.capsule_radius(0.3, 1.08) < module.capsule_radius(1.13, 0.23)
    assert module.capsule_radius(20.0, 0.3) == 0.05
    assert module.capsule_radius(1e-6, 1.0) == 0.006
    with pytest.raises(ValueError):
        module.capsule_radius(-1.0, 0.3)
    with pytest.raises(ValueError):
        module.capsule_radius(1.0, 0.0)


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
    radius = {c.body.rsplit("/", 1)[-1]: c.radius_m for c in skeleton.capsules}
    assert radius["Clubface Vector"] < radius["LLowerForearm"] < radius["LowerTorso"]
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
    assert len(skeleton.capsules) >= len(
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


def test_visual_hints_add_shapes_and_override_capsule_radius() -> None:
    spec = json.loads(FULL_BODY.read_text())
    club = next(
        b["name"] for b in spec["bodies"] if b["name"].endswith("Clubface Vector")
    )
    head = next(
        s["name"]
        for s in next(b for b in spec["bodies"] if b["name"] == club)["solids"]
        if s["name"].endswith("Clubhead")
    )
    spec["visual_hints"] = {
        "shapes": {
            head: {
                "shape": "ellipsoid",
                "half_size_m": [0.05, 0.06, 0.03],
                "center_m": [0, 0, 0.064],
            }
        },
        "capsule_radius_m": {club: 0.0065},
    }
    skeleton = module.derive_visual_skeleton(spec)
    assert any(s.body == club and s.kind == "ellipsoid" for s in skeleton.shapes)
    assert all(
        c.radius_m == pytest.approx(0.0065) for c in skeleton.capsules if c.body == club
    )
    spec["visual_hints"]["shapes"][head]["shape"] = "cone"
    with pytest.raises(ValueError):
        module.derive_visual_skeleton(spec)
    spec["visual_hints"]["shapes"][head]["shape"] = "box"
    spec["visual_hints"]["capsule_radius_m"][club] = -1.0
    with pytest.raises(ValueError):
        module.derive_visual_skeleton(spec)
