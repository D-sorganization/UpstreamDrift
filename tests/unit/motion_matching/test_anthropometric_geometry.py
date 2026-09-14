"""Tests for the anthropometric native upper-body geometry (AN-1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.native_model import (
    NativeMujocoModel,
)
from src.shared.python.motion_matching import anthropometric_geometry as module
from src.shared.python.motion_matching.anthropometric_candidate import (
    zero_pose_joint_positions,
)
from src.shared.python.motion_matching.anthropometry import segment_parameters

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
NATIVE = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
STATURE, MASS = 1.71, 78.0


@pytest.fixture(scope="module")
def upper() -> dict:
    return module.build_upper_body(
        json.loads(NATIVE.read_text()), stature_m=STATURE, mass_kg=MASS
    )


def _joint(upper: dict, child_suffix: str) -> dict:
    return next(j for j in upper["joints"] if j["child"].endswith(child_suffix))


def test_names_coordinates_and_closure_are_preserved(upper: dict) -> None:
    native = json.loads(NATIVE.read_text())
    assert upper["coordinate_order"] == native["coordinate_order"] + list(
        module.NECK_COORDINATES
    )
    assert {b["name"] for b in upper["bodies"]} == {
        b["name"] for b in native["bodies"]
    } | {module.HEAD_BODY}
    assert {j["name"] for j in upper["joints"]} == {
        j["name"] for j in native["joints"]
    } | {module.NECK_JOINT}
    assert {f["name"] for f in upper["frames"]} == {
        f["name"] for f in native["frames"]
    } | {"Head"}
    assert upper["closure"] == native["closure"]
    club_native = next(
        b for b in native["bodies"] if b["name"].endswith("Clubface Vector")
    )
    club = next(b for b in upper["bodies"] if b["name"].endswith("Clubface Vector"))
    assert club["solids"] == club_native["solids"]
    assert "unqualified" in upper["qualification"]


def test_zero_pose_lengths_follow_the_subject(upper: dict) -> None:
    anchors = zero_pose_joint_positions(upper)
    names = {j["child"].rsplit("/", 1)[-1]: j["name"] for j in upper["joints"]}

    def length(a: str, b: str) -> float:
        return float(np.linalg.norm(anchors[names[a]] - anchors[names[b]]))

    trunk = segment_parameters(STATURE, MASS, "trunk").length_m
    assert length("LowerTorso", "UpperTorsoBase") == pytest.approx(
        module.PELVIS_HEIGHT_FRACTION * trunk
    )
    assert length("UpperTorsoBase", "HubtoLS") == pytest.approx(
        trunk
        - module.SHOULDER_BELOW_CERVICALE_M
        - module.PELVIS_HEIGHT_FRACTION * trunk
    )
    assert length("HubtoLS", "LUpperArm") == pytest.approx(
        module.BIACROMIAL_FRACTION_OF_STATURE * STATURE / 2
    )
    assert length("LUpperArm", "RUpperArm") == pytest.approx(
        module.BIACROMIAL_FRACTION_OF_STATURE * STATURE
    )
    assert length("LUpperArm", "Spherical Solid") == pytest.approx(
        segment_parameters(STATURE, MASS, "upper_arm").length_m
    )
    assert length("Spherical Solid", "LLowerForearm") + length(
        "LLowerForearm", "Clubface Vector"
    ) == pytest.approx(segment_parameters(STATURE, MASS, "forearm").length_m)
    # Standing zero pose: hips to shoulder centre is about the trunk length.
    hub = anchors[names["HubtoLS"]]
    assert 0.44 < hub[2] < 0.50 and abs(hub[0]) < 1e-9 and abs(hub[1]) < 1e-9


def test_masses_and_inertias(upper: dict) -> None:
    total = sum(s["mass_kg"] for b in upper["bodies"] for s in b["solids"])
    club = sum(
        s["mass_kg"]
        for b in upper["bodies"]
        if b["name"].endswith(("Clubface Vector", "RHandStandoff"))
        for s in b["solids"]
    )
    expected = sum(
        segment_parameters(STATURE, MASS, seg).mass_kg * n
        for seg, n in (
            ("lower_trunk", 1),
            ("middle_trunk", 1),
            ("upper_trunk", 1),
            ("head", 1),
            ("upper_arm", 2.4),
            ("forearm", 2),
        )
    )
    assert total - club - 0.3 == pytest.approx(expected, rel=1e-6)  # 0.3 kg joint body
    assert abs(total - MASS * (1 - 2 * (0.1416 + 0.0433 + 0.0137))) / MASS < 0.05
    for b in upper["bodies"]:
        for s in b["solids"]:
            if s["mass_kg"] > 0:
                assert np.all(np.linalg.eigvalsh(np.array(s["inertia_com_kg_m2"])) > 0)


def test_joint_axes_are_anatomical_in_mujoco(upper: dict) -> None:
    model = NativeMujocoModel(json.dumps(upper).encode())
    m, d = model.model, model.data
    model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))
    up, left = np.array([0, 0, 1.0]), np.array([0, 1.0, 0])
    assert abs(d.xaxis[m.joint("TorsoInput").id] @ up) > 0.999
    assert abs(d.xaxis[m.joint("NeckInputX").id] @ np.array([1.0, 0, 0])) > 0.999
    assert abs(d.xaxis[m.joint("NeckInputY").id] @ left) > 0.999  # nod
    assert abs(d.xaxis[m.joint("NeckInputZ").id] @ up) > 0.999  # turn
    assert abs(d.xaxis[m.joint("SpineInputY").id] @ left) > 0.999  # flexion axis
    fwd = np.array([1.0, 0, 0])
    assert abs(d.xaxis[m.joint("LEInput").id] @ left) > 0.999  # elbow flexion
    assert abs(d.xaxis[m.joint("LFInput").id] @ fwd) > 0.999  # pronation along the arm
    assert len(model.coordinate_order) == 30
    # The neck sits at the cervicale, above the hub, and turns the head only.
    zero_pose = dict.fromkeys(model.coordinate_order, 0.0)
    turned = dict(zero_pose)
    turned["NeckInputZ"] = 1.0
    poses0, poses1 = model.frame_poses(zero_pose), model.frame_poses(turned)
    assert poses0["Head"][2, 3] > poses0["Hub"][2, 3] + 0.03
    np.testing.assert_allclose(poses1["Hub"], poses0["Hub"])
    assert not np.allclose(poses1["Head"][:3, :3], poses0["Head"][:3, :3])
    # Arms point forward at zero pose; elbow flexion (negative) lifts the wrist.
    straight = model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))["LW"][
        :3, 3
    ]
    shoulder = model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))["LS"][
        :3, 3
    ]
    assert straight[0] > shoulder[0] + 0.5 and abs(straight[2] - shoulder[2]) < 1e-9
    bent = dict.fromkeys(model.coordinate_order, 0.0)
    bent["LEInput"] = -0.5
    moved = model.frame_poses(bent)["LW"][:3, 3]
    assert moved[2] > straight[2] + 0.05 and abs(moved[1] - straight[1]) < 1e-9
    # Wrist cock (Rx) turns about the elbow axis, and a positive cock lifts the
    # club toward the elbow pit (+z at zero pose) on both sides.
    model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))
    for side in "LR":
        assert abs(d.xaxis[m.joint(f"{side}WInputX").id] @ left) > 0.999
        assert abs(d.xaxis[m.joint(f"{side}WInputY").id] @ fwd) > 0.999  # spin
    cocked = dict.fromkeys(model.coordinate_order, 0.0)
    cocked["LWInputX"] = 0.3
    head0 = model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))["Clubhead"]
    head1 = model.frame_poses(cocked)["Clubhead"]
    assert head1[2, 3] > head0[2, 3] + 0.2 and abs(head1[1, 3] - head0[1, 3]) < 1e-9
    # The address seed lowers the arms below the shoulders.
    seed = dict.fromkeys(model.coordinate_order, 0.0)
    seed.update({k: np.radians(v) for k, v in module.ADDRESS_SEED_DEG.items()})
    assert model.frame_poses(seed)["LW"][2, 3] < shoulder[2] - 0.2
    with pytest.raises(ValueError):
        module.build_upper_body(
            json.loads(NATIVE.read_text()), stature_m=0.0, mass_kg=MASS
        )


def test_ranges_seed_and_scapula_axes(upper: dict) -> None:
    ranges = upper["coordinate_ranges_deg"]
    assert set(ranges) <= set(upper["coordinate_order"])
    assert {"SpineInputX", "SpineInputY", "TorsoInput", "LEInput", "REInput"} <= set(
        ranges
    )
    assert all(lo < 0 < hi or lo <= 0 <= hi for lo, hi in ranges.values())
    assert set(upper["address_seed_deg"]) <= set(upper["coordinate_order"])
    for name, value in upper["address_seed_deg"].items():
        lo, hi = ranges.get(name, (-180.0, 180.0))
        assert lo <= value <= hi
    model = NativeMujocoModel(json.dumps(upper).encode())
    m, d = model.model, model.data
    model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))
    up, fwd = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    assert abs(d.xaxis[m.joint("LScapInputX").id] @ fwd) > 0.999  # elevation
    assert abs(d.xaxis[m.joint("LScapInputY").id] @ up) > 0.999  # protraction
    # Elevation lifts the left shoulder joint; protraction moves it forward.
    zero = model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))["LS"][:3, 3]
    q = dict.fromkeys(model.coordinate_order, 0.0)
    q["LScapInputX"] = 0.3
    assert model.frame_poses(q)["LS"][2, 3] > zero[2] + 0.03
    q = dict.fromkeys(model.coordinate_order, 0.0)
    q["LScapInputY"] = -0.3
    assert model.frame_poses(q)["LS"][0, 3] > zero[0] + 0.03
