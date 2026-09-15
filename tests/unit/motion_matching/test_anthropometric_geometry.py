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
from src.shared.python.motion_matching.grip_fit import grip_rotation
from src.shared.python.motion_matching.anthropometric_candidate import (
    zero_pose_joint_positions,
)
from src.shared.python.motion_matching.anthropometry import segment_parameters
from src.shared.python.motion_matching.club_models import IRON_7

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
    # The closure keeps its bodies and rotations; the club moves the hands
    # along the shaft to the driver's length.
    for key in ("body_a", "body_b", "name", "placement_a"):
        assert upper["closure"][key] == native["closure"][key]
    np.testing.assert_allclose(
        np.array(upper["closure"]["placement_b"])[:3, :3],
        np.array(native["closure"]["placement_b"])[:3, :3],
    )
    club_native = next(
        b for b in native["bodies"] if b["name"].endswith("Clubface Vector")
    )
    club = next(b for b in upper["bodies"] if b["name"].endswith("Clubface Vector"))
    # Hand solids are the native ones (shifted along the shaft); the club itself
    # is the typical driver.
    hands = lambda body: sorted(  # noqa: E731
        (s["name"], s["mass_kg"])
        for s in body["solids"]
        if "Hand" in s["name"].rsplit("/", 1)[-1]
    )
    assert hands(club) == hands(club_native)
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
    wrist = m.joint("LWInputX").id
    shoulder = model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))["LS"][
        :3, 3
    ]
    straight = d.xanchor[wrist].copy()
    assert straight[0] > shoulder[0] + 0.5 and abs(straight[2] - shoulder[2]) < 1e-9
    bent = dict.fromkeys(model.coordinate_order, 0.0)
    bent["LEInput"] = -0.5
    model.frame_poses(bent)
    moved = d.xanchor[wrist].copy()
    assert moved[2] > straight[2] + 0.05 and abs(moved[1] - straight[1]) < 1e-9
    # Wrist cock (Rx) turns about the elbow axis, and a positive cock lifts the
    # club toward the elbow pit (+z at zero pose) on both sides.
    model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))
    for side in "LR":
        assert abs(d.xaxis[m.joint(f"{side}WInputX").id] @ left) > 0.999
        assert abs(d.xaxis[m.joint(f"{side}WInputY").id] @ fwd) < 0.01  # not a spin
        assert abs(d.xaxis[m.joint(f"{side}WInputY").id] @ left) < 0.01  # flexion
    # A neutral grip holds the shaft GRIP_ULNAR_OFFSET_DEG below the forearm
    # line, on the side away from the pit (with the fitted hand rotation off).
    neutral = module.build_upper_body(
        json.loads(NATIVE.read_text()),
        stature_m=STATURE,
        mass_kg=MASS,
        grip_rotation_deg={"L": (0.0, 0.0, 0.0), "R": (0.0, 0.0, 0.0)},
    )
    model = NativeMujocoModel(json.dumps(neutral).encode())
    poses = model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))
    forearm = poses["LW"][:3, 3] - poses["LE"][:3, 3]
    shaft = poses["Clubhead"][:3, 3] - poses["LW"][:3, 3]
    angle = np.degrees(
        np.arccos(forearm @ shaft / np.linalg.norm(forearm) / np.linalg.norm(shaft))
    )
    assert angle == pytest.approx(module.GRIP_ULNAR_OFFSET_DEG, abs=1.5)
    assert shaft[2] < 0  # toward the ulnar side (away from the pit, which is +z)
    cocked = dict.fromkeys(model.coordinate_order, 0.0)
    cocked["LWInputX"] = 0.3
    head0 = model.frame_poses(dict.fromkeys(model.coordinate_order, 0.0))["Clubhead"]
    head1 = model.frame_poses(cocked)["Clubhead"]
    assert head1[2, 3] > head0[2, 3] + 0.2 and abs(head1[1, 3] - head0[1, 3]) < 1e-9
    flexed = dict.fromkeys(model.coordinate_order, 0.0)
    flexed["LWInputY"] = 0.3
    head2 = model.frame_poses(flexed)["Clubhead"]
    assert abs(head2[1, 3] - head0[1, 3]) > 0.2  # flexion swings the club sideways
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


def test_document_carries_club_torso_hints_and_shared_ranges(upper: dict) -> None:
    assert upper["club"]["name"] == "driver"
    assert upper["club"]["total_mass_kg"] == pytest.approx(0.313)
    hints = upper["visual_hints"]
    assert len(hints["shapes"]) == 4  # pelvis, abdomen, thorax, club head
    assert all(
        v > 0 for shape in hints["shapes"].values() for v in shape["half_size_m"]
    )
    assert any(k.endswith("Clubface Vector") for k in hints["capsule_radius_m"])
    assert upper["coordinate_ranges_deg"]["LEInput"] == [-150.0, 5.0]
    iron = module.build_upper_body(
        json.loads(NATIVE.read_text()), stature_m=STATURE, mass_kg=MASS, club=IRON_7
    )
    assert iron["club"]["name"] == "iron7" and iron["club"]["length_m"] < 1.0


def test_fitted_grip_rotation_turns_each_hand_on_its_wrist(upper: dict) -> None:
    assert upper["subject"]["grip_rotation_deg"] == {
        k: list(v) for k, v in module.GRIP_ROTATION_DEG.items()
    }
    native = json.loads(NATIVE.read_text())
    neutral = module.build_upper_body(
        native,
        stature_m=STATURE,
        mass_kg=MASS,
        grip_rotation_deg={"L": (0.0, 0.0, 0.0), "R": (0.0, 0.0, 0.0)},
    )
    zero = dict.fromkeys(upper["coordinate_order"], 0.0)
    a = NativeMujocoModel(json.dumps(neutral).encode()).frame_poses(zero)
    b = NativeMujocoModel(json.dumps(upper).encode()).frame_poses(zero)
    # The wrist stays put; the club turns on it by the fitted rotation.
    np.testing.assert_allclose(a["LW"][:3, 3], b["LW"][:3, 3], atol=1e-12)
    rel = a["Clubhead"][:3, :3].T @ b["Clubhead"][:3, :3]
    expected = grip_rotation(module.GRIP_ROTATION_DEG["L"])
    turned = np.degrees(np.arccos((np.trace(rel) - 1) / 2))
    assert turned == pytest.approx(
        np.degrees(np.arccos((np.trace(expected) - 1) / 2)), abs=0.1
    )
    with pytest.raises(ValueError):
        module.build_upper_body(
            native, stature_m=STATURE, mass_kg=MASS, grip_rotation_deg={"L": (0, 0, 0)}
        )


def test_grip_roll_turns_the_hands_about_the_shaft_only() -> None:
    native = json.loads(NATIVE.read_text())
    plain = module.build_upper_body(native, stature_m=STATURE, mass_kg=MASS)
    rolled = module.build_upper_body(
        native, stature_m=STATURE, mass_kg=MASS, grip_roll_deg=40.0
    )
    zero = dict.fromkeys(plain["coordinate_order"], 0.0)
    a = NativeMujocoModel(json.dumps(plain).encode()).frame_poses(zero)
    b = NativeMujocoModel(json.dumps(rolled).encode()).frame_poses(zero)
    # The shaft direction from the wrist is unchanged; the club frame rolled.
    shaft_a = a["Clubhead"][:3, 3] - a["LW"][:3, 3]
    shaft_b = b["Clubhead"][:3, 3] - b["LW"][:3, 3]
    cos = shaft_a @ shaft_b / np.linalg.norm(shaft_a) / np.linalg.norm(shaft_b)
    assert cos > 0.999
    rel = a["Clubhead"][:3, :3].T @ b["Clubhead"][:3, :3]
    assert np.degrees(np.arccos((np.trace(rel) - 1) / 2)) == pytest.approx(
        40.0, abs=0.1
    )
    assert rolled["subject"]["grip_roll_deg"] == 40.0
