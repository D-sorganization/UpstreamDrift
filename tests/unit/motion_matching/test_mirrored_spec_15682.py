"""Contract tests of the left-handed mirror of a full-body spec (#15682 step 1).

All on the committed right-handed driver spec. World and pelvis frames are
x forward, y left, z up, so the sagittal reflection is ``M = diag(1, -1, 1)``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.full_body_spec import (
    validate_full_body_structure,
)
from src.shared.python.motion_matching.mirrored_spec import (
    MIRROR,
    body_reflections,
    mirror_coordinates,
    mirror_name,
    mirror_spec_to_left_handed,
    negated_coordinates,
)

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
CLUB = "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector"
STANDOFF = "solid_reference:GolfSwing3D_Kinetic/Grip/LHandStandoff"


@pytest.fixture(scope="module")
def right() -> dict[str, Any]:
    return json.loads(SPEC.read_text("utf-8"))


@pytest.fixture(scope="module")
def left(right: dict[str, Any]) -> dict[str, Any]:
    return mirror_spec_to_left_handed(right)


def _h(reflection: Any) -> Any:
    """4x4 homogeneous form of a 3x3 matrix."""
    out = np.eye(4)
    out[:3, :3] = reflection
    return out


def _frame_world(document: dict[str, Any], q: dict[str, float]) -> dict[str, Any]:
    poses = spec_body_poses(document, q)
    return {f["name"]: poses[f["body"]] @ f["placement"] for f in document["frames"]}


def _q(document: dict[str, Any], random: bool) -> dict[str, float]:
    names = document["coordinate_order"]
    values = np.random.default_rng(15682).uniform(-0.6, 0.6, len(names))
    return dict(zip(names, values * random, strict=True))


def _assert_close(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            _assert_close(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list) and a and not isinstance(a[0], (int, float)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, (list, float)):
        np.testing.assert_allclose(a, b, atol=1e-12, err_msg=path)
    else:
        assert a == b, path


def test_names_swap_sides_as_an_involution(right: dict[str, Any]) -> None:
    assert mirror_name("x/RHandStandoff") == "x/LHandStandoff"
    assert mirror_name("x/HubtoLS") == "x/HubtoRS"
    assert mirror_name(CLUB) == CLUB
    elbow = "solid_reference:GolfSwing3D_Kinetic/Left Elbow Joint/Spherical Solid"
    assert mirror_name(elbow) == elbow.replace("Left", "Right") + "1"
    for name in ("COMRod", "solid_reference:x", "LWInputY", "hip_flexion_l", "LW"):
        assert mirror_name(mirror_name(name)) == name
    bodies = {b["name"] for b in right["bodies"]}
    assert {mirror_name(n) for n in bodies} - bodies == {STANDOFF}


def test_right_hand_carries_the_club_and_left_hand_is_welded(
    right: dict[str, Any], left: dict[str, Any]
) -> None:
    wrist = next(j for j in left["joints"] if j["child"] == CLUB)
    assert "Right Wrist" in wrist["name"] and wrist["parent"].endswith("RLowerForearm")
    assert left["closure"]["body_a"] == STANDOFF
    assert left["closure"]["name"].endswith("LeftHandOnClubForce")
    frames = {f["name"]: f["body"] for f in left["frames"]}
    assert (frames["RW"], frames["LW"]) == (CLUB, STANDOFF)
    zero = _q(right, random=False)
    assert (
        _frame_world(right, zero)["LW"][1, 3] > 0 > _frame_world(left, zero)["RW"][1, 3]
    )
    assert subject_handedness(left) == "left"
    grip = left["subject"]["grip_rotation_deg"]
    assert grip["R"] == right["subject"]["grip_rotation_deg"]["L"]


def test_masses_equal_inertia_mirrored_and_structure_valid(
    right: dict[str, Any], left: dict[str, Any]
) -> None:
    solids = {s["name"]: s for b in left["bodies"] for s in b["solids"]}
    for solid in (s for b in right["bodies"] for s in b["solids"]):
        twin = solids.pop(mirror_name(solid["name"]))
        assert twin["mass_kg"] == solid["mass_kg"]
        np.testing.assert_allclose(
            np.linalg.eigvalsh(twin["inertia_com_kg_m2"]),
            np.linalg.eigvalsh(solid["inertia_com_kg_m2"]),
            atol=1e-15,
        )
    assert not solids
    validate_full_body_structure(left)
    for joint in left["joints"]:
        for field in ("parent_to_base", "child_to_follower"):
            det = np.linalg.det(np.asarray(joint[field])[:3, :3])
            assert det == pytest.approx(1.0, abs=1e-12)


def test_mirror_of_mirror_is_the_input_which_is_not_mutated(
    right: dict[str, Any], left: dict[str, Any]
) -> None:
    before = json.dumps(right, sort_keys=True)
    mirror_spec_to_left_handed(right)
    assert json.dumps(right, sort_keys=True) == before
    with pytest.raises(ValueError, match="right-handed"):
        mirror_spec_to_left_handed(left)
    reset = copy.deepcopy(left)
    reset["subject"]["handedness"] = "right"
    back = mirror_spec_to_left_handed(reset)
    expected = copy.deepcopy(right)
    for doc in (back, expected):
        doc.pop("handedness_mirror", None)
        doc.pop("provenance")
        doc["subject"].pop("handedness", None)
    _assert_close(back, expected)


def test_only_root_coordinates_are_negated(right: dict[str, Any]) -> None:
    assert negated_coordinates(right) == {"TranslationInputY", "HipInputX", "HipInputZ"}
    q = {"TranslationInputY": 0.2, "HipInputX": 0.1, "LSInputX": 0.3}
    want = {"TranslationInputY": -0.2, "HipInputX": -0.1, "RSInputX": 0.3}
    assert mirror_coordinates(right, q) == want


@pytest.mark.parametrize("random", [False, True], ids=["zero", "random"])
def test_forward_kinematics_is_the_mirror_image(
    right: dict[str, Any], left: dict[str, Any], random: bool
) -> None:
    q = _q(right, random)
    q_left = mirror_coordinates(right, q)
    mirrored = spec_body_poses(left, q_left)
    n = body_reflections(right)
    for name, pose in spec_body_poses(right, q).items():
        expected = _h(MIRROR) @ pose @ _h(n[name])
        np.testing.assert_allclose(mirrored[mirror_name(name)], expected, atol=1e-12)
    mirrored_frames = _frame_world(left, q_left)
    for name, pose in _frame_world(right, q).items():
        np.testing.assert_allclose(
            mirrored_frames[mirror_name(name)][:3, 3], MIRROR @ pose[:3, 3], atol=1e-12
        )


def _weld_error(document: dict[str, Any], q: dict[str, float]) -> Any:
    closure = document["closure"]
    poses = spec_body_poses(document, q)
    a = poses[closure["body_a"]] @ np.asarray(closure["placement_a"])
    b = poses[closure["body_b"]] @ np.asarray(closure["placement_b"])
    return np.linalg.inv(a) @ b


def test_closure_weld_error_is_mirrored(
    right: dict[str, Any], left: dict[str, Any]
) -> None:
    q = _q(right, random=True)
    n_a = _h(body_reflections(right)[right["closure"]["body_a"]])
    np.testing.assert_allclose(
        _weld_error(left, mirror_coordinates(right, q)),
        n_a @ _weld_error(right, q) @ n_a,
        atol=1e-12,
    )


def test_limits_markers_and_club_frame_follow_the_mirror(
    right: dict[str, Any], left: dict[str, Any]
) -> None:
    ranges = left["coordinate_ranges_deg"]
    assert ranges["RScapInputX"] == right["coordinate_ranges_deg"]["LScapInputX"]
    assert left["address_seed_deg"]["RSInputY"] == right["address_seed_deg"]["LSInputY"]
    assert left["coordinate_order"][9] == "REInput"
    markers, original = left["marker_attachments"], right["marker_attachments"]
    assert markers["RElbowOut"]["body"] == "RS"
    assert markers["RKneeOut"] == {"body": "femur_r", "offset_m": None}
    np.testing.assert_allclose(
        markers["BackRight"]["offset_m"], original["BackRight"]["offset_m"]
    )
    assert left["tibial_torsion_deg"]["l"] == right["tibial_torsion_deg"]["r"]
    rh, lh = assembly_from_spec(right), assembly_from_spec(left)
    assert rh is not None and lh is not None and lh.handedness == "left"
    assert (lh.length_m, lh.axis_offset_m) == (rh.length_m, rh.axis_offset_m)
