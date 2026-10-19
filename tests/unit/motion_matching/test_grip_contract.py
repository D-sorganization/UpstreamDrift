"""Anatomical grip contract: (alpha, psi, gamma) <-> Euler bijection (HRG-GR1, #13202)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance.grip_pose import GripPose, GripStyle, Hand
from src.shared.python.motion_matching import anthropometric_geometry as geometry
from src.shared.python.motion_matching.grip_contract import (
    GRIP_STRENGTH_PRESETS_DEG,
    NEUTRAL_EULER_DEG,
    PALM_PRESETS_DEG,
    GripSpec,
    HandAngles,
    follower_rotation,
    from_grip_rotation,
    grip_matrix,
    hand_from_euler,
    hand_to_euler,
    to_grip_rotation,
)
from src.shared.python.motion_matching.grip_fit import grip_rotation

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
NATIVE = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
SHAFT = np.array([0.0, 1.0, 0.0])  # butt to head, club frame
P = np.diag([1.0, -1.0, 1.0])  # sagittal mirror (GR0 convention)


def _native_rotations() -> dict[str, np.ndarray]:
    doc = json.loads(NATIVE.read_text())
    out: dict[str, np.ndarray] = {}
    for joint in doc["joints"]:
        for side, suffix in (("L", "Clubface Vector"), ("R", "RHandStandoff")):
            if joint["child"].endswith(suffix):
                out[side] = np.asarray(joint["child_to_follower"], float)[:3, :3]
    return out


def _committed_rotations(
    name: str,
) -> tuple[dict[str, list[float]], dict[str, np.ndarray]]:
    doc = json.loads((MODELS / name).read_text())
    rot: dict[str, np.ndarray] = {}
    for joint in doc["joints"]:
        for side, suffix in (("L", "Clubface Vector"), ("R", "RHandStandoff")):
            if joint["child"].endswith(suffix):
                rot[side] = np.asarray(joint["child_to_follower"], float)[:3, :3]
    return doc["subject"]["grip_rotation_deg"], rot


def test_round_trip_1000_seeded_samples() -> None:
    rng = np.random.default_rng(13202)
    worst_deg, worst_frob = 0.0, 0.0
    for _ in range(1000):
        a = HandAngles(
            alpha_deg=rng.uniform(-60, 60),
            psi_deg=rng.uniform(15, 75),
            gamma_deg=rng.uniform(-90, 90),
        )
        euler = hand_to_euler(a)
        back = hand_from_euler(euler)
        worst_deg = max(
            worst_deg,
            abs(back.alpha_deg - a.alpha_deg),
            abs(back.psi_deg - a.psi_deg),
            abs(back.gamma_deg - a.gamma_deg),
        )
        m0 = grip_matrix(a.alpha_deg, a.psi_deg, a.gamma_deg)
        m1 = grip_matrix(back.alpha_deg, back.psi_deg, back.gamma_deg)
        via_euler = Rotation.from_euler("x", 25.0, degrees=True).as_matrix() @ (
            grip_rotation(euler)
        )
        worst_frob = max(
            worst_frob,
            float(np.linalg.norm(m0 - m1)),
            float(np.linalg.norm(m0 - via_euler)),
        )
    assert worst_deg <= 1e-9
    assert worst_frob <= 1e-12


@pytest.mark.parametrize(
    "euler",
    [
        {"L": (-90.0, 47.0, 0.0), "R": (-34.0, 46.0, 62.0)},
        {"L": (-36.864, 57.315, 70.926), "R": (9.769, 52.918, 83.914)},
    ],
)
def test_decompose_recompose_matches_builder_matrices(euler: dict) -> None:
    native = _native_rotations()
    for side, e in euler.items():
        expected = native[side] @ (
            Rotation.from_euler("x", 25.0, degrees=True).as_matrix() @ grip_rotation(e)
        )
        angles = hand_from_euler(e)
        got = follower_rotation(native[side], angles)
        assert np.abs(got - expected).max() <= 1e-9


@pytest.mark.parametrize(
    "name", ["full_body_spec_anthro_driver.json", "full_body_spec_anthro_iron7.json"]
)
def test_reproduces_committed_child_to_follower(name: str) -> None:
    triples, committed = _committed_rotations(name)
    native = _native_rotations()
    for side in ("L", "R"):
        angles = hand_from_euler(triples[side])
        got = follower_rotation(native[side], angles)
        assert np.abs(got - committed[side]).max() <= 1e-9, (name, side)


def test_default_build_reproduced_by_the_contract() -> None:
    doc = geometry.build_upper_body(
        json.loads(NATIVE.read_text()), stature_m=1.71, mass_kg=78.0
    )
    native = _native_rotations()
    for joint in doc["joints"]:
        for side, suffix in (("L", "Clubface Vector"), ("R", "RHandStandoff")):
            if joint["child"].endswith(suffix):
                angles = hand_from_euler(geometry.GRIP_ROTATION_DEG[side])
                got = follower_rotation(native[side], angles)
                rot = np.asarray(joint["child_to_follower"], float)[:3, :3]
                assert np.abs(got - rot).max() <= 1e-9


def test_published_decomposition_of_the_defaults() -> None:
    lead = hand_from_euler(geometry.GRIP_ROTATION_DEG["L"])
    assert (lead.alpha_deg, lead.psi_deg, lead.gamma_deg) == pytest.approx(
        (0.0, 25.0, 47.0), abs=1e-9
    )


def test_native_follower_frames_share_the_shaft_axis() -> None:
    """Both hands: club<-follower of the native frames is R_y(80 deg)."""
    native = _native_rotations()
    closure = np.asarray(json.loads(NATIVE.read_text())["closure"]["placement_b"])
    ry80 = Rotation.from_euler("y", 80.0, degrees=True).as_matrix()
    assert np.abs(native["L"] - ry80).max() <= 1e-12
    assert np.abs(closure[:3, :3] @ native["R"] - ry80).max() <= 1e-12


def test_alpha_is_a_pure_rotation_about_the_shaft() -> None:
    native = _native_rotations()
    base = HandAngles(0.0, 40.0, 0.0)
    for side in ("L", "R"):
        for alpha in (-15.0, 15.0, 33.0):
            a = follower_rotation(native[side], HandAngles(alpha, 40.0, 0.0))
            b = follower_rotation(native[side], base)
            closure = np.asarray(
                json.loads(NATIVE.read_text())["closure"]["placement_b"]
            )
            q = np.eye(3) if side == "L" else closure[:3, :3]
            twist = (q @ a) @ (q @ b).T  # club-frame rotation between the two
            rotvec = Rotation.from_matrix(twist).as_rotvec()
            assert np.abs(np.degrees(rotvec) - np.array([0.0, alpha, 0.0])).max() < 1e-9


def test_gamma_is_a_flexion_offset_of_the_distal_primitive() -> None:
    """R_z(gamma) on the right of M commutes with the distal Rz primitive."""
    for flex in (0.0, 20.0, -35.0):
        left = (
            Rotation.from_euler("z", flex, degrees=True).as_matrix()
            @ grip_matrix(10.0, 40.0, 0.0).T
        )
        right = Rotation.from_euler("z", flex + 12.0, degrees=True).as_matrix() @ (
            grip_matrix(10.0, 40.0, 12.0).T
        )
        assert np.abs(left - right).max() <= 1e-12


def test_strong_preset_rotates_visual_and_physical_by_alpha_with_no_drift() -> None:
    neutral = GripSpec.preset("neutral")
    strong = GripSpec.preset("strong")
    native = _native_rotations()["L"]
    c2f = np.eye(4)
    c2f[:3, :3] = follower_rotation(native, neutral.lead)
    c2f[:3, 3] = [-0.025, -1.0762, 0.0591]
    strong_c2f = c2f.copy()
    strong_c2f[:3, :3] = follower_rotation(native, strong.lead)
    strong_c2f[:3, 3] = c2f[:3, 3]
    twist = strong_c2f[:3, :3] @ c2f[:3, :3].T
    rotvec = Rotation.from_matrix(twist).as_rotvec()
    assert np.degrees(rotvec[1]) == pytest.approx(15.0, abs=1e-9)
    assert np.abs(rotvec[[0, 2]]).max() < 1e-12
    assert np.abs(strong_c2f[:3, 3] - c2f[:3, 3]).max() <= 1e-12
    # Visual: the lead V direction turns by the same 15 deg about the grip axis.
    v0 = neutral.pose().landmarks(Hand.LEAD)["v_direction"]
    v1 = strong.pose().landmarks(Hand.LEAD)["v_direction"]
    assert np.degrees(np.arccos(np.clip(v0 @ v1, -1, 1))) == pytest.approx(
        15.0, abs=1e-9
    )
    grip_axis = np.array([0.0, 0.0, -1.0])  # butt to head in the body frame
    assert np.cross(v0, v1) @ grip_axis > 0.0  # right-hand rotation about g


def test_left_handed_configuration_is_the_reflected_right_handed_one() -> None:
    right = GripSpec.preset("strong")
    left = GripSpec.preset("strong", handedness="left")
    for hand in (Hand.LEAD, Hand.TRAIL):
        for name, point in right.pose().landmarks(hand).items():
            np.testing.assert_allclose(
                P @ point, left.pose().landmarks(hand)[name], atol=1e-9, err_msg=name
            )
    r, lft = to_grip_rotation(right), to_grip_rotation(left)
    assert set(r) == {"L", "R"} == set(lft)
    assert r["L"] == lft["R"] and r["R"] == lft["L"]


def test_presets_use_golf_words() -> None:
    assert GRIP_STRENGTH_PRESETS_DEG == {"weak": -15.0, "neutral": 0.0, "strong": 15.0}
    assert PALM_PRESETS_DEG == {"fingers": 25.0, "neutral": 40.0, "palm": 65.0}
    assert GripSpec.preset("weak").lead.alpha_deg == -15.0
    assert GripSpec.preset("neutral", palm="fingers").lead.psi_deg == 25.0
    with pytest.raises(ValueError, match="preset"):
        GripSpec.preset("sideways")


def test_neutral_transform_is_published_and_explicit() -> None:
    assert pytest.approx((-75.0, 0.0, 0.0), abs=1e-9) == NEUTRAL_EULER_DEG
    assert to_grip_rotation(GripSpec.preset("neutral"))["L"] == pytest.approx(
        NEUTRAL_EULER_DEG, abs=1e-9
    )


def test_grip_strength_is_alpha_not_a_second_knob() -> None:
    spec = GripSpec.preset("strong")
    assert spec.grip_strength_deg == spec.lead.alpha_deg == 15.0
    pose = spec.pose()
    assert isinstance(pose, GripPose)
    assert pose.lead_v_azimuth_deg == pytest.approx(35.0)
    assert pose.palm_diagonal_deg == pytest.approx(spec.lead.psi_deg)
    assert GripSpec.preset("neutral").pose() == GripPose()


def test_from_grip_rotation_inverts_to_grip_rotation() -> None:
    spec = GripSpec(
        lead=HandAngles(-12.0, 33.0, 20.0),
        trail=HandAngles(18.0, 61.0, -40.0),
        style=GripStyle.INTERLOCK,
        lead_below_butt_m=0.04,
        hand_spacing_m=0.07,
    )
    back = from_grip_rotation(to_grip_rotation(spec), template=spec)
    assert back.lead.alpha_deg == pytest.approx(-12.0, abs=1e-9)
    assert back.trail.gamma_deg == pytest.approx(-40.0, abs=1e-9)
    assert back.style is GripStyle.INTERLOCK


def test_dbc_rejects_bad_values_and_the_singularity() -> None:
    with pytest.raises(ValueError, match="alpha_deg"):
        HandAngles(alpha_deg=float("nan"), psi_deg=40.0, gamma_deg=0.0)
    with pytest.raises(ValueError, match="psi_deg"):
        HandAngles(alpha_deg=0.0, psi_deg=0.0, gamma_deg=0.0)
    with pytest.raises(ValueError, match="singular"):
        # psi = 0 deg puts the YXZ middle angle at -90 deg.
        hand_from_euler(
            Rotation.from_matrix(
                Rotation.from_euler("x", -25.0, degrees=True).as_matrix()
                @ grip_matrix(0.0, 0.0, 0.0)
            ).as_euler("xyz", degrees=True)
        )
    with pytest.raises(ValueError, match="three"):
        hand_from_euler((1.0, 2.0))
    with pytest.raises(AttributeError):
        GripSpec.preset("neutral").lead.alpha_deg = 1.0  # type: ignore[misc]
