"""Tests for functional hip calibration and the hip-joint rewrite of a spec."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import hip_calibration as module
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
RECEIPT = ROOT / "docs/development/full_body_models/build_receipt.json"
WAIST = {
    "WaistLeft": (0.05, 0.12, 0.10),
    "WaistRight": (0.05, -0.12, 0.10),
    "WaistLBack": (-0.10, 0.08, 0.12),
    "WaistRBack": (-0.10, -0.08, 0.12),
}
CENTRES = {"r": np.array([0.0, -0.09, -0.05]), "l": np.array([0.0, 0.09, -0.05])}


def _synthetic_capture(
    frames: int = 40,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(7)
    labels = tuple(WAIST) + ("RKneeOut", "LKneeOut")
    points = np.zeros((frames, len(labels), 3))
    for f in range(frames):
        rot = Rotation.from_rotvec(rng.normal(0.0, 0.3, 3)).as_matrix()
        pos = rng.normal(0.0, 0.5, 3)
        for k, label in enumerate(WAIST):
            points[f, k] = rot @ np.array(WAIST[label]) + pos
        for side, k in (("r", 4), ("l", 5)):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            points[f, k] = rot @ (CENTRES[side] + 0.41 * direction) + pos
    valid = np.ones((frames, len(labels)), dtype=bool)
    valid[3, 4] = False
    return points, valid, labels


def test_sphere_fit_recovers_centre_and_radius() -> None:
    rng = np.random.default_rng(1)
    d = rng.normal(size=(30, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    fit = module.fit_sphere(np.array([1.0, -2.0, 0.5]) + 0.37 * d)
    np.testing.assert_allclose(fit.centre, [1.0, -2.0, 0.5], atol=1e-9)
    assert fit.radius_m == pytest.approx(0.37) and fit.residual_sd_m < 1e-9
    with pytest.raises(ValueError):
        module.fit_sphere(d[:3])


def test_functional_hip_calibration_recovers_centres_and_axes() -> None:
    points, valid, labels = _synthetic_capture()
    cal = module.functional_hip_calibration(points, valid, labels, WAIST)
    np.testing.assert_allclose(cal.centre_r, CENTRES["r"], atol=1e-8)
    np.testing.assert_allclose(cal.centre_l, CENTRES["l"], atol=1e-8)
    assert cal.radius_r_m == pytest.approx(0.41, abs=1e-8)
    assert cal.residual_sd_l_m < 1e-8 and cal.frames == 39
    axes = np.array(cal.pelvis_axes).T
    assert np.linalg.det(axes) == pytest.approx(1.0)
    np.testing.assert_allclose(axes[:, 2], [0.0, -1.0, 0.0], atol=1e-8)  # right
    np.testing.assert_allclose(axes[:, 1], [0.0, 0.0, 1.0], atol=1e-8)  # up = frame z
    np.testing.assert_allclose(axes[:, 0], [1.0, 0.0, 0.0], atol=1e-8)  # forward
    with pytest.raises(ValueError):
        module.functional_hip_calibration(
            points, valid, labels, WAIST, superior_axis=(0.0, 1.0, 0.0)
        )
    with pytest.raises(ValueError):
        module.functional_hip_calibration(
            points, valid, labels, {"WaistLeft": (0, 0, 0)}
        )


def test_apply_hip_calibration_moves_joints_and_keeps_the_rest() -> None:
    spec = json.loads(SPEC.read_text())
    receipt = json.loads(RECEIPT.read_text())
    points, valid, labels = _synthetic_capture()
    cal = module.functional_hip_calibration(points, valid, labels, WAIST)
    new = module.apply_hip_calibration(
        spec, cal, receipt["pelvis_alignment"]["hip_from_opensim_pelvis"]
    )
    hip_frame = np.array(
        next(f for f in spec["frames"] if f["name"] == "Hip")["placement"]
    )
    for side in ("r", "l"):
        joint = next(j for j in new["joints"] if j["name"] == f"hip_{side}")
        translation = np.array(joint["parent_to_base"])[:3, 3]
        expected = (hip_frame @ np.append(CENTRES[side], 1.0))[:3]
        np.testing.assert_allclose(translation, expected, atol=1e-9)
        rotation = np.array(joint["parent_to_base"])[:3, :3]
        assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert canonical_sha256(new) != canonical_sha256(spec)
    assert new["bodies"] == spec["bodies"] and new["contact"] == spec["contact"]
    assert [j for j in new["joints"] if not j["name"].startswith("hip_")] == [
        j for j in spec["joints"] if not j["name"].startswith("hip_")
    ]
    assert "functional centres" in new["provenance"]
    with pytest.raises(ValueError):
        module.apply_hip_calibration(spec, cal, np.eye(3))


def _synthetic_thigh_capture(
    twist_r_deg: float = 18.5,
    twist_l_deg: float = -12.3,
    static_frames: int = 30,
    dynamic_frames: int = 80,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(7)
    labels = tuple(WAIST) + ("RKneeOut", "RKneeIn", "LKneeOut", "LKneeIn")
    total_frames = static_frames + dynamic_frames
    points = np.zeros((total_frames, len(labels), 3))
    valid = np.ones((total_frames, len(labels)), dtype=bool)

    rot_r = Rotation.from_rotvec([0.0, 0.0, np.radians(twist_r_deg)]).as_matrix()
    rot_l = Rotation.from_rotvec([0.0, 0.0, np.radians(twist_l_deg)]).as_matrix()

    knee_offsets_r = {
        "RKneeOut": rot_r @ np.array([0.0, -0.05, -0.40]),
        "RKneeIn": rot_r @ np.array([0.0, 0.05, -0.40]),
    }
    knee_offsets_l = {
        "LKneeOut": rot_l @ np.array([0.0, 0.05, -0.40]),
        "LKneeIn": rot_l @ np.array([0.0, -0.05, -0.40]),
    }

    # Static frames 0..static_frames
    for f in range(static_frames):
        pelvis_rot = Rotation.from_rotvec(rng.normal(0.0, 0.1, 3)).as_matrix()
        pelvis_pos = rng.normal(0.0, 0.2, 3)
        for k, label in enumerate(WAIST):
            points[f, k] = pelvis_rot @ np.array(WAIST[label]) + pelvis_pos
        for label, offset in knee_offsets_r.items():
            points[f, labels.index(label)] = (
                pelvis_rot @ (CENTRES["r"] + offset) + pelvis_pos
            )
        for label, offset in knee_offsets_l.items():
            points[f, labels.index(label)] = (
                pelvis_rot @ (CENTRES["l"] + offset) + pelvis_pos
            )

    # Dynamic frames static_frames..total_frames for functional hip calibration
    for f in range(static_frames, total_frames):
        pelvis_rot = Rotation.from_rotvec(rng.normal(0.0, 0.3, 3)).as_matrix()
        pelvis_pos = rng.normal(0.0, 0.5, 3)
        for k, label in enumerate(WAIST):
            points[f, k] = pelvis_rot @ np.array(WAIST[label]) + pelvis_pos
        for side, out_label, in_label in (
            ("r", "RKneeOut", "RKneeIn"),
            ("l", "LKneeOut", "LKneeIn"),
        ):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            k_out = CENTRES[side] + 0.40 * direction
            k_in = k_out + (
                np.array([0, 0.10, 0]) if side == "r" else np.array([0, -0.10, 0])
            )
            points[f, labels.index(out_label)] = pelvis_rot @ k_out + pelvis_pos
            points[f, labels.index(in_label)] = pelvis_rot @ k_in + pelvis_pos

    return points, valid, labels


def test_hip_rotation_zero_recovers_known_angles() -> None:
    expected_r = 18.5
    expected_l = -12.3
    points, valid, labels = _synthetic_thigh_capture(
        twist_r_deg=expected_r, twist_l_deg=expected_l
    )
    zero = module.hip_rotation_zero(points, valid, labels, WAIST)
    assert zero.offset_r_deg == pytest.approx(expected_r, abs=0.1)
    assert zero.offset_l_deg == pytest.approx(expected_l, abs=0.1)
    assert zero["r"] == pytest.approx(expected_r, abs=0.1)
    assert zero["l"] == pytest.approx(expected_l, abs=0.1)
    # Check tuple unpacking
    zr, zl = zero
    assert zr == pytest.approx(expected_r, abs=0.1)
    assert zl == pytest.approx(expected_l, abs=0.1)


def test_hip_rotation_zero_ankle_fallback() -> None:
    expected_r = 14.0
    expected_l = -10.0
    rng = np.random.default_rng(9)
    labels = tuple(WAIST) + ("RKneeOut", "RAnkleOut", "LKneeOut", "LAnkleOut")
    static_frames, dynamic_frames = 25, 60
    total_frames = static_frames + dynamic_frames
    points = np.zeros((total_frames, len(labels), 3))
    valid = np.ones((total_frames, len(labels)), dtype=bool)

    rot_r = Rotation.from_rotvec([0.0, 0.0, np.radians(expected_r)]).as_matrix()
    rot_l = Rotation.from_rotvec([0.0, 0.0, np.radians(expected_l)]).as_matrix()

    # Knee flexed 20 deg posteriorly (shank bends backward)
    flex_r = Rotation.from_rotvec([0.0, np.radians(20.0), 0.0]).as_matrix()
    flex_l = Rotation.from_rotvec([0.0, np.radians(20.0), 0.0]).as_matrix()

    thigh_offset = np.array([0.0, 0.0, -0.40])
    shank_offset = np.array([0.0, 0.0, -0.40])

    k_offset_r = rot_r @ thigh_offset
    a_offset_r = k_offset_r + rot_r @ flex_r @ shank_offset
    k_offset_l = rot_l @ thigh_offset
    a_offset_l = k_offset_l + rot_l @ flex_l @ shank_offset

    for f in range(static_frames):
        pelvis_rot = Rotation.from_rotvec(rng.normal(0.0, 0.1, 3)).as_matrix()
        pelvis_pos = rng.normal(0.0, 0.2, 3)
        for k, label in enumerate(WAIST):
            points[f, k] = pelvis_rot @ np.array(WAIST[label]) + pelvis_pos
        points[f, labels.index("RKneeOut")] = (
            pelvis_rot @ (CENTRES["r"] + k_offset_r) + pelvis_pos
        )
        points[f, labels.index("RAnkleOut")] = (
            pelvis_rot @ (CENTRES["r"] + a_offset_r) + pelvis_pos
        )
        points[f, labels.index("LKneeOut")] = (
            pelvis_rot @ (CENTRES["l"] + k_offset_l) + pelvis_pos
        )
        points[f, labels.index("LAnkleOut")] = (
            pelvis_rot @ (CENTRES["l"] + a_offset_l) + pelvis_pos
        )

    for f in range(static_frames, total_frames):
        pelvis_rot = Rotation.from_rotvec(rng.normal(0.0, 0.3, 3)).as_matrix()
        pelvis_pos = rng.normal(0.0, 0.5, 3)
        for k, label in enumerate(WAIST):
            points[f, k] = pelvis_rot @ np.array(WAIST[label]) + pelvis_pos
        for side, k_lbl, a_lbl in (
            ("r", "RKneeOut", "RAnkleOut"),
            ("l", "LKneeOut", "LAnkleOut"),
        ):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            k_pos = CENTRES[side] + 0.40 * d
            points[f, labels.index(k_lbl)] = pelvis_rot @ k_pos + pelvis_pos
            points[f, labels.index(a_lbl)] = (
                pelvis_rot @ (k_pos + np.array([0, 0, -0.40])) + pelvis_pos
            )

    zero = module.hip_rotation_zero(points, valid, labels, WAIST)
    assert zero.offset_r_deg == pytest.approx(expected_r, abs=0.2)
    assert zero.offset_l_deg == pytest.approx(expected_l, abs=0.2)


def test_hip_rotation_zero_input_validation() -> None:
    points, valid, labels = _synthetic_thigh_capture()
    with pytest.raises(ValueError, match="Points must be"):
        module.hip_rotation_zero(points[0], valid, labels, WAIST)
    with pytest.raises(ValueError, match="waist markers"):
        module.hip_rotation_zero(points, valid, labels, {"WaistLeft": (0, 0, 0)})
    with pytest.raises(ValueError, match="knee_out_labels"):
        module.hip_rotation_zero(
            points, valid, labels, WAIST, knee_out_labels=("UnknownR", "UnknownL")
        )
    with pytest.raises(ValueError, match="medial knee markers or lateral ankle"):
        module.hip_rotation_zero(
            points,
            valid,
            ("WaistLeft", "WaistRight", "WaistLBack", "RKneeOut", "LKneeOut"),
            WAIST,
        )


def test_apply_hip_calibration_records_zero_twist() -> None:
    spec = json.loads(SPEC.read_text())
    receipt = json.loads(RECEIPT.read_text())
    points, valid, labels = _synthetic_capture()
    cal = module.functional_hip_calibration(points, valid, labels, WAIST)
    zero_twist = module.HipRotationZero(offset_r_deg=15.2, offset_l_deg=-8.7)
    new = module.apply_hip_calibration(
        spec,
        cal,
        receipt["pelvis_alignment"]["hip_from_opensim_pelvis"],
        zero_twist_deg=zero_twist,
    )
    hip_frame = np.array(
        next(f for f in spec["frames"] if f["name"] == "Hip")["placement"]
    )
    for side in ("r", "l"):
        joint = next(j for j in new["joints"] if j["name"] == f"hip_{side}")
        translation = np.array(joint["parent_to_base"])[:3, 3]
        expected = (hip_frame @ np.append(CENTRES[side], 1.0))[:3]
        np.testing.assert_allclose(translation, expected, atol=1e-9)
        rotation = np.array(joint["parent_to_base"])[:3, :3]
        assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert new["subject"]["hip_zero_twist_deg"] == {"r": 15.2, "l": -8.7}
    assert "zero twist" in new["provenance"]
