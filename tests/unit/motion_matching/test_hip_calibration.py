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
