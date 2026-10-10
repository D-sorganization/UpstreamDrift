"""The hip rewrite must use the pelvis alignment of the spec it rewrites (#12109).

``apply_hip_calibration`` recovers each hip joint frame in OpenSim pelvis
coordinates as ``X = A_old^-1 H^-1 P``. The pipeline used to take ``A_old``
from ``build_receipt_v2.json`` for every spec. The anthropometric specs are
built with a different alignment (``pelvis_alignment_for``), so ``X`` came out
as an arbitrary rotation and the calibrated hips started 38/69 degrees yawed
off the pelvis: ``hip_rotation_*`` then sat on its +-40 degree range limits at
address.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import hip_calibration as module
from src.shared.python.motion_matching.anthropometric_geometry import (
    pelvis_alignment_for,
)

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
ANTHRO_SPECS = ("full_body_spec_anthro_driver.json", "full_body_spec_anthro_iron7.json")


def _load(name: str) -> dict:
    return json.loads((MODELS / name).read_text(encoding="utf-8"))


def _hip_joint(spec: dict, side: str) -> dict:
    return next(j for j in spec["joints"] if j["name"] == f"hip_{side}")


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@pytest.mark.parametrize(
    ("spec_name", "receipt_name"),
    [
        ("full_body_spec_v1.json", "build_receipt.json"),
        ("full_body_spec_v2.json", "build_receipt_v2.json"),
    ],
)
def test_legacy_specs_reproduce_their_build_receipt(
    spec_name: str, receipt_name: str
) -> None:
    receipt = _load(receipt_name)["pelvis_alignment"]["hip_from_opensim_pelvis"]
    derived = module.pelvis_alignment_from_spec(_load(spec_name))
    np.testing.assert_allclose(derived[:3, :3], np.asarray(receipt)[:3, :3], atol=1e-6)


@pytest.mark.parametrize("spec_name", ANTHRO_SPECS)
def test_anthropometric_specs_use_their_builder_alignment(spec_name: str) -> None:
    derived = module.pelvis_alignment_from_spec(_load(spec_name))
    rotation, _ = pelvis_alignment_for(1.0)
    np.testing.assert_allclose(derived[:3, :3], rotation, atol=1e-9)


def _self_calibration(spec: dict) -> module.HipCalibration:
    """A calibration that reproduces the spec's own hips (centres and axes)."""
    hip = next(f for f in spec["frames"] if f["name"] == "Hip")
    h_inv = np.linalg.inv(np.asarray(hip["placement"], dtype=float))
    centres = {
        side: (h_inv @ np.asarray(_hip_joint(spec, side)["parent_to_base"]))[:3, 3]
        for side in ("r", "l")
    }
    axes = module.pelvis_alignment_from_spec(spec)[:3, :3]
    return module.HipCalibration(
        centre_r=tuple(float(v) for v in centres["r"]),
        centre_l=tuple(float(v) for v in centres["l"]),
        radius_r_m=0.4,
        radius_l_m=0.4,
        residual_sd_r_m=0.0,
        residual_sd_l_m=0.0,
        frames=1,
        pelvis_axes=tuple(tuple(float(v) for v in axes[:, k]) for k in range(3)),
        waist_fit_max_residual_m=0.0,
    )


@pytest.mark.parametrize("spec_name", ANTHRO_SPECS)
def test_self_calibration_leaves_the_hip_frames_unchanged(spec_name: str) -> None:
    spec = _load(spec_name)
    cal = _self_calibration(spec)
    new = module.apply_hip_calibration(
        spec, cal, module.pelvis_alignment_from_spec(spec)
    )
    for side in ("r", "l"):
        before = np.asarray(_hip_joint(spec, side)["parent_to_base"], dtype=float)
        after = np.asarray(_hip_joint(new, side)["parent_to_base"], dtype=float)
        np.testing.assert_allclose(after, before, atol=1e-9)


@pytest.mark.parametrize("spec_name", ANTHRO_SPECS)
def test_the_v2_receipt_alignment_rotates_anthropometric_hips(spec_name: str) -> None:
    """Documents the defect: the foreign alignment turns each hip frame."""
    spec = _load(spec_name)
    receipt = _load("build_receipt_v2.json")["pelvis_alignment"]
    new = module.apply_hip_calibration(
        spec, _self_calibration(spec), receipt["hip_from_opensim_pelvis"]
    )
    for side in ("r", "l"):
        before = np.asarray(_hip_joint(spec, side)["parent_to_base"])[:3, :3]
        after = np.asarray(_hip_joint(new, side)["parent_to_base"])[:3, :3]
        assert _angle_deg(before, after) > 20.0


def test_disagreeing_hips_fail_closed() -> None:
    spec = _load(ANTHRO_SPECS[0])
    joint = _hip_joint(spec, "l")
    c, s = np.cos(np.radians(10.0)), np.sin(np.radians(10.0))
    yaw = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    joint["parent_to_base"] = (yaw @ np.asarray(joint["parent_to_base"])).tolist()
    with pytest.raises(ValueError, match="disagree"):
        module.pelvis_alignment_from_spec(spec)


def test_missing_hip_frame_fails_closed() -> None:
    spec = _load(ANTHRO_SPECS[0])
    with pytest.raises(ValueError, match="frame"):
        module.pelvis_alignment_from_spec(spec, hip_frame="NoSuchFrame")
