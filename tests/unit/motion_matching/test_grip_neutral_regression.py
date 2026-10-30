"""HRG-HX1 (#13758) neutral-envelope regression tests for constant grip rotation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import grip_fit
from src.shared.python.motion_matching.grip_contract import (
    NEUTRAL_EULER_BY_SIDE_DEG,
    neutral_euler_deg,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs" / "development" / "full_body_models"
BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
RANGES = [(-90.0, 90.0), (-40.0, 25.0), (-70.0, 70.0)]
NEUTRAL_ENVELOPE_TOL_DEG = 3.0


def _geodesic_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = rot_a.T @ rot_b
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec())))


@pytest.mark.xfail(
    strict=True,
    reason="HRG-HX1 (#13758): committed specs re-set in follow-on PR via set_spec_grip_rotation.py.",
)
@pytest.mark.parametrize(
    "spec_name",
    ["full_body_spec_anthro_driver.json", "full_body_spec_anthro_iron7.json"],
)
def test_committed_grip_rotation_within_hrg_gr1_neutral_envelope(
    spec_name: str,
) -> None:
    """Committed grip_rotation_deg must sit within 3 deg of per-hand neutral M_0."""
    doc = json.loads((MODELS / spec_name).read_text(encoding="utf-8"))
    recorded = doc["subject"]["grip_rotation_deg"]
    for side in ("L", "R"):
        target = grip_fit.grip_rotation(neutral_euler_deg(side))
        actual = grip_fit.grip_rotation(recorded[side])
        assert _geodesic_angle_deg(actual, target) <= NEUTRAL_ENVELOPE_TOL_DEG, (
            spec_name,
            side,
            NEUTRAL_EULER_BY_SIDE_DEG[side],
        )


@pytest.mark.parametrize("side", ["L", "R"])
def test_regularized_fit_stays_within_three_degree_neutral_envelope_for_both_hands(
    side: str,
) -> None:
    """Positive neutral_prior_weight_deg2 keeps fitted grip within 3 deg of neutral."""
    rng = np.random.default_rng(13758)
    wrist_rad = rng.uniform(-0.25, 0.25, size=(24, 3))
    target_euler = neutral_euler_deg(side)
    target_rot = grip_fit.grip_rotation(target_euler)
    relative = grip_fit.wrist_rotation(wrist_rad, BASE) @ target_rot.T

    fit = grip_fit.fit_grip_rotation(
        relative,
        BASE,
        RANGES,
        grid_step_deg=30.0,
        neutral_prior_weight_deg2=1e-3,
        side=side,
    )
    assert fit.neutral_deviation_after_deg <= NEUTRAL_ENVELOPE_TOL_DEG
    fitted_rot = grip_fit.grip_rotation(fit.rotation_deg)
    assert _geodesic_angle_deg(fitted_rot, target_rot) <= NEUTRAL_ENVELOPE_TOL_DEG
