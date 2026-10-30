"""Contract tests for wrist child_to_follower composition (HRG-HX1, #13758)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.anthropometric_geometry import (
    GRIP_ULNAR_OFFSET_DEG,
)
from src.shared.python.motion_matching.grip_contract import (
    NEUTRAL_EULER_BY_SIDE_DEG,
    follower_rotation,
    hand_from_euler,
    neutral_grip_matrix,
)
from src.shared.python.motion_matching.grip_fit import grip_rotation

pytestmark = pytest.mark.unit

HAND_BODY_SUFFIX = {"L": "Clubface Vector", "R": "RHandStandoff"}

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
NATIVE = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)


def _wrist_rotations(doc: dict) -> dict[str, np.ndarray]:
    return {
        side: np.asarray(joint["child_to_follower"], dtype=float)[:3, :3]
        for joint in doc["joints"]
        for side, suffix in HAND_BODY_SUFFIX.items()
        if joint["child"].endswith(suffix)
    }


def _geodesic_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    return float(
        np.degrees(np.linalg.norm(Rotation.from_matrix(rot_a.T @ rot_b).as_rotvec()))
    )


@pytest.mark.parametrize(
    "spec_name",
    ["full_body_spec_anthro_driver.json", "full_body_spec_anthro_iron7.json"],
)
def test_driver_wrist_child_to_follower_matches_grip_rotation_contract(
    spec_name: str,
) -> None:
    """Wrist child_to_follower equals N_side @ R_x(25 deg) @ G(e_side) within 0.5 deg."""
    native_rot = _wrist_rotations(json.loads(NATIVE.read_text(encoding="utf-8")))
    spec_doc = json.loads((MODELS / spec_name).read_text(encoding="utf-8"))
    committed_rot = _wrist_rotations(spec_doc)
    ulnar = np.asarray(
        Rotation.from_euler("x", GRIP_ULNAR_OFFSET_DEG, degrees=True).as_matrix()
    )
    for side in ("L", "R"):
        c2f_rot = committed_rot[side]
        assert np.allclose(c2f_rot @ c2f_rot.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(c2f_rot) == pytest.approx(1.0, abs=1e-9)
        expected_rot = (
            native_rot[side]
            @ ulnar
            @ grip_rotation(spec_doc["subject"]["grip_rotation_deg"][side])
        )
        assert _geodesic_angle_deg(c2f_rot, expected_rot) < 0.5


def test_recomposed_per_hand_neutral_c2f_matches_neutral_grip_matrix() -> None:
    """Re-composing c2f to NEUTRAL_EULER_BY_SIDE_DEG yields N_side @ M_{0,side}."""
    native_doc = json.loads(NATIVE.read_text(encoding="utf-8"))
    driver_doc = json.loads(
        (MODELS / "full_body_spec_anthro_driver.json").read_text(encoding="utf-8")
    )
    native_rot = _wrist_rotations(native_doc)
    committed_rot = _wrist_rotations(driver_doc)
    recorded = driver_doc["subject"]["grip_rotation_deg"]

    for side in ("L", "R"):
        target_euler = NEUTRAL_EULER_BY_SIDE_DEG[side]
        recomposed = (
            committed_rot[side]
            @ grip_rotation(recorded[side]).T
            @ grip_rotation(target_euler)
        )
        expected = native_rot[side] @ neutral_grip_matrix(side)
        via_contract = follower_rotation(
            native_rot[side], hand_from_euler(target_euler)
        )
        assert _geodesic_angle_deg(recomposed, expected) < 0.5
        assert np.allclose(recomposed, expected, atol=1e-9)
        assert np.allclose(via_contract, expected, atol=1e-9)
