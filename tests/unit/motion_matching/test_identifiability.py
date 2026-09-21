"""Tests for motion-matching identifiability probe and null direction resolution (#10361, #9769)."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.identifiability import (
    IdentifiabilityResult,
    compute_spec_marker_positions,
    probe_spec_identifiability,
    probe_synthetic_chain_identifiability,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_SPEC_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "full_body_spec_anthro_driver.json"
)

# Standard anatomical calibration offsets for lower limbs (seeds)
LOWER_LIMB_CALIBRATED_OFFSETS = {
    "LKneeOut": ("femur_l", (0.05, 0.0, 0.0)),
    "RKneeOut": ("femur_r", (0.05, 0.0, 0.0)),
    "LAnkleOut": ("tibia_l", (0.04, 0.0, 0.0)),
    "RAnkleOut": ("tibia_r", (0.04, 0.0, 0.0)),
    "LToeIn": ("toes_l", (0.0, 0.08, 0.03)),
    "RToeIn": ("toes_r", (0.0, 0.08, 0.03)),
    "LToeOut": ("toes_l", (0.0, 0.08, -0.05)),
    "RToeOut": ("toes_r", (0.0, 0.08, -0.05)),
}


def test_synthetic_chain_detects_planted_null_direction() -> None:
    """TDD Step 3: Identifiability on a synthetic chain finds the planted null direction.

    Two co-axial revolute joints rotating about Z along the spine (similar to #9769:
    hip_rotation vs trunk_rotation). A marker placed along the common axis or on the
    distal segment cannot distinguish between the two rotations if they counter-rotate.
    """
    dof_names = ("hip_rotation", "trunk_rotation")

    # Forward kinematics: marker at offset [0.1, 0.0, 0.5] from base
    # Both joints rotate around Z: total angle is theta1 + theta2
    def observation_fn(q: np.ndarray) -> np.ndarray:
        theta = float(q[0] + q[1])
        c, s = np.cos(theta), np.sin(theta)
        # End marker rotating in X-Y plane around Z
        mx = 0.2 * c
        my = 0.2 * s
        mz = 0.5
        return np.array([mx, my, mz], dtype=np.float64)

    q0 = np.array([0.1, 0.2], dtype=np.float64)
    result = probe_synthetic_chain_identifiability(dof_names, observation_fn, q0)

    assert isinstance(result, IdentifiabilityResult)
    assert result.rank == 1  # 2 DOFs but only 1 observable singular value
    assert not result.is_full_rank
    assert result.condition_number > 1e6
    assert set(result.unobservable_dofs) == {"hip_rotation", "trunk_rotation"}
    assert len(result.nullspace_directions) >= 1

    # Verify the null direction has opposite signs for hip and trunk rotation [1, -1] / sqrt(2)
    null_dir = next(iter(result.nullspace_directions.values()))
    assert np.isclose(abs(null_dir[0]), abs(null_dir[1]), atol=1e-3)
    assert np.sign(null_dir[0]) != np.sign(null_dir[1])


def test_synthetic_chain_resolves_null_direction_via_prior() -> None:
    """Resolution of #9769 null direction via anthropometric prior regularization."""
    dof_names = ("hip_rotation", "trunk_rotation")

    def observation_fn(q: np.ndarray) -> np.ndarray:
        theta = float(q[0] + q[1])
        return np.array(
            [0.2 * np.cos(theta), 0.2 * np.sin(theta), 0.5], dtype=np.float64
        )

    q0 = np.array([0.1, 0.2], dtype=np.float64)
    # With prior_weight > 0, the null direction is regularized
    result = probe_synthetic_chain_identifiability(
        dof_names, observation_fn, q0, prior_weight=1.0
    )

    assert result.rank == 2
    assert result.is_full_rank
    assert result.condition_number < 100.0
    assert len(result.unobservable_dofs) == 0
    assert set(result.reliable_dofs) == {"hip_rotation", "trunk_rotation"}


def test_synthetic_chain_resolves_null_direction_via_off_axis_marker() -> None:
    """Resolution of #9769 null direction via adding an intermediate off-axis marker."""
    dof_names = ("hip_rotation", "trunk_rotation")

    # Intermediate marker sits on pelvis (rotates only by q[0])
    def observation_fn(q: np.ndarray) -> np.ndarray:
        m1 = np.array([0.15 * np.cos(q[0]), 0.15 * np.sin(q[0]), 0.2])
        m2 = np.array([0.2 * np.cos(q[0] + q[1]), 0.2 * np.sin(q[0] + q[1]), 0.5])
        return np.concatenate([m1, m2])

    q0 = np.array([0.1, 0.2], dtype=np.float64)
    result = probe_synthetic_chain_identifiability(dof_names, observation_fn, q0)

    assert result.rank == 2
    assert result.is_full_rank
    assert len(result.unobservable_dofs) == 0


@pytest.mark.skipif(not DRIVER_SPEC_PATH.is_file(), reason="driver spec not present")
def test_44_dof_spec_uncalibrated_detects_unobservable_lower_limbs() -> None:
    """Raw 44-DOF spec has null lower limb marker offsets; probe must flag all 14 leg DOFs."""
    spec = json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))
    assert len(spec["coordinate_order"]) == 44

    result = probe_spec_identifiability(spec)

    assert result.rank < 44
    assert not result.is_full_rank

    leg_dofs = {
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "knee_angle_r",
        "ankle_angle_r",
        "subtalar_angle_r",
        "mtp_angle_r",
        "hip_flexion_l",
        "hip_adduction_l",
        "hip_rotation_l",
        "knee_angle_l",
        "ankle_angle_l",
        "subtalar_angle_l",
        "mtp_angle_l",
    }
    # All 14 leg DOFs must be reported as unobservable because their markers have offset_m: null
    assert leg_dofs.issubset(set(result.unobservable_dofs))


@pytest.mark.skipif(not DRIVER_SPEC_PATH.is_file(), reason="driver spec not present")
def test_44_dof_spec_calibrated_makes_lower_limbs_observable() -> None:
    """Providing calibrated lower limb offsets makes the lower limb DOFs observable."""
    spec = json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))

    result = probe_spec_identifiability(
        spec, marker_offsets=LOWER_LIMB_CALIBRATED_OFFSETS
    )

    # Rank should increase significantly because all 8 leg markers are now observable
    assert result.rank >= 35

    # Knee and hip flexion should be observable with leg markers attached
    assert "knee_angle_r" not in result.unobservable_dofs
    assert "knee_angle_l" not in result.unobservable_dofs


@pytest.mark.skipif(not DRIVER_SPEC_PATH.is_file(), reason="driver spec not present")
def test_44_dof_spec_with_anthropometric_prior_achieves_full_rank() -> None:
    """Anthropometric prior resolves remaining null/weak directions to full rank 44."""
    spec = json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))

    result = probe_spec_identifiability(
        spec,
        marker_offsets=LOWER_LIMB_CALIBRATED_OFFSETS,
        prior_weight=0.05,
    )

    assert result.rank == 44
    assert result.is_full_rank
    assert len(result.unobservable_dofs) == 0
    assert np.isfinite(result.condition_number)
    assert result.condition_number < 1e5
