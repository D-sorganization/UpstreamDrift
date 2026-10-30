"""SO(3) observability evidence for the constant grip fit (HRG-HX1, #13758)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import grip_fit

pytestmark = pytest.mark.unit

BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _measurement_residual(relative: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """Reconstruct forearm-to-club rotations after fitting wrist coordinates."""
    wrist = grip_fit.wrist_angles(relative @ grip.T, BASE)
    reconstructed = grip_fit.wrist_rotation(wrist, BASE) @ grip
    return (
        Rotation.from_matrix(reconstructed @ relative.transpose(0, 2, 1))
        .as_rotvec()
        .ravel()
    )


def test_multi_pose_measurement_residual_jacobian_is_rank_deficient() -> None:
    """Wrist DOFs absorb grip changes; this Jacobian cannot identify grip."""
    rng = np.random.default_rng(13758)
    wrist_angles = rng.uniform(-0.8, 0.8, size=(12, 3))
    true_grip = grip_fit.grip_rotation((17.0, -28.0, 43.0))
    relative = grip_fit.wrist_rotation(wrist_angles, BASE) @ true_grip

    step = 1e-6
    jacobian = np.column_stack(
        [
            (
                _measurement_residual(
                    relative,
                    Rotation.from_rotvec(np.eye(3)[axis] * step).as_matrix()
                    @ true_grip,
                )
                - _measurement_residual(
                    relative,
                    Rotation.from_rotvec(-np.eye(3)[axis] * step).as_matrix()
                    @ true_grip,
                )
            )
            / (2.0 * step)
            for axis in range(3)
        ]
    )

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    assert singular_values.shape == (3,)
    assert singular_values.max() < 1e-8
    assert np.linalg.matrix_rank(jacobian, tol=1e-8) == 0


@pytest.mark.parametrize(
    "bad_weight",
    [-1e-6, -1.0, float("nan"), float("inf"), True],
)
def test_fit_grip_rotation_rejects_negative_neutral_prior_weight(
    bad_weight: float,
) -> None:
    """DbC: neutral_prior_weight_deg2 must be a finite non-negative float."""
    relative = grip_fit.wrist_rotation(np.zeros((4, 3)), BASE)
    ranges = [(-90.0, 90.0), (-40.0, 25.0), (-70.0, 70.0)]
    with pytest.raises(ValueError, match="neutral_prior_weight_deg2"):
        grip_fit.fit_grip_rotation(
            relative,
            BASE,
            ranges,
            neutral_prior_weight_deg2=bad_weight,
        )


def test_fit_grip_rotation_rejects_invalid_neutral_target_or_side() -> None:
    relative = grip_fit.wrist_rotation(np.zeros((4, 3)), BASE)
    ranges = [(-90.0, 90.0), (-40.0, 25.0), (-70.0, 70.0)]
    with pytest.raises(ValueError, match="side"):
        grip_fit.fit_grip_rotation(relative, BASE, ranges, side="invalid")
    with pytest.raises(ValueError, match="neutral_rotation_deg"):
        grip_fit.fit_grip_rotation(relative, BASE, ranges, side=(0.0, 1.0))


@pytest.mark.parametrize("side", ["L", "R"])
def test_weak_geodesic_prior_recovers_per_hand_neutral_on_zero_rom_plateau(
    side: str,
) -> None:
    """When ROM excess has a flat zero-cost plateau, weak geodesic prior recovers neutral."""
    rng = np.random.default_rng(13758)
    wrist_rad = rng.uniform(-0.15, 0.15, size=(16, 3))
    ranges = [(-180.0, 180.0), (-85.0, 85.0), (-180.0, 180.0)]
    target_euler = (-75.0, 0.0, 0.0) if side == "L" else (75.0, 0.0, 180.0)
    target_rot = grip_fit.grip_rotation(target_euler)
    relative = grip_fit.wrist_rotation(wrist_rad, BASE) @ target_rot.T

    unregularized = grip_fit.fit_grip_rotation(
        relative, BASE, ranges, grid_step_deg=45.0, side=side
    )
    assert unregularized.rotation_deg == (0.0, 0.0, 0.0)
    assert unregularized.neutral_deviation_after_deg > 70.0

    regularized = grip_fit.fit_grip_rotation(
        relative,
        BASE,
        ranges,
        grid_step_deg=45.0,
        neutral_prior_weight_deg2=1e-4,
        side=side,
    )
    assert regularized.cost_after <= regularized.cost_before
    assert regularized.neutral_deviation_after_deg < 0.5
    assert np.allclose(
        grip_fit.grip_rotation(regularized.rotation_deg), target_rot, atol=1e-2
    )
