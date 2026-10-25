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
