"""Tests for MuJoCo humanoid golf Coriolis Jacobian utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.jacobian_utils import (
    compute_coriolis_matrix,
)

pytestmark = pytest.mark.unit


def test_compute_coriolis_matrix_uses_central_difference_accuracy() -> None:
    model = SimpleNamespace(nv=2, nq=2)
    qpos = np.array([0.25, -0.5])
    qvel = np.array([0.4, -0.3])

    def coriolis_forces(_qpos: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        return np.array(
            [
                velocity[0] ** 2 + 3.0 * velocity[1],
                np.sin(velocity[0]) + velocity[1] ** 3,
            ]
        )

    matrix = compute_coriolis_matrix(model, qpos, qvel, coriolis_forces)

    expected = np.array(
        [
            [2.0 * qvel[0], 3.0],
            [np.cos(qvel[0]), 3.0 * qvel[1] ** 2],
        ]
    )
    np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize(
    ("qpos", "qvel", "message"),
    [
        (np.array([0.0]), np.array([0.0, 0.0]), "qpos must have shape"),
        (np.array([0.0, np.nan]), np.array([0.0, 0.0]), "qpos must be finite"),
        (np.array([0.0, 0.0]), np.array([0.0]), "qvel must have shape"),
        (np.array([0.0, 0.0]), np.array([0.0, np.inf]), "qvel must be finite"),
    ],
)
def test_compute_coriolis_matrix_rejects_bad_state_vectors(
    qpos: np.ndarray,
    qvel: np.ndarray,
    message: str,
) -> None:
    model = SimpleNamespace(nv=2, nq=2)

    with pytest.raises(ValueError, match=message):
        compute_coriolis_matrix(
            model,
            qpos,
            qvel,
            lambda _qpos, _qvel: np.zeros(2),
        )


@pytest.mark.parametrize(
    ("forces", "message"),
    [
        (np.zeros((2, 1)), "compute_coriolis_fn must return shape"),
        (np.array([0.0, np.nan]), "compute_coriolis_fn must return finite"),
    ],
)
def test_compute_coriolis_matrix_rejects_bad_callback_output(
    forces: np.ndarray,
    message: str,
) -> None:
    model = SimpleNamespace(nv=2, nq=2)

    def coriolis_forces(_qpos: np.ndarray, _qvel: np.ndarray) -> np.ndarray:
        return forces

    with pytest.raises(ValueError, match=message):
        compute_coriolis_matrix(
            model,
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            coriolis_forces,
        )
