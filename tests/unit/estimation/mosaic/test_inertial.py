"""Tests for physically consistent inertial-parameter algebra (MOSAIC)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.inertial import (
    PLANAR_PARAMETERS_PER_BODY,
    SPATIAL_PARAMETERS_PER_BODY,
    from_log_cholesky,
    is_planar_consistent,
    is_spatial_consistent,
    planar_consistency_margin,
    project_planar_consistent,
    project_spatial_consistent,
    pseudo_inertia,
    to_log_cholesky,
)

pytestmark = pytest.mark.unit


def _spatial_from_box(mass: float, com: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Build a physically consistent 10-vector for a uniform box."""
    hx, hy, hz = half
    inertia_com = (mass / 3.0) * np.diag([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
    skew = np.array(
        [[0.0, -com[2], com[1]], [com[2], 0.0, -com[0]], [-com[1], com[0], 0.0]]
    )
    inertia_origin = inertia_com - mass * skew @ skew
    ixx, iyy, izz = np.diag(inertia_origin)
    ixy, ixz, iyz = inertia_origin[0, 1], inertia_origin[0, 2], inertia_origin[1, 2]
    return np.array([mass, *(mass * com), ixx, ixy, ixz, iyy, iyz, izz])


def test_parameter_counts() -> None:
    assert PLANAR_PARAMETERS_PER_BODY == 4
    assert SPATIAL_PARAMETERS_PER_BODY == 10


def test_pseudo_inertia_of_consistent_body_is_positive_definite() -> None:
    pi = _spatial_from_box(2.0, np.array([0.1, -0.05, 0.2]), np.array([0.2, 0.1, 0.3]))
    pseudo = pseudo_inertia(pi[None, :])[0]
    assert pseudo.shape == (4, 4)
    np.testing.assert_allclose(pseudo, pseudo.T)
    assert np.all(np.linalg.eigvalsh(pseudo) > 0.0)
    assert is_spatial_consistent(pi[None, :])[0]


def test_spatial_projection_is_identity_on_consistent_and_fixes_inconsistent() -> None:
    good = _spatial_from_box(1.5, np.array([0.0, 0.1, 0.0]), np.array([0.1, 0.2, 0.1]))
    bad = good.copy()
    bad[4] = -0.5  # negative Ixx is physically impossible
    batch = np.stack([good, bad])
    projected = project_spatial_consistent(batch)
    np.testing.assert_allclose(projected[0], good, atol=1e-10)
    assert is_spatial_consistent(projected)[1]
    # projection in pseudo-inertia Frobenius metric is idempotent
    np.testing.assert_allclose(
        project_spatial_consistent(projected), projected, atol=1e-10
    )


def test_log_cholesky_round_trip_and_consistency() -> None:
    pi = _spatial_from_box(
        3.0, np.array([0.05, 0.0, -0.1]), np.array([0.15, 0.1, 0.25])
    )
    theta = to_log_cholesky(pi[None, :])
    assert theta.shape == (1, SPATIAL_PARAMETERS_PER_BODY)
    back = from_log_cholesky(theta)
    np.testing.assert_allclose(back[0], pi, rtol=1e-9, atol=1e-9)
    # any theta maps to a consistent body
    rng = np.random.default_rng(0)
    arbitrary = rng.normal(size=(5, SPATIAL_PARAMETERS_PER_BODY))
    assert np.all(is_spatial_consistent(from_log_cholesky(arbitrary)))


def test_planar_consistency_margin_and_projection() -> None:
    # (m, h_x, h_y, I_o) with I_o >= |h|^2/m  <=>  m*I_o >= |h|^2
    consistent = np.array([[2.0, 0.4, 0.0, 0.5]])  # m I = 1.0 >= 0.16
    inconsistent = np.array([[2.0, 1.5, 0.0, 0.5]])  # m I = 1.0 <  2.25
    assert planar_consistency_margin(consistent)[0] > 0.0
    assert planar_consistency_margin(inconsistent)[0] < 0.0
    assert is_planar_consistent(consistent)[0]
    assert not is_planar_consistent(inconsistent)[0]
    projected = project_planar_consistent(inconsistent)
    assert is_planar_consistent(projected, tolerance=1e-9)[0]
    np.testing.assert_allclose(project_planar_consistent(consistent), consistent)
    # projection is the nearest point: moving further from it increases distance
    assert np.linalg.norm(projected - inconsistent) < np.linalg.norm(
        np.array([[2.0, 0.0, 0.0, 0.5]]) - inconsistent
    )


def test_pseudo_inertia_rejects_wrong_width() -> None:
    with pytest.raises(ContractViolationError):
        pseudo_inertia(np.zeros((1, 9)))
