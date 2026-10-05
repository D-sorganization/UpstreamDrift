"""Tests for the B-spline kinematic basis used by MOSAIC."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis

pytestmark = pytest.mark.unit


def test_basis_reproduces_polynomial_and_its_derivatives() -> None:
    times = np.linspace(0.0, 1.0, 41)
    basis = BSplineBasis.uniform(times, n_coefficients=12, degree=5)
    samples = np.stack([times**3 - 0.5 * times, 2.0 * times**2], axis=1)
    coefficients = basis.fit_least_squares(samples)
    assert coefficients.shape == (12, 2)
    np.testing.assert_allclose(basis.position @ coefficients, samples, atol=1e-9)
    expected_v = np.stack([3 * times**2 - 0.5, 4.0 * times], axis=1)
    expected_a = np.stack([6 * times, np.full_like(times, 4.0)], axis=1)
    np.testing.assert_allclose(basis.velocity @ coefficients, expected_v, atol=1e-7)
    np.testing.assert_allclose(basis.acceleration @ coefficients, expected_a, atol=1e-5)


def test_basis_matrices_have_expected_shapes_and_partition_of_unity() -> None:
    times = np.linspace(0.3, 1.7, 23)
    basis = BSplineBasis.uniform(times, n_coefficients=8, degree=3)
    assert basis.position.shape == (23, 8)
    assert basis.n_coefficients == 8
    np.testing.assert_allclose(basis.position.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(basis.velocity.sum(axis=1), 0.0, atol=1e-9)


def test_basis_rejects_bad_inputs() -> None:
    with pytest.raises(ContractViolationError):
        BSplineBasis.uniform(np.array([0.0, 0.0, 1.0]), n_coefficients=5, degree=3)
    with pytest.raises(ContractViolationError):
        BSplineBasis.uniform(np.linspace(0, 1, 10), n_coefficients=3, degree=3)
