"""Local sensitivity diagnostics preserve units and expose rank truncation."""

import numpy as np
import pytest
from src.shared.python.motion_matching.sensitivity_diagnostics import (
    analyze_linearized_residual,
)

pytestmark = pytest.mark.unit


def test_scaled_rank_and_uncontrollable_residual() -> None:
    result = analyze_linearized_residual(
        np.array([[100.0, 0.0], [0.0, 0.0]]),
        np.array([2.0, 3.0]),
        np.array([0.01, 2.0]),
    )
    assert result.rank == 1
    np.testing.assert_allclose(result.singular_values, [1.0, 0.0])
    np.testing.assert_allclose(result.physical_step, [-0.02, 0.0])
    np.testing.assert_allclose(result.linear_residual, [0.0, 3.0])
    assert not result.physical_step.flags.writeable


def test_relative_cutoff_is_explicit() -> None:
    result = analyze_linearized_residual(
        np.diag([1.0, 1e-9]), np.ones(2), np.ones(2), relative_cutoff=1e-6
    )
    assert result.rank == 1
    np.testing.assert_allclose(result.linear_residual, [0.0, 1.0])


@pytest.mark.parametrize("scale", [[0.0, 1.0], [-1.0, 1.0], [1.0], [float("nan"), 1.0]])
def test_bad_physical_scales_rejected(scale) -> None:
    with pytest.raises(ValueError):
        analyze_linearized_residual(np.eye(2), np.ones(2), np.array(scale))


def test_zero_jacobian_preserves_residual() -> None:
    result = analyze_linearized_residual(np.zeros((2, 1)), [2.0, 3.0], [1.0])
    assert result.rank == 0
    np.testing.assert_allclose(result.linear_residual, [2.0, 3.0])
    np.testing.assert_allclose(result.physical_step, [0.0])


@pytest.mark.parametrize(
    "jac,res,cutoff",
    [
        (np.eye(2), [1.0], 1e-8),
        ([[float("nan")]], [1.0], 1e-8),
        (np.eye(2), [1.0, 1.0], 0.0),
    ],
)
def test_invalid_matrix_or_residual_rejected(jac, res, cutoff) -> None:
    with pytest.raises(ValueError):
        analyze_linearized_residual(
            jac, res, np.ones(np.asarray(jac).shape[1]), relative_cutoff=cutoff
        )
