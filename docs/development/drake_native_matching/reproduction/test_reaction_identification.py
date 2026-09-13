"""Analytic reaction-elimination checks; no engine runtime is required."""

import numpy as np
import pytest
from reaction_identification import projected_system


def fixture() -> tuple:
    t = np.linspace(0, 1, 15)
    basis = np.polynomial.polynomial.polyvander(t, 6)
    effort = np.asarray([np.kron(np.eye(2), row) for row in basis])
    controls = np.array([1.0, -2.0, 3.0, 0.0, 0.0, 1.0, 0.5] + [0.0] * 7)
    torque = (effort @ controls)[:, 0]
    acceleration = np.column_stack((torque / 5, torque / 5))
    return (
        np.tile(np.diag([2.0, 3.0]), (len(t), 1, 1)),
        np.zeros((len(t), 2)),
        acceleration,
        np.tile([[1.0, -1.0]], (len(t), 1, 1)),
        np.zeros((len(t), 1)),
        effort,
        controls,
    )


def test_closed_weld_polynomial_recovers_only_identifiable_force_sum() -> None:
    mass, bias, qdd, jac, gamma, effort, true = fixture()
    matrix, rhs = projected_system(mass, bias, qdd, jac, gamma, effort)
    np.testing.assert_allclose(matrix @ true, rhs, atol=1e-13)
    recovered, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=1e-12)
    assert rank == 7  # 14 coefficients, seven reaction-indeterminate directions.
    np.testing.assert_allclose(matrix @ recovered, rhs, atol=1e-12)
    assert not np.allclose(recovered, true)


def test_weld_reactions_drop_out_exactly() -> None:
    mass, bias, qdd, jac, gamma, effort, _ = fixture()
    original = projected_system(mass, bias, qdd, jac, gamma, effort)
    reactions = np.arange(len(bias), dtype=float)
    altered = bias + np.einsum("ski,sk->si", jac, reactions[:, None])
    changed = projected_system(mass, altered, qdd, jac, gamma, effort)
    np.testing.assert_allclose(changed[0], original[0], atol=1e-13)
    np.testing.assert_allclose(changed[1], original[1], atol=1e-13)


def test_inconsistent_acceleration_is_rejected() -> None:
    mass, bias, qdd, jac, gamma, effort, _ = fixture()
    qdd[0, 0] += 1
    with pytest.raises(ValueError, match="closure"):
        projected_system(mass, bias, qdd, jac, gamma, effort)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_inputs_rejected(bad: float) -> None:
    mass, bias, qdd, jac, gamma, effort, _ = fixture()
    effort[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        projected_system(mass, bias, qdd, jac, gamma, effort)


def test_redundant_constraint_rejected() -> None:
    mass, bias, qdd, jac, gamma, effort, _ = fixture()
    with pytest.raises(ValueError, match="rank"):
        projected_system(
            mass,
            bias,
            qdd,
            np.zeros_like(jac),
            gamma,
            effort,
        )
