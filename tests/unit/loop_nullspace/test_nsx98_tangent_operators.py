"""NSX-98 (#12727): ``A(t)`` and ``B(t)`` assembly matches Track N blocks."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace import tangent_linearization as tl
from tests.integration.cross_engine.loop_nullspace_derivative_support import (
    NamedDerivatives,
    full_state_matrix,
    tangent_chain_operators,
    tangent_kinematics_coupling,
)
from tests.unit.loop_nullspace._tangent_fixtures import FiveBar

pytestmark = pytest.mark.unit

STEP = 1e-5
MAN = tl.EuclideanManifold(4)


def _named(
    ops: dict[str, np.ndarray], *, basis: np.ndarray | None = None
) -> NamedDerivatives:
    nv = ops["d_acc_d_xi"].shape[0]
    k = ops["d_acc_d_xi"].shape[1]
    if basis is None:
        basis = np.eye(nv)[:, :k]
    return NamedDerivatives(
        ops,
        "finite_difference",
        np.eye(nv),
        basis,
    )


def test_tangent_chain_operators_stack_state_and_load_blocks() -> None:
    k, nv, m = 3, 5, 6
    ops = {
        "d_acc_d_xi": np.arange(nv * k, dtype=float).reshape(nv, k),
        "d_acc_d_eta": np.arange(100, 100 + nv * k, dtype=float).reshape(nv, k),
        "d_acc_d_tau": np.eye(nv) * 2.0,
        "d_lambda_d_xi": np.ones((m, k)) * 0.1,
        "d_lambda_d_eta": np.ones((m, k)) * 0.2,
        "d_lambda_d_tau": np.ones((m, nv)) * 0.3,
    }
    named = _named(ops)
    chain = tangent_chain_operators(named)
    np.testing.assert_allclose(chain.a_acc[:, :k], ops["d_acc_d_xi"])
    np.testing.assert_allclose(chain.a_acc[:, k:], ops["d_acc_d_eta"])
    np.testing.assert_allclose(chain.b_acc, ops["d_acc_d_tau"])
    np.testing.assert_allclose(chain.a_lambda[:, :k], ops["d_lambda_d_xi"])
    np.testing.assert_allclose(chain.a_lambda[:, k:], ops["d_lambda_d_eta"])


def test_full_state_matrix_includes_kinematic_coupling() -> None:
    k, nv = 2, 4
    ops = {
        "d_acc_d_xi": np.ones((nv, k)),
        "d_acc_d_eta": np.zeros((nv, k)),
        "d_acc_d_tau": np.eye(nv),
        "d_lambda_d_xi": np.zeros((1, k)),
        "d_lambda_d_eta": np.zeros((1, k)),
        "d_lambda_d_tau": np.zeros((1, nv)),
    }
    basis = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]).T
    full = full_state_matrix(_named(ops, basis=basis))
    assert full.shape == (2 * k, 2 * k)
    np.testing.assert_allclose(full[:k], tangent_kinematics_coupling(k))
    np.testing.assert_allclose(
        full[k:], basis.T @ np.hstack((ops["d_acc_d_xi"], ops["d_acc_d_eta"]))
    )


def test_full_state_matrix_matches_finite_difference_tangent_state_space() -> None:
    """``A(t)`` matches ``tangent_state_space`` at ``v = 0`` (no connection terms)."""
    bar = FiveBar()
    q, _, tau = bar.state()
    v = np.zeros(4)
    derivs = tl.FiniteDifferenceDerivatives(
        bar, MAN, step=STEP, basis_field=bar.frame
    ).constrained_acceleration_derivatives(q, v, tau)
    ss = tl.tangent_state_space(
        derivs, q=q, v=v, manifold=MAN, basis_field=bar.frame, step=STEP
    )
    ops = {
        "d_acc_d_xi": derivs.d_acc_d_xi,
        "d_acc_d_eta": derivs.d_acc_d_eta,
        "d_acc_d_tau": derivs.d_acc_d_tau,
        "d_lambda_d_xi": derivs.d_lambda_d_xi,
        "d_lambda_d_eta": derivs.d_lambda_d_eta,
        "d_lambda_d_tau": derivs.d_lambda_d_tau,
    }
    assembled = full_state_matrix(
        NamedDerivatives(ops, derivs.method.value, np.eye(4), derivs.basis)
    )
    assert assembled.shape == ss.F.shape == (4, 4)
    np.testing.assert_allclose(assembled, ss.F, atol=1e-6)


def test_tangent_chain_operators_refuse_mismatched_blocks() -> None:
    ops = {
        "d_acc_d_xi": np.zeros((4, 3)),
        "d_acc_d_eta": np.zeros((4, 2)),
        "d_acc_d_tau": np.eye(4),
        "d_lambda_d_xi": np.zeros((2, 3)),
        "d_lambda_d_eta": np.zeros((2, 3)),
        "d_lambda_d_tau": np.zeros((2, 4)),
    }
    with pytest.raises(ValueError, match="d_acc_d_xi"):
        tangent_chain_operators(_named(ops))
