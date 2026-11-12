"""NSX-98 (#12727): ``A(t)`` and ``B(t)`` assembly matches Track N blocks."""

from __future__ import annotations

import numpy as np
import pytest

from tests.integration.cross_engine.loop_nullspace_derivative_support import (
    NamedDerivatives,
    full_state_matrix,
    tangent_chain_operators,
    tangent_kinematics_coupling,
)

pytestmark = pytest.mark.unit


def _named(ops: dict[str, np.ndarray]) -> NamedDerivatives:
    nv = ops["d_acc_d_xi"].shape[0]
    return NamedDerivatives(
        ops,
        "finite_difference",
        np.eye(nv),
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
    full = full_state_matrix(_named(ops))
    assert full.shape == (k + nv, 2 * k)
    np.testing.assert_allclose(full[:k], tangent_kinematics_coupling(k))
    np.testing.assert_allclose(
        full[k:], np.hstack((ops["d_acc_d_xi"], ops["d_acc_d_eta"]))
    )


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
