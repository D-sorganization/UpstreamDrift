"""NSX-30 regression: the pendulum golfer delegates to the single KKT projector.

The reference is main's #12608 semantics, reproduced here from the same public
dynamics terms: the augmented KKT system on the independent closure rows only
(rows 2-3 rotate rows 0-1), solved by minimum-norm least squares because the
massless wrist and club DOFs leave it singular; redundant rows carry zero
multiplier.  The refactor onto the single projector cannot change any output.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pendulum_simulator import constraint_solver as cs
from src.shared.python.pendulum_simulator.counterfactual_golfer import (
    zero_torque_accelerations,
)
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams

pytestmark = pytest.mark.unit


def _params() -> GolferParams:
    return GolferParams(
        m_hub=0.01,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.3,
        L_hub=0.05,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=0.2,
        grip_left=0.25,
    )


def _torque(t: float) -> tuple:
    return tuple(0.5 * (i + 1) * np.cos(t + i) for i in range(7))


def _independent_row_reference(state, t, params, torque_func, alpha, beta):
    """Main's #12608 independent-row min-norm solve (reference)."""
    q, qdot = state[:N_DOF], state[N_DOF:]
    tau = np.zeros(N_DOF)
    tau[:7] = torque_func(t)
    mass = cs.mass_matrix(q, params)
    rhs_dyn = (
        tau
        + cs.friction_torque_vector(qdot, params)
        - cs.coriolis_matrix(q, qdot, params)
        - cs.gravity_vector(q, params)
    )
    phi = cs.constraint_vector(q, params)
    phi_q = cs.constraint_jacobian(q, params)
    gamma = cs._constraint_acceleration_bias(q, qdot, params)
    rhs_con = -gamma - 2.0 * alpha * (phi_q @ qdot) - beta**2 * phi
    rows = list(cs.INDEPENDENT_CLOSURE_ROWS)
    n, m = N_DOF, len(rows)
    kkt = np.zeros((n + m, n + m))
    kkt[:n, :n] = mass
    kkt[:n, n:] = phi_q[rows].T
    kkt[n:, :n] = phi_q[rows]
    rhs = np.concatenate([rhs_dyn, rhs_con[rows]])
    sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    lam = np.zeros(phi_q.shape[0])
    lam[rows] = sol[n:]
    return sol[:n], lam


@pytest.fixture(autouse=True)
def _python_path(monkeypatch):
    """Exercise the Python path, not an optional native kernel."""
    monkeypatch.setattr(
        cs._native_backend, "golfer_constrained_dynamics", lambda *a, **k: None
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("alpha", "beta"), [(5.0, 5.0), (0.0, 0.0), (12.0, 3.0)])
def test_accelerations_and_multipliers_match_the_independent_row_reference(
    seed, alpha, beta
) -> None:
    rng = np.random.default_rng(seed)
    state = np.concatenate([rng.uniform(-0.6, 0.6, N_DOF), rng.normal(0, 0.8, N_DOF)])
    params = _params()
    a_new, lam_new = cs._solve_constrained_dynamics(
        state, 0.3, params, _torque, alpha, beta
    )
    a_old, lam_old = _independent_row_reference(state, 0.3, params, _torque, alpha, beta)
    assert np.max(np.abs(a_new - a_old)) / max(1.0, np.max(np.abs(a_old))) < 1e-8
    assert np.max(np.abs(lam_new - lam_old)) / max(1.0, np.max(np.abs(lam_old))) < 1e-7


def test_zero_torque_counterfactual_is_unchanged() -> None:
    state = np.concatenate(
        [np.linspace(-0.3, 0.4, N_DOF), np.linspace(0.2, -0.5, N_DOF)]
    )
    params = _params()
    ref, _ = _independent_row_reference(state, 0.0, params, lambda t: (0.0,) * 7, 5.0, 5.0)
    out = zero_torque_accelerations(state, params)
    assert np.max(np.abs(out - ref)) / max(1.0, np.max(np.abs(ref))) < 1e-8


def test_solver_calls_the_single_kkt_implementation(monkeypatch) -> None:
    calls = []
    real = cs.constrained_acceleration

    def spy(*args, **kwargs):
        calls.append(kwargs.get("baumgarte"))
        return real(*args, **kwargs)

    monkeypatch.setattr(cs, "constrained_acceleration", spy)
    state = np.concatenate([np.linspace(-0.2, 0.3, N_DOF), np.zeros(N_DOF)])
    cs._solve_constrained_dynamics(state, 0.0, _params(), _torque)
    assert len(calls) == 1 and calls[0] is not None
