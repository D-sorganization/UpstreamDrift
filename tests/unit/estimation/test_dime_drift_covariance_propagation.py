"""Behavioural tests for drift-prediction propagation fixes (#11549).

* Covariance propagates through the one-step transition Jacobian
  ``F = df/dx``: ``P_{k+1} = F P_k F^T + G Sigma_u G^T + Q_w``.
* A horizon ``H > 1`` integrates ``H`` steps of ``dt`` (state and covariance),
  not one step of ``H * dt``.
* The default selection matrix never actuates the floating-base root DOFs;
  root actuation requires an explicit caller declaration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.estimation.dime_contracts import DimeCompleteState
from src.shared.python.estimation.drift_prediction import (
    DimeTransitionRequest,
    linearize_drift,
    predict_dime_transition,
    predict_step,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

pytestmark = pytest.mark.unit

FIXED_BASE: tuple[int, ...] = ()
Q = np.array([0.4, -0.2])
V = np.array([3.0, -6.0])


class _LinearProvider:
    """M a + K q + D v = tau with constant M: the step map is linear in x."""

    def __init__(self) -> None:
        self.M = np.array([[2.0, 0.3], [0.3, 1.0]])
        self.K = np.array([[4.0, -1.0], [-1.0, 3.0]])
        self.D = np.array([[0.2, 0.0], [0.0, 0.1]])

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return self.M.copy()

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.K @ np.asarray(q) + self.D @ np.asarray(v)


class _FloatingBaseProvider:
    """Six root DOFs (0..5) followed by two actuated joints, linear dynamics."""

    def __init__(self) -> None:
        rng = np.random.default_rng(11549)
        a = rng.normal(size=(8, 8))
        self.M = a @ a.T + 8.0 * np.eye(8)
        self.K = np.diag(np.linspace(1.0, 2.0, 8))

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return self.M.copy()

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.K @ np.asarray(q)


def _analytic_step_matrices(
    provider: _LinearProvider, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``A`` and ``G`` of the constant-acceleration step ``x+ = A x + G u``."""
    n = provider.M.shape[0]
    minv = np.linalg.inv(provider.M)
    eye = np.eye(n)
    da_dq = -minv @ provider.K
    da_dv = -minv @ provider.D
    a_mat = np.block(
        [
            [eye + 0.5 * dt**2 * da_dq, dt * eye + 0.5 * dt**2 * da_dv],
            [dt * da_dq, eye + dt * da_dv],
        ]
    )
    g_mat = np.vstack([0.5 * dt**2 * minv, dt * minv])
    return a_mat, g_mat


class TestCovariancePropagatesThroughTransitionJacobian:
    def test_linear_system_matches_analytic_recursion(self) -> None:
        """(a) P_{k+1} = A P_k A^T + G Sigma_u G^T + Q_w and x_{k+1} = A x_k + G mu."""
        provider = _LinearProvider()
        dt, horizon = 0.05, 4
        mu = np.array([0.5, -0.2])
        sigma_u = np.array([[4.0, 1.0], [1.0, 2.0]])
        p0 = np.diag([1e-3, 2e-3, 1e-2, 3e-2])
        q_w = np.diag([1e-6, 1e-6, 1e-5, 1e-5])
        request = DimeTransitionRequest(
            state=DimeCompleteState(t=0.0, q=Q, v=V, model_hash="linear"),
            control_mean=mu,
            control_covariance=sigma_u,
            dt=dt,
            horizon=horizon,
            state_covariance=p0,
            model_uncertainty=q_w,
            floating_base_root_dofs=FIXED_BASE,
        )
        result = predict_dime_transition(provider, request)  # type: ignore[arg-type]

        a_mat, g_mat = _analytic_step_matrices(provider, dt)
        x = np.r_[Q, V]
        p = p0.copy()
        for _ in range(horizon):
            x = a_mat @ x + g_mat @ mu
            p = a_mat @ p @ a_mat.T + g_mat @ sigma_u @ g_mat.T + q_w

        assert result.valid and result.controlled_prediction is not None
        np.testing.assert_allclose(result.controlled_prediction.mean, x, atol=1e-12)
        np.testing.assert_allclose(
            result.controlled_prediction.covariance, p, rtol=1e-7, atol=1e-12
        )

    def test_single_step_covariance_includes_dynamics_sensitivity(self) -> None:
        """Even for H = 1 the state covariance is mapped by df/dx, not kinematics."""
        provider = _LinearProvider()
        dt = 0.05
        p0 = np.diag([1e-3, 2e-3, 1e-2, 3e-2])
        request = DimeTransitionRequest(
            state=DimeCompleteState(t=0.0, q=Q, v=V, model_hash="linear"),
            control_mean=np.zeros(2),
            control_covariance=np.zeros((2, 2)),
            dt=dt,
            state_covariance=p0,
            floating_base_root_dofs=FIXED_BASE,
        )
        result = predict_dime_transition(provider, request)  # type: ignore[arg-type]
        a_mat, _ = _analytic_step_matrices(provider, dt)
        assert result.controlled_prediction is not None
        np.testing.assert_allclose(
            result.controlled_prediction.covariance,
            a_mat @ p0 @ a_mat.T,
            rtol=1e-7,
            atol=1e-12,
        )


class TestHorizonStepsEveryDt:
    def test_nonlinear_horizon_is_h_steps_of_dt(self) -> None:
        """(b) H steps of dt differ from one H*dt step and match an H-step reference."""
        pendulum: Any = make_backend("ode", GolfModelParams.default())
        dt, horizon = 0.01, 10
        mu = np.array([5.0, -2.0])
        request = DimeTransitionRequest(
            state=DimeCompleteState(t=0.0, q=Q, v=V, model_hash="pendulum"),
            control_mean=mu,
            control_covariance=np.eye(2),
            dt=dt,
            horizon=horizon,
            floating_base_root_dofs=FIXED_BASE,
        )
        result = predict_dime_transition(pendulum, request)

        def rollout(tau: np.ndarray) -> np.ndarray:
            q, v = Q.copy(), V.copy()
            for _ in range(horizon):
                lin = linearize_drift(pendulum, q, v, floating_base_root_dofs=())
                q, v = predict_step(lin, q, v, tau, dt)
            return np.r_[q, v]

        lin0 = linearize_drift(pendulum, Q, V, floating_base_root_dofs=())
        one_big_step = np.r_[predict_step(lin0, Q, V, np.zeros(2), horizon * dt)]

        assert result.valid and result.zero_control_branch is not None
        zero_branch = np.r_[result.zero_control_branch]
        np.testing.assert_allclose(zero_branch, rollout(np.zeros(2)), atol=1e-12)
        assert np.linalg.norm(zero_branch - one_big_step) > 1e-3
        assert result.controlled_prediction is not None
        np.testing.assert_allclose(
            result.controlled_prediction.mean, rollout(mu), atol=1e-12
        )


class TestRootIsNeverActuatedByDefault:
    def test_default_selection_zeroes_root_rows_and_columns(self) -> None:
        """(c) Default S = diag(0 x 6, 1, 1): torque never reaches the root."""
        provider = _FloatingBaseProvider()
        q, v = np.full(8, 0.1), np.full(8, -0.2)
        lin = linearize_drift(provider, q, v)  # type: ignore[arg-type]
        selection = provider.M @ lin.control_influence
        expected = np.diag([0.0] * 6 + [1.0, 1.0])
        np.testing.assert_allclose(selection, expected, atol=1e-12)

    def test_root_actuation_without_declaration_is_rejected(self) -> None:
        """(c) A selection that drives the root raises the contract error."""
        provider = _FloatingBaseProvider()
        q, v = np.zeros(8), np.zeros(8)
        with pytest.raises(PreconditionError, match="root"):
            linearize_drift(provider, q, v, selection=np.eye(8))  # type: ignore[arg-type]

    def test_declared_root_actuation_is_allowed(self) -> None:
        provider = _FloatingBaseProvider()
        q, v = np.zeros(8), np.zeros(8)
        lin = linearize_drift(
            provider,  # type: ignore[arg-type]
            q,
            v,
            selection=np.eye(8),
            allow_root_actuation=True,
        )
        np.testing.assert_allclose(
            provider.M @ lin.control_influence, np.eye(8), atol=1e-12
        )

    def test_undeclared_base_on_small_model_fails_closed(self) -> None:
        """A model too small for the default root must declare its base explicitly."""
        with pytest.raises(PreconditionError, match="floating_base_root_dofs"):
            linearize_drift(_LinearProvider(), Q, V)  # type: ignore[arg-type]
