"""Drift-prediction horizon, covariance and root contracts (MOSAIC-13, #11545).

Analytic truth for a linear system ``M a + K q + D v = tau`` (constant ``M``):
the constant-acceleration step is exactly ``x_{k+1} = A x_k + G u`` with

    A = [[I + dt^2/2 A_q, dt I + dt^2/2 A_v], [dt A_q, I + dt A_v]],
    G = [dt^2/2 M^-1; dt M^-1],   A_q = -M^-1 K,   A_v = -M^-1 D,

so ``P_{k+1} = A P_k A^T + G Sigma_u G^T + Q_w``.  The defect recorded in the
reference (``sec:dime_review``) mapped ``P`` by the kinematic chain
``[[I, dt I], [0, I]]`` and took one step of ``H dt``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.dime_contracts import DimeCompleteState
from src.shared.python.estimation.drift_prediction import (
    FIXED_BASE,
    DimeTransitionRequest,
    predict_dime_transition,
)

pytestmark = pytest.mark.unit

Q0 = np.array([0.4, -0.2])
V0 = np.array([3.0, -6.0])
P0 = np.diag([1e-3, 2e-3, 1e-2, 3e-2])


class _LinearProvider:
    """``M a + K q + D v = tau`` with constant ``M``; the step map is linear."""

    M = np.array([[2.0, 0.3], [0.3, 1.0]])
    K = np.array([[4.0, -1.0], [-1.0, 3.0]])
    D = np.array([[0.2, 0.0], [0.0, 0.1]])

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return self.M.copy()

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.K @ np.asarray(q) + self.D @ np.asarray(v)


def _step_matrices(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``A`` (= ``df/dx``) and ``G`` of the linear system's step."""
    minv = np.linalg.inv(_LinearProvider.M)
    eye = np.eye(2)
    a_q = -minv @ _LinearProvider.K
    a_v = -minv @ _LinearProvider.D
    a_mat = np.block(
        [
            [eye + 0.5 * dt**2 * a_q, dt * eye + 0.5 * dt**2 * a_v],
            [dt * a_q, eye + dt * a_v],
        ]
    )
    return a_mat, np.vstack([0.5 * dt**2 * minv, dt * minv])


def _request(dt: float, horizon: object, **kwargs: object) -> DimeTransitionRequest:
    defaults: dict[str, object] = {
        "control_mean": np.zeros(2),
        "control_covariance": np.zeros((2, 2)),
        "state_covariance": P0,
    }
    defaults.update(kwargs)
    return DimeTransitionRequest(
        state=DimeCompleteState(t=0.0, q=Q0, v=V0, model_hash="linear"),
        dt=dt,
        horizon=horizon,  # type: ignore[arg-type]
        root_policy=FIXED_BASE,
        **defaults,  # type: ignore[arg-type]
    )


class TestCovarianceUsesStateJacobian:
    def test_covariance_differs_measurably_from_kinematic_chain(self) -> None:
        """With F != I the result is A P A^T, ~3.7 % away from the kinematic map."""
        dt = 0.1
        result = predict_dime_transition(_LinearProvider(), _request(dt, 1))  # type: ignore[arg-type]
        a_mat, _ = _step_matrices(dt)
        kinematic = np.block(
            [[np.eye(2), dt * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]]
        )
        correct = a_mat @ P0 @ a_mat.T
        buggy = kinematic @ P0 @ kinematic.T

        assert result.controlled_prediction is not None
        cov = result.controlled_prediction.covariance
        np.testing.assert_allclose(cov, correct, rtol=1e-7, atol=1e-12)
        gap = np.linalg.norm(correct - buggy) / np.linalg.norm(correct)
        assert gap > 1e-2
        assert np.linalg.norm(cov - buggy) / np.linalg.norm(correct) > 1e-2


class TestHorizonProducesEveryStep:
    def test_n_step_horizon_returns_n_distinct_analytic_predictions(self) -> None:
        """H = 5 yields 5 per-step predictions equal to the analytic recursion."""
        dt, horizon = 0.05, 5
        mu = np.array([0.5, -0.2])
        sigma_u = np.array([[4.0, 1.0], [1.0, 2.0]])
        q_w = np.diag([1e-6, 1e-6, 1e-5, 1e-5])
        result = predict_dime_transition(
            _LinearProvider(),  # type: ignore[arg-type]
            _request(
                dt,
                horizon,
                control_mean=mu,
                control_covariance=sigma_u,
                model_uncertainty=q_w,
            ),
        )
        a_mat, g_mat = _step_matrices(dt)

        assert result.valid
        trajectory = result.controlled_trajectory
        assert len(trajectory) == horizon
        x, p = np.r_[Q0, V0], P0.copy()
        for step in trajectory:
            x = a_mat @ x + g_mat @ mu
            p = a_mat @ p @ a_mat.T + g_mat @ sigma_u @ g_mat.T + q_w
            np.testing.assert_allclose(step.mean, x, atol=1e-12)
            np.testing.assert_allclose(step.covariance, p, rtol=1e-7, atol=1e-12)
        means = np.array([step.mean for step in trajectory])
        gaps = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=-1)
        assert np.all(gaps[~np.eye(horizon, dtype=bool)] > 1e-3)
        assert trajectory[-1] is result.controlled_prediction

    def test_every_step_covariance_is_symmetric_psd(self) -> None:
        result = predict_dime_transition(_LinearProvider(), _request(0.05, 4))  # type: ignore[arg-type]
        for step in result.controlled_trajectory:
            np.testing.assert_array_equal(step.covariance, step.covariance.T)
            assert np.min(np.linalg.eigvalsh(step.covariance)) >= -1e-15


class TestHorizonContract:
    @pytest.mark.parametrize("horizon", [0, -3, 2.5, True])
    def test_non_positive_or_non_integer_horizon_fails_closed(
        self, horizon: object
    ) -> None:
        """Horizon must be an integer >= 1; anything else is refused with a receipt."""
        result = predict_dime_transition(_LinearProvider(), _request(0.05, horizon))  # type: ignore[arg-type]
        assert not result.valid
        assert result.receipt["code"] == "INVALID_HORIZON"
        assert result.controlled_trajectory == ()
