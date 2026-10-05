"""Behavioural tests for ZTCF-anchored, control-bounded prediction (DIME-04, #11425).

Tests the zero-torque counterfactual (ZTCF) drift acceleration anchor,
reachable control acceleration bounds, Gaussian control marginalization,
drift dominance index, fail-closed contact refusal, and runtime factor exclusivity.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.dime_contracts import (
    DimeCompleteState,
    EstimationIntervalFactor,
    RuntimeExclusivityContract,
)
from src.shared.python.estimation.dime_manifest import NumericAcceptanceThresholds
from src.shared.python.estimation.drift_prediction import (
    FIXED_BASE,
    ControlBand,
    DimeTransitionRequest,
    drift_dominance_index,
    integrate_step,
    linearize_drift,
    predict_dime_transition,
    predict_step,
    PropagatedStateCovariance,
    reachable_acceleration_interval,
    transition_jacobian,
    uncertain_control_prediction,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

pytestmark = pytest.mark.unit


class _LinearProvider:
    """M a + K q + D v = tau with constant M -- closed-form linear dynamics."""

    def __init__(self) -> None:
        self.M = np.array([[2.0, 0.3], [0.3, 1.0]])
        self.K = np.array([[4.0, -1.0], [-1.0, 3.0]])
        self.D = np.array([[0.2, 0.0], [0.0, 0.1]])

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return self.M.copy()

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.K @ np.asarray(q) + self.D @ np.asarray(v)


@pytest.fixture
def linear() -> _LinearProvider:
    return _LinearProvider()


@pytest.fixture
def pendulum():  # type: ignore[no-untyped-def]
    return make_backend("ode", GolfModelParams.default())


Q = np.array([0.4, -0.2])
V = np.array([3.0, -6.0])


class TestControlBand:
    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="lower"):
            ControlBand(lower=np.array([1.0, 0.0]), upper=np.array([0.0, 1.0]))

    def test_rejects_non_finite_bounds(self) -> None:
        with pytest.raises(ValueError):
            ControlBand(lower=np.array([-np.inf, 0.0]), upper=np.array([1.0, 1.0]))

    def test_local_band_intersects_rate_limit_with_global_box(self) -> None:
        band = ControlBand(
            lower=np.array([-10.0, -5.0]),
            upper=np.array([10.0, 5.0]),
            rate_limit=np.array([100.0, 100.0]),
        )
        lo, hi = band.local(np.array([9.5, 0.0]), dt=0.01)
        np.testing.assert_allclose(lo, [8.5, -1.0])
        np.testing.assert_allclose(hi, [10.0, 1.0])

    def test_local_band_without_rate_limit_is_global_box(self) -> None:
        band = ControlBand(lower=-np.ones(2), upper=np.ones(2))
        lo, hi = band.local(np.zeros(2), dt=0.01)
        np.testing.assert_array_equal(lo, -np.ones(2))
        np.testing.assert_array_equal(hi, np.ones(2))

    def test_previous_torque_outside_box_is_clipped_not_rejected(self) -> None:
        band = ControlBand(
            lower=-np.ones(2), upper=np.ones(2), rate_limit=np.full(2, 10.0)
        )
        lo, hi = band.local(np.array([5.0, 0.0]), dt=0.01)
        assert np.all(lo <= hi)
        assert hi[0] == pytest.approx(1.0)


class TestLinearizeDrift:
    def test_drift_matches_existing_ztcf_operator(
        self, linear: _LinearProvider
    ) -> None:
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        expected = np.linalg.solve(linear.M, -(linear.K @ Q + linear.D @ V))
        np.testing.assert_allclose(lin.drift_acceleration, expected, atol=1e-12)

    def test_control_influence_is_inverse_mass_times_selection(
        self, linear: _LinearProvider
    ) -> None:
        sel = np.array([[1.0], [0.0]])  # only the first joint is actuated
        lin = linearize_drift(linear, Q, V, selection=sel, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        np.testing.assert_allclose(
            lin.control_influence, np.linalg.solve(linear.M, sel), atol=1e-12
        )

    def test_total_acceleration_is_drift_plus_control(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        tau = np.array([25.0, -4.0])
        lin = linearize_drift(pendulum, Q, V, root_policy=FIXED_BASE)
        np.testing.assert_allclose(
            lin.acceleration(tau), pendulum.forward_dynamics(Q, V, tau), atol=1e-9
        )

    def test_rejects_selection_with_wrong_row_count(
        self, linear: _LinearProvider
    ) -> None:
        with pytest.raises(ValueError, match="selection"):
            linearize_drift(linear, Q, V, selection=np.eye(3), root_policy=FIXED_BASE)  # type: ignore[arg-type]

    def test_refuses_active_contact_with_receipt(self, linear: _LinearProvider) -> None:
        with pytest.raises(ValueError, match="contact"):
            linearize_drift(linear, Q, V, contact_active=True)  # type: ignore[arg-type]


class TestReachableAcceleration:
    def test_interval_is_exact_for_box_controls(self, linear: _LinearProvider) -> None:
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        lo_u, hi_u = np.array([-2.0, -1.0]), np.array([3.0, 1.0])
        lo_a, hi_a = reachable_acceleration_interval(lin, lo_u, hi_u)
        corners = [
            lin.acceleration(np.array([a, b])) for a in (-2.0, 3.0) for b in (-1.0, 1.0)
        ]
        np.testing.assert_allclose(lo_a, np.min(corners, axis=0), atol=1e-12)
        np.testing.assert_allclose(hi_a, np.max(corners, axis=0), atol=1e-12)

    def test_zero_width_band_collapses_to_constant_torque_prediction(
        self, pendulum
    ) -> None:  # type: ignore[no-untyped-def]
        tau = np.array([10.0, 2.0])
        lin = linearize_drift(pendulum, Q, V, root_policy=FIXED_BASE)
        lo_a, hi_a = reachable_acceleration_interval(lin, tau, tau)
        np.testing.assert_allclose(lo_a, hi_a, atol=1e-12)
        np.testing.assert_allclose(lo_a, lin.acceleration(tau), atol=1e-12)

    def test_drift_is_inside_band_only_when_zero_torque_is_admissible(
        self, pendulum
    ) -> None:  # type: ignore[no-untyped-def]
        lin = linearize_drift(pendulum, Q, V, root_policy=FIXED_BASE)
        a0 = lin.drift_acceleration
        lo_a, hi_a = reachable_acceleration_interval(lin, -np.ones(2), np.ones(2))
        assert np.all(lo_a <= a0) and np.all(a0 <= hi_a)
        lo_a, hi_a = reachable_acceleration_interval(
            lin, np.array([50.0, 50.0]), np.array([60.0, 60.0])
        )
        assert not (np.all(lo_a <= a0) and np.all(a0 <= hi_a))


class TestPredictStep:
    def test_constant_acceleration_kinematics(self, linear: _LinearProvider) -> None:
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        tau = np.array([1.0, -0.5])
        dt = 0.002
        q1, v1 = predict_step(lin, Q, V, tau, dt)
        a = lin.acceleration(tau)
        np.testing.assert_allclose(q1, Q + dt * V + 0.5 * dt**2 * a)
        np.testing.assert_allclose(v1, V + dt * a)

    def test_rejects_quaternion_configuration(self, linear: _LinearProvider) -> None:
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="manifold"):
            predict_step(lin, np.r_[Q, 1.0], V, np.zeros(2), 0.01)

    def test_rejects_non_positive_dt(self, linear: _LinearProvider) -> None:
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="dt"):
            predict_step(lin, Q, V, np.zeros(2), 0.0)


class TestUncertainControlPrediction:
    def test_marginalised_covariance_equals_explicit_gaussian_elimination(
        self, linear: _LinearProvider
    ) -> None:
        """Linear map of a Gaussian control: Cov = G Sigma_u G^T exactly."""
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        dt = 0.01
        mu = np.array([0.5, -0.2])
        sigma_u = np.array([[4.0, 1.0], [1.0, 2.0]])
        pred = uncertain_control_prediction(lin, Q, V, mu, sigma_u, dt)
        b = lin.control_influence
        g = np.vstack([0.5 * dt**2 * b, dt * b])
        np.testing.assert_allclose(pred.covariance, g @ sigma_u @ g.T, atol=1e-15)
        q1, v1 = predict_step(lin, Q, V, mu, dt)
        np.testing.assert_allclose(pred.mean, np.r_[q1, v1])

    def test_monte_carlo_agrees_with_closed_form(self, linear: _LinearProvider) -> None:
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        dt = 0.01
        mu = np.zeros(2)
        sigma_u = np.diag([9.0, 1.0])
        pred = uncertain_control_prediction(lin, Q, V, mu, sigma_u, dt)
        rng = np.random.default_rng(7)
        draws = rng.multivariate_normal(mu, sigma_u, size=20000)
        samples = np.array([np.r_[predict_step(lin, Q, V, u, dt)] for u in draws])
        np.testing.assert_allclose(
            np.cov(samples.T), pred.covariance, rtol=0.05, atol=1e-12
        )

    def test_state_covariance_is_propagated_and_result_is_psd(
        self, linear: _LinearProvider
    ) -> None:
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        state_cov = np.diag([1e-4, 1e-4, 1e-2, 1e-2])
        f = transition_jacobian(
            linear,  # type: ignore[arg-type]
            Q,
            V,
            np.zeros(2),
            0.01,
            root_policy=FIXED_BASE,
        )
        pred = uncertain_control_prediction(
            lin,
            Q,
            V,
            np.zeros(2),
            np.eye(2),
            0.01,
            state_prior=PropagatedStateCovariance(state_cov, f),
        )
        assert np.all(np.linalg.eigvalsh(pred.covariance) >= -1e-15)
        b = lin.control_influence
        g = np.vstack([0.5 * 0.01**2 * b, 0.01 * b])
        np.testing.assert_allclose(
            pred.covariance, f @ state_cov @ f.T + g @ g.T, rtol=1e-12, atol=1e-18
        )
        assert pred.covariance[0, 0] > state_cov[0, 0]

    def test_rejects_non_psd_control_covariance(self, linear: _LinearProvider) -> None:
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive semi-definite"):
            uncertain_control_prediction(
                lin, Q, V, np.zeros(2), np.diag([1.0, -1.0]), 0.01
            )

    def test_zero_mean_control_is_declared_not_asserted(
        self, linear: _LinearProvider
    ) -> None:
        """Zero mean with broad covariance must keep the band wide (not passive)."""
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        narrow = uncertain_control_prediction(
            lin, Q, V, np.zeros(2), 1e-6 * np.eye(2), 0.01
        )
        broad = uncertain_control_prediction(
            lin, Q, V, np.zeros(2), 100.0 * np.eye(2), 0.01
        )
        assert np.trace(broad.covariance) > 1e6 * np.trace(narrow.covariance)

    def test_model_uncertainty_is_incorporated(self, linear: _LinearProvider) -> None:
        lin = linearize_drift(linear, Q, V, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        model_q = np.diag([1e-6, 1e-6, 1e-5, 1e-5])
        pred = uncertain_control_prediction(
            lin, Q, V, np.zeros(2), np.eye(2), 0.01, model_uncertainty=model_q
        )
        assert pred.covariance[0, 0] >= 1e-6


class TestDriftDominance:
    def test_index_is_one_when_controls_have_no_authority(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        lin = linearize_drift(pendulum, Q, V, root_policy=FIXED_BASE)
        assert drift_dominance_index(lin, np.zeros(2), np.zeros(2)) == pytest.approx(
            1.0
        )

    def test_index_grows_with_velocity(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        band = (np.full(2, -20.0), np.full(2, 20.0))
        slow = drift_dominance_index(
            linearize_drift(pendulum, Q, 0.1 * V, root_policy=FIXED_BASE), *band
        )
        fast = drift_dominance_index(
            linearize_drift(pendulum, Q, 5.0 * V, root_policy=FIXED_BASE), *band
        )
        assert 0.0 <= slow < fast <= 1.0

    def test_index_is_not_fooled_by_cancelling_total_acceleration(
        self, pendulum
    ) -> None:  # type: ignore[no-untyped-def]
        """A strong control that cancels the drift leaves total accel ~ 0.

        The index compares drift with control authority, not realised total acceleration,
        so cancelling control stays bounded and informative in (0, 1).
        """
        lin = linearize_drift(pendulum, Q, 5.0 * V, root_policy=FIXED_BASE)
        cancelling = np.linalg.lstsq(
            lin.control_influence, -lin.drift_acceleration, rcond=None
        )[0]
        assert np.linalg.norm(lin.acceleration(cancelling)) < 1e-9
        index = drift_dominance_index(lin, -np.abs(cancelling), np.abs(cancelling))
        assert 0.0 < index < 1.0


class TestIntegrateStep:
    def test_matches_reference_backend_rk4_rollout(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        from src.shared.python.simulation_backends.protocol import SimState

        tau = np.array([30.0, -5.0])
        dt = 0.002
        pendulum.reset(SimState(q=Q.copy(), v=V.copy(), time=0.0))
        trace = pendulum.rollout(tau[None, :], horizon=1, dt=dt)
        q1, v1 = integrate_step(pendulum, Q, V, tau, dt, root_policy=FIXED_BASE)
        np.testing.assert_allclose(q1, trace.q[1], atol=1e-10)
        np.testing.assert_allclose(v1, trace.v[1], atol=1e-10)

    def test_zero_torque_step_follows_ztcf_drift(self, linear: _LinearProvider) -> None:
        dt = 1e-4
        q1, v1 = integrate_step(linear, Q, V, np.zeros(2), dt, root_policy=FIXED_BASE)  # type: ignore[arg-type]
        a0 = linearize_drift(linear, Q, V, root_policy=FIXED_BASE).drift_acceleration  # type: ignore[arg-type]
        np.testing.assert_allclose((v1 - V) / dt, a0, rtol=1e-3)

    def test_refuses_contact(self, linear: _LinearProvider) -> None:
        with pytest.raises(ValueError, match="contact"):
            integrate_step(
                linear,
                Q,
                V,
                np.zeros(2),
                0.01,
                contact_active=True,  # type: ignore[arg-type]
            )


class TestDimeTransitionPrediction:
    """RED/GREEN behavioral cases for DIME-04 transition criterion & contracts."""

    def test_torque_biased_initialization_does_not_corrupt_ztcf_branch(
        self, linear: _LinearProvider
    ) -> None:
        """RED/GREEN: torque-biased mean cannot alter native zero-control anchor."""
        state = DimeCompleteState(t=0.0, q=Q, v=V, model_hash="linear_model")
        req_zero = DimeTransitionRequest(
            state=state,
            root_policy=FIXED_BASE,
            control_mean=np.zeros(2),
            control_covariance=np.eye(2),
            dt=0.01,
        )
        req_biased = DimeTransitionRequest(
            state=state,
            root_policy=FIXED_BASE,
            control_mean=np.array([200.0, -150.0]),
            control_covariance=np.eye(2),
            dt=0.01,
        )
        res_zero = predict_dime_transition(linear, req_zero)  # type: ignore[arg-type]
        res_biased = predict_dime_transition(linear, req_biased)  # type: ignore[arg-type]

        assert res_zero.valid and res_biased.valid
        assert res_zero.zero_control_branch is not None
        assert res_biased.zero_control_branch is not None
        # ZTCF branch must be identical regardless of torque bias
        np.testing.assert_allclose(
            res_zero.zero_control_branch[0], res_biased.zero_control_branch[0]
        )
        np.testing.assert_allclose(
            res_zero.zero_control_branch[1], res_biased.zero_control_branch[1]
        )
        # Controlled prediction differs due to biased control mean
        assert res_zero.controlled_prediction is not None
        assert res_biased.controlled_prediction is not None
        assert not np.allclose(
            res_zero.controlled_prediction.mean, res_biased.controlled_prediction.mean
        )

    def test_high_drift_with_nonzero_control_dominates(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        """RED/GREEN: high drift dominance ratio with active control."""
        state = DimeCompleteState(t=0.0, q=Q, v=10.0 * V, model_hash="pendulum")
        req = DimeTransitionRequest(
            state=state,
            root_policy=FIXED_BASE,
            control_mean=np.array([10.0, -5.0]),
            control_covariance=4.0 * np.eye(2),
            dt=0.005,
        )
        res = predict_dime_transition(pendulum, req)
        assert res.valid
        assert res.drift_dominance is not None
        assert res.drift_dominance > 0.7

    def test_strong_opposing_control_near_zero_total_accel(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        """RED/GREEN: strong opposing control yielding near-zero total accel stays bounded."""
        lin = linearize_drift(pendulum, Q, V, root_policy=FIXED_BASE)
        cancelling = np.linalg.lstsq(
            lin.control_influence, -lin.drift_acceleration, rcond=None
        )[0]
        state = DimeCompleteState(t=0.0, q=Q, v=V, model_hash="pendulum")
        req = DimeTransitionRequest(
            state=state,
            root_policy=FIXED_BASE,
            control_mean=cancelling,
            control_covariance=np.eye(2),
            dt=0.005,
        )
        res = predict_dime_transition(pendulum, req)
        assert res.valid
        assert res.drift_dominance is not None
        assert 0.0 < res.drift_dominance < 1.0

    def test_contact_switch_disables_proposal_with_receipt(
        self, linear: _LinearProvider
    ) -> None:
        """RED/GREEN: contact_active=True disables proposal with fail-closed receipt."""
        state = DimeCompleteState(t=0.0, q=Q, v=V, model_hash="linear_model")
        req = DimeTransitionRequest(
            state=state,
            control_mean=np.zeros(2),
            control_covariance=np.eye(2),
            dt=0.01,
            contact_active=True,
        )
        res = predict_dime_transition(linear, req)  # type: ignore[arg-type]
        assert not res.valid
        assert res.receipt["code"] == "CONTACT_ACTIVE_REJECTED"
        assert res.zero_control_branch is None

    def test_invalid_horizon_or_dt_disables_proposal_with_receipt(
        self, linear: _LinearProvider
    ) -> None:
        """RED/GREEN: non-positive dt or horizon disables proposal with receipt."""
        state = DimeCompleteState(t=0.0, q=Q, v=V, model_hash="linear_model")
        req_zero_dt = DimeTransitionRequest(
            state=state,
            control_mean=np.zeros(2),
            control_covariance=np.eye(2),
            dt=0.0,
        )
        res = predict_dime_transition(linear, req_zero_dt)  # type: ignore[arg-type]
        assert not res.valid
        assert res.receipt["code"] == "INVALID_HORIZON"

    def test_exclusivity_contract_refuses_overlapping_duplicate_factors(
        self, linear: _LinearProvider
    ) -> None:
        """RED/GREEN: enforce mutual exclusion of duplicate physics likelihoods."""
        registry = RuntimeExclusivityContract()
        registry.register_factor(
            EstimationIntervalFactor(
                name="existing_explicit_likelihood",
                factor_type="explicit_input_likelihood",
                t_start=0.0,
                t_end=0.05,
                contributes_to_objective=True,
            )
        )
        state = DimeCompleteState(t=0.01, q=Q, v=V, model_hash="linear_model")
        req = DimeTransitionRequest(
            state=state,
            control_mean=np.zeros(2),
            control_covariance=np.eye(2),
            dt=0.01,
            mode="marginalized",
        )
        res = predict_dime_transition(
            linear,
            req,
            exclusivity_contract=registry,  # type: ignore[arg-type]
        )
        assert not res.valid
        assert res.receipt["code"] == "EXCLUSIVITY_VIOLATION"

    def test_drift_gain_against_frozen_baseline_thresholds(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        """GREEN: drift gain against frozen thresholds without tuning test data."""
        thresholds = NumericAcceptanceThresholds()
        lin = linearize_drift(pendulum, Q, V, root_policy=FIXED_BASE)
        assert lin.mass_condition_number < thresholds.condition_number_max
        assert np.linalg.norm(lin.drift_acceleration) > 0.0
