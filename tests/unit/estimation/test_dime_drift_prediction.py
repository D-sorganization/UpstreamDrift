"""Behavioral unit tests for DIME uncertain-control ZTCF prediction and estimation criterion (#11425).

Enforces:
- RED: torque-biased initialization, high drift with nonzero control, strong opposing control,
  near-zero total acceleration, contact switch and uncertain mass.
- GREEN: analytic linear-system marginalization equals explicit Gaussian-control elimination;
  covariance includes uncertain input and is PSD; nonlinear sampled reference bounds
  linearization error. Wrong torque mean cannot force observations onto passive motion.
  Test drift gain against the frozen baseline without tuning test data.
- Mutual exclusion of duplicate physics likelihoods via RuntimeExclusivityContract.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

from src.shared.python.contracts import ContractViolationError, PreconditionError
from src.shared.python.estimation.dime_manifest import CANONICAL_DIME_UNITS
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
    make_underactuated_analytic_fixture,
)
from src.shared.python.estimation.dime_contracts import (
    AnalyticPendulumProvider,
    ContactPolicy,
    ControlChannelSpec,
    DeterministicFakeProvider,
    DimeCompleteState,
    DimeFullStepRequest,
    EstimationIntervalFactor,
    ProviderCapability,
    RuntimeExclusivityContract,
    UnderactuatedAnalyticProvider,
    VectorSpaceManifold,
)
from src.shared.python.estimation.dime_drift_prediction import (
    ControlDistribution,
    DimeDriftPredictionResult,
    DimeDriftPredictor,
    DimeDriftTransitionCriterion,
    ModelContactUncertainty,
    PredictionMode,
    PredictionValidityStatus,
    StateDistribution,
)


def _make_pendulum_provider() -> tuple[AnalyticPendulumProvider, DimeCompleteState]:
    fixture = make_fixed_base_pendulum_fixture(n_frames=10, fps=100.0)
    provider = AnalyticPendulumProvider(fixture=fixture)
    state = DimeCompleteState(
        t=0.0,
        q=np.array([0.5], dtype=np.float64),
        v=np.array([0.0], dtype=np.float64),
        model_hash=provider.model_hash,
        units=dict(CANONICAL_DIME_UNITS),
    )
    return provider, state


class TestDimeDriftPredictionRedSuite:
    """RED test cases enforcing failure on invalid state, torque bias, and contact conditions."""

    def test_red_torque_biased_initialization(self) -> None:
        """Torque-biased initialization correctly influences prediction; wrong torque mean cannot force passive motion."""
        provider, state = _make_pendulum_provider()
        predictor = DimeDriftPredictor(provider)

        cov_x = np.diag([1e-4, 1e-4])
        state_dist = StateDistribution(mean=state, covariance=cov_x)
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
            max_qualified_horizon_s=0.5,
        )

        # 1. Forward step with true control torque = 2.0 N*m
        ctrl_dist_correct = ControlDistribution(
            mean=np.array([2.0], dtype=np.float64),
            covariance=np.array([[1e-3]], dtype=np.float64),
            physical_types=("torque",),
            units=("N*m",),
        )
        res_correct = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist_correct,
            uncertainty=uncertainty,
            horizon_s=0.05,
            dt=0.01,
            mode=PredictionMode.MARGINALIZED_CONTROL,
        )
        assert res_correct.validity_status == PredictionValidityStatus.VALID
        assert res_correct.predicted_mean is not None
        assert res_correct.zero_control_branch is not None

        # Compare with zero-control prior
        ctrl_dist_zero = ControlDistribution(
            mean=np.array([0.0], dtype=np.float64),
            covariance=np.array([[1e-3]], dtype=np.float64),
            physical_types=("torque",),
            units=("N*m",),
        )
        res_zero = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist_zero,
            uncertainty=uncertainty,
            horizon_s=0.05,
            dt=0.01,
            mode=PredictionMode.MARGINALIZED_CONTROL,
        )
        assert res_zero.predicted_mean is not None

        # With positive torque, predicted angle/velocity must be strictly greater than zero-torque branch
        assert res_correct.predicted_mean.v[0] > res_zero.predicted_mean.v[0]

        # 2. Wrong torque mean cannot force observations onto passive motion:
        # If observed candidate state matches the ZTCF passive drift,
        # evaluating transition criterion with false torque prior produces large residual.
        ztcf_next_state = res_zero.zero_control_branch.states[-1]
        criterion_true = DimeDriftTransitionCriterion(
            provider=provider,
            control_dist=ctrl_dist_zero,
            uncertainty=uncertainty,
            dt=0.05,
        )
        res_passive_zero = criterion_true.evaluate_residual(state, ztcf_next_state)

        criterion_false = DimeDriftTransitionCriterion(
            provider=provider,
            control_dist=ControlDistribution(
                mean=np.array([5.0], dtype=np.float64),
                covariance=np.array([[1e-3]], dtype=np.float64),
            ),
            uncertainty=uncertainty,
            dt=0.05,
        )
        res_passive_biased = criterion_false.evaluate_residual(state, ztcf_next_state)
        # Biased torque expectation on passive drift motion results in significantly higher residual norm
        assert (
            np.linalg.norm(res_passive_biased) > np.linalg.norm(res_passive_zero) + 1.0
        )

    def test_red_high_drift_with_nonzero_control(self) -> None:
        """High drift velocity combined with nonzero control truthfully decomposes components."""
        provider = UnderactuatedAnalyticProvider()
        predictor = DimeDriftPredictor(provider)

        high_drift_state = DimeCompleteState(
            t=0.0,
            q=np.array([1.2, -0.5], dtype=np.float64),
            v=np.array([4.0, -2.0], dtype=np.float64),  # High angular speed
            model_hash=provider.model_hash,
            units=dict(CANONICAL_DIME_UNITS),
        )
        state_dist = StateDistribution(
            mean=high_drift_state, covariance=np.diag([1e-4, 1e-4, 1e-4, 1e-4])
        )
        ctrl_dist = ControlDistribution(
            mean=np.array([1.5], dtype=np.float64),
            covariance=np.array([[1e-3]], dtype=np.float64),
            physical_types=("torque",),
            units=("N*m",),
        )
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5, 1e-5, 1e-5]),
        )

        res = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            horizon_s=0.02,
            dt=0.01,
            mode=PredictionMode.EXPLICIT_CONTROL,
        )
        assert res.validity_status == PredictionValidityStatus.VALID
        assert res.zero_control_branch is not None
        assert res.linearization_drift_jacobian_F is not None
        assert res.linearization_control_jacobian_G is not None

        # Drift component must be non-zero and separate from control
        decomp = res.zero_control_branch.decompositions[0]
        assert abs(decomp.a_drift[0]) > 0.1
        assert res.drift_gain is not None and res.drift_gain > 0.0

    def test_red_strong_opposing_control_and_near_zero_total_accel(self) -> None:
        """Near-zero total acceleration under opposing control is decomposed truthfully without hiding drift."""
        provider, _ = _make_pendulum_provider()
        predictor = DimeDriftPredictor(provider)

        q_val = 0.5
        m, g, length_m = 1.0, 9.81, 1.0
        # Gravity torque on pendulum of mass m, length length_m: tau_g = - m*g*length_m*sin(q)
        # For zero total acceleration at v=0, required balancing torque is +m*g*length_m*sin(q)
        holding_torque = m * g * length_m * np.sin(q_val)

        state = DimeCompleteState(
            t=0.0,
            q=np.array([q_val], dtype=np.float64),
            v=np.array([0.0], dtype=np.float64),
            model_hash=provider.model_hash,
            units=dict(CANONICAL_DIME_UNITS),
        )
        state_dist = StateDistribution(mean=state, covariance=np.diag([1e-4, 1e-4]))
        ctrl_dist = ControlDistribution(
            mean=np.array([holding_torque], dtype=np.float64),
            covariance=np.array([[1e-4]], dtype=np.float64),
            physical_types=("torque",),
            units=("N*m",),
        )
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
        )

        res = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            horizon_s=0.01,
            dt=0.01,
            mode=PredictionMode.EXPLICIT_CONTROL,
        )
        assert res.validity_status == PredictionValidityStatus.VALID
        # Near-zero total acceleration
        assert res.predicted_mean is not None
        # Net residual force ratio must be near zero (< 0.05) under opposing control
        assert res.cancellation_index is not None
        assert res.cancellation_index < 0.05

    def test_red_contact_switch_disables_proposal_with_receipt(self) -> None:
        """Contact switch disables the proposal and returns an explanatory validity receipt."""
        provider, state = _make_pendulum_provider()
        predictor = DimeDriftPredictor(provider)

        state_dist = StateDistribution(mean=state, covariance=np.diag([1e-4, 1e-4]))
        ctrl_dist = ControlDistribution(
            mean=np.array([0.0], dtype=np.float64),
            covariance=np.array([[1e-3]], dtype=np.float64),
        )
        # Contact phase unstable / switch detected
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
            contact_phase_stable=False,
        )

        res = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            horizon_s=0.1,
            dt=0.01,
        )
        assert res.validity_status == PredictionValidityStatus.INVALID_CONTACT
        assert "contact" in res.validity_receipt.lower()
        assert res.zero_control_branch is None
        assert res.predicted_mean is None

    def test_red_invalid_horizon_disables_proposal(self) -> None:
        """Horizon exceeding max qualified horizon disables proposal with receipt."""
        provider, state = _make_pendulum_provider()
        predictor = DimeDriftPredictor(provider)

        state_dist = StateDistribution(mean=state, covariance=np.diag([1e-4, 1e-4]))
        ctrl_dist = ControlDistribution(
            mean=np.array([0.0], dtype=np.float64),
            covariance=np.array([[1e-3]], dtype=np.float64),
        )
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
            max_qualified_horizon_s=0.2,
        )

        res = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            horizon_s=0.5,  # Exceeds 0.2s
            dt=0.01,
        )
        assert res.validity_status == PredictionValidityStatus.INVALID_HORIZON
        assert "horizon" in res.validity_receipt.lower()
        assert res.zero_control_branch is None

    def test_red_stale_model_hash_fails(self) -> None:
        """Mismatched model hash triggers STALE_MODEL_HASH receipt."""
        provider, state = _make_pendulum_provider()
        predictor = DimeDriftPredictor(provider)

        stale_state = DimeCompleteState(
            t=0.0,
            q=state.q,
            v=state.v,
            model_hash="wrong-stale-hash",
            units=dict(CANONICAL_DIME_UNITS),
        )
        state_dist = StateDistribution(
            mean=stale_state, covariance=np.diag([1e-4, 1e-4])
        )
        ctrl_dist = ControlDistribution(
            mean=np.array([0.0], dtype=np.float64),
            covariance=np.array([[1e-3]], dtype=np.float64),
        )
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
        )

        res = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            horizon_s=0.05,
            dt=0.01,
        )
        assert res.validity_status == PredictionValidityStatus.STALE_MODEL_HASH
        assert "model_hash" in res.validity_receipt

    def test_red_uncertain_mass_propagates_into_covariance(self) -> None:
        """Uncertain mass strictly inflates the predicted state covariance."""
        provider, state = _make_pendulum_provider()
        predictor = DimeDriftPredictor(provider)

        state_dist = StateDistribution(mean=state, covariance=np.diag([1e-4, 1e-4]))
        ctrl_dist = ControlDistribution(
            mean=np.array([0.0], dtype=np.float64),
            covariance=np.array([[1e-3]], dtype=np.float64),
        )

        unc_no_mass = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
            mass_uncertainty_std=0.0,
        )
        res_no_mass = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=unc_no_mass,
            horizon_s=0.05,
            dt=0.01,
        )

        unc_with_mass = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
            mass_uncertainty_std=0.2,  # 20% mass standard deviation
        )
        res_with_mass = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=unc_with_mass,
            horizon_s=0.05,
            dt=0.01,
        )

        assert res_no_mass.predicted_covariance is not None
        assert res_with_mass.predicted_covariance is not None
        # Trace of covariance must be strictly higher with mass uncertainty
        assert np.trace(res_with_mass.predicted_covariance) > np.trace(
            res_no_mass.predicted_covariance
        )


class TestDimeDriftPredictionGreenSuite:
    """GREEN test cases verifying exact mathematical equivalence and contracts."""

    def test_green_linear_marginalization_equals_explicit_control_elimination(
        self,
    ) -> None:
        """Analytic linear-system marginalization equals explicit Gaussian-control elimination.

        Verifies that:
            min_u [ 1/2 ||x_{k+1} - (x_ZTCF + G*u)||_Q^2 + 1/2 ||u - mu_u||_{Sigma_u}^2 ]
        is algebraically and numerically identical to:
            1/2 || (x_{k+1} - x_ZTCF) - G*mu_u ||_{G*Sigma_u*G^T + Q}^2
        """
        dim_x = 2
        dim_u = 1

        G = np.array([[0.01], [0.05]], dtype=np.float64)  # Control matrix G
        Q = np.diag([1e-3, 2e-3])  # Dynamics process noise
        Q_inv = np.linalg.inv(Q)
        Sigma_u = np.array([[0.04]], dtype=np.float64)  # Control prior covariance
        Sigma_u_inv = np.linalg.inv(Sigma_u)
        mu_u = np.array([1.2], dtype=np.float64)  # Control prior mean

        x_ztcf = np.array([0.5, 0.1], dtype=np.float64)
        x_kp1 = np.array([0.52, 0.18], dtype=np.float64)

        # 1. Marginalized distance:
        Sigma_trans = G @ Sigma_u @ G.T + Q
        Sigma_trans_inv = np.linalg.inv(Sigma_trans)
        diff_marg = (x_kp1 - x_ztcf) - G @ mu_u
        J_marg = 0.5 * float(diff_marg.T @ Sigma_trans_inv @ diff_marg)

        # 2. Explicit Gaussian-control elimination via analytical minimum:
        # J(u) = 1/2 (d - G*u)^T Q^-1 (d - G*u) + 1/2 (u - mu_u)^T Sigma_u^-1 (u - mu_u)
        # where d = x_{k+1} - x_ZTCF.
        d = x_kp1 - x_ztcf
        H_u = G.T @ Q_inv @ G + Sigma_u_inv
        g_u = -G.T @ Q_inv @ d - Sigma_u_inv @ mu_u
        u_star = -np.linalg.solve(H_u, g_u)

        r_dyn = d - G @ u_star
        r_ctrl = u_star - mu_u
        J_explicit_min = 0.5 * float(
            r_dyn.T @ Q_inv @ r_dyn + r_ctrl.T @ Sigma_u_inv @ r_ctrl
        )

        # Equality within tight tolerance
        assert np.isclose(J_marg, J_explicit_min, rtol=1e-7, atol=1e-9)

    def test_green_propagated_covariance_includes_uncertain_input_and_is_psd(
        self,
    ) -> None:
        """Propagated state covariance includes uncertain input covariance and remains strictly PSD."""
        provider, state = _make_pendulum_provider()
        predictor = DimeDriftPredictor(provider)

        state_dist = StateDistribution(
            mean=state, covariance=np.array([[2e-4, 1e-5], [1e-5, 3e-4]])
        )
        ctrl_dist = ControlDistribution(
            mean=np.array([0.5], dtype=np.float64),
            covariance=np.array([[0.05]], dtype=np.float64),
        )
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 2e-5]),
        )

        res = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            horizon_s=0.05,
            dt=0.01,
            mode=PredictionMode.MARGINALIZED_CONTROL,
        )
        assert res.predicted_covariance is not None
        cov = res.predicted_covariance

        # Symmetry check
        assert np.allclose(cov, cov.T, atol=1e-12)

        # Positive semi-definite (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert bool(np.all(eigenvalues >= 0.0))
        assert eigenvalues[0] > 0.0  # Strictly positive definite with process noise

    def test_green_nonlinear_sampled_reference_bounds_linearization_error(
        self,
    ) -> None:
        """Nonlinear sampled reference bounds first-order Taylor linearization error."""
        provider, state = _make_pendulum_provider()
        predictor = DimeDriftPredictor(provider)

        state_dist = StateDistribution(mean=state, covariance=np.diag([1e-4, 1e-4]))
        ctrl_dist = ControlDistribution(
            mean=np.array([0.5], dtype=np.float64),
            covariance=np.array([[1e-3]], dtype=np.float64),
        )
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
        )

        res = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            horizon_s=0.05,
            dt=0.01,
        )
        assert res.linearization_error_bound is not None
        assert res.linearization_error_bound >= 0.0
        # For small dt=0.01, horizon=0.05s, linearization error is small (< 0.1)
        assert res.linearization_error_bound < 0.1

    def test_green_drift_gain_against_frozen_baseline(self) -> None:
        """Test drift gain on the frozen pendulum fixture without tuning test data."""
        provider, state = _make_pendulum_provider()
        predictor = DimeDriftPredictor(provider)

        # Zero control test
        state_dist = StateDistribution(mean=state, covariance=np.diag([1e-4, 1e-4]))
        ctrl_dist = ControlDistribution(
            mean=np.array([0.0], dtype=np.float64),
            covariance=np.array([[1e-4]], dtype=np.float64),
        )
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
        )

        res = predictor.predict(
            state_dist=state_dist,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            horizon_s=0.05,
            dt=0.01,
        )
        assert res.drift_gain is not None
        # With zero control, total accel equals drift accel, so drift gain = ||a_drift|| / ||a_tot|| == 1.0
        assert np.isclose(res.drift_gain, 1.0, atol=1e-4)

    def test_green_mutual_exclusion_of_duplicate_physics_likelihoods(self) -> None:
        """Registering both marginalized-control transition and explicit-control dynamics on the same interval fails exclusivity."""
        provider, state = _make_pendulum_provider()

        ctrl_dist = ControlDistribution(
            mean=np.array([0.0], dtype=np.float64),
            covariance=np.array([[1e-3]], dtype=np.float64),
        )
        uncertainty = ModelContactUncertainty(
            process_noise_covariance=np.diag([1e-5, 1e-5]),
        )

        factor_marg = DimeDriftTransitionCriterion(
            provider=provider,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            dt=0.05,
            factor_type="marginalized_input_transition",
            t_start=0.0,
            t_end=0.05,
        )

        factor_explicit = DimeDriftTransitionCriterion(
            provider=provider,
            control_dist=ctrl_dist,
            uncertainty=uncertainty,
            dt=0.05,
            factor_type="explicit_input_likelihood",
            t_start=0.0,
            t_end=0.05,
        )

        exclusivity_contract = RuntimeExclusivityContract()
        exclusivity_contract.register_factor(factor_marg)

        # Adding both factors on the same interval [0.0, 0.05] must raise PreconditionError
        with pytest.raises(PreconditionError) as exc_info:
            exclusivity_contract.register_factor(factor_explicit)
        assert "runtime exclusivity violation" in str(exc_info.value).lower()
