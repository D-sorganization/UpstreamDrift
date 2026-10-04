"""Behavioral test suite for coupled state-control full-dynamics window factors (#11426).

Validates:
- RED tests:
  1. Known actuated pendulum under observation noise: recovers identifiable controls/states
     within frozen tolerances (max_angular_drift_rad <= 0.05, min_alignment >= 0.95).
  2. Two-link dynamic coupling: verifies dynamic coupling and acceleration defects on uncoordinated
     state trajectories.
  3. Arbitrary root wrench shortcut: underactuated root constraints reject unphysical root wrenches
     on unactuated DOFs.
  4. Discontinuous controls: penalized by control variation regularizers.
  5. Excessive model-discrepancy slack: bounded model discrepancy reports slack violations.
  6. Contact hooks fail-closed: active contact asserted without a qualified contact provider fails closed.
  7. Failed dynamics: dynamics engine failures or non-finite results are not converted into finite best-fit success.
- GREEN tests:
  1. Exact vs soft defect selection: full integrated transition defects, bounded model discrepancy,
     actuator bounds, underactuated root constraints, and control variation regularizers are separately exported.
  2. Derivative verification: factor derivatives match finite differences on smooth segments.
  3. Noiseless recovery: noiseless consistent trajectory yields zero transition defect and zero cost.
  4. Clean contact exposure: contact hooks expose native no-contact semantics without claiming stance completeness.
  5. Actuator limits: controls exceeding actuator channel limits are penalized and reported in actuator bound residuals.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError, PreconditionError
from src.shared.python.estimation.dime_contracts import (
    AnalyticPendulumProvider,
    DeterministicFakeProvider,
    DimeCompleteState,
    UnderactuatedAnalyticProvider,
)
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    NumericAcceptanceThresholds,
)
from src.shared.python.estimation.dime_observation_factors import (
    Marker3DObservationFactor,
    ObservationTiming,
)
from src.shared.python.estimation.dime_window_factors import (
    DimeContactHook,
    DimeDynamicsWindowFactor,
    DimeDynamicsWindowResult,
    DimeWindowFactorConfig,
    DimeWindowResidualComponents,
)
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
    make_underactuated_analytic_fixture,
)


# ==============================================================================
# Helper Factories
# ==============================================================================


def _make_pendulum_states(
    theta_arr: np.ndarray,
    omega_arr: np.ndarray,
    times: np.ndarray,
    model_hash: str = "sha256-pendulum-model-v1",
) -> tuple[DimeCompleteState, ...]:
    """Build tuple of DimeCompleteState instances for pendulum tests."""
    return tuple(
        DimeCompleteState(
            t=float(t),
            q=np.array([th], dtype=np.float64),
            v=np.array([om], dtype=np.float64),
            model_hash=model_hash,
            units=dict(CANONICAL_DIME_UNITS),
        )
        for t, th, om in zip(times, theta_arr, omega_arr, strict=True)
    )


def _make_twolink_states(
    q_arr: np.ndarray,
    v_arr: np.ndarray,
    times: np.ndarray,
    model_hash: str = "sha256-underactuated-model-v1",
) -> tuple[DimeCompleteState, ...]:
    """Build tuple of DimeCompleteState instances for two-link tests."""
    return tuple(
        DimeCompleteState(
            t=float(t),
            q=np.array(q, dtype=np.float64),
            v=np.array(v, dtype=np.float64),
            model_hash=model_hash,
            units=dict(CANONICAL_DIME_UNITS),
        )
        for t, q, v in zip(times, q_arr, v_arr, strict=True)
    )


# ==============================================================================
# RED Test Suite
# ==============================================================================


class TestRedDynamicsWindowFactors:
    """RED behavioral suite enforcing physical invariants, fail-closed rules, and recovery."""

    @pytest.mark.unit
    def test_red_known_actuated_pendulum_under_observation_noise_recovery(self) -> None:
        """Actuated pendulum under observation noise recovers identifiable controls/states within frozen tolerances."""
        provider = AnalyticPendulumProvider()
        dt = 0.02
        n_steps = 10
        times = np.linspace(0.0, dt * n_steps, n_steps + 1)

        # Ground truth forward rollout with known constant control tau = 2.0 Nm
        u_true = np.full((n_steps, 1), 2.0, dtype=np.float64)
        true_states = [provider.get_state()]
        curr_state = true_states[0]
        for k in range(n_steps):
            from src.shared.python.estimation.dime_contracts import DimeFullStepRequest

            step_res = provider.step(
                DimeFullStepRequest(
                    state=curr_state,
                    controls=u_true[k],
                    dt=dt,
                    model_hash=provider.model_hash,
                )
            )
            curr_state = step_res.next_state
            true_states.append(curr_state)

        true_theta = np.array([float(s.q[0]) for s in true_states])
        true_omega = np.array([float(s.v[0]) for s in true_states])

        # Add calibrated observation noise to angles (e.g. sigma = 0.01 rad)
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0.0, 0.008, size=len(true_theta))
        noisy_theta = true_theta + noise

        # Marker observation setup: 1 marker at tip of pendulum [0, sin(th), -cos(th)]
        obs_timing = ObservationTiming(timestamps=times)

        def _fk(q: np.ndarray) -> np.ndarray:
            th = float(q[0])
            return np.array([[0.0, float(np.sin(th)), -float(np.cos(th))]])

        # Generate noisy marker 3D observations
        noisy_markers = np.array(
            [[0.0, float(np.sin(th)), -float(np.cos(th))] for th in noisy_theta]
        )
        obs_factor = Marker3DObservationFactor(
            observations_3d_m=noisy_markers,
            kinematics_fn=_fk,
            covariance=np.array([1e-4, 1e-4, 1e-4]),
            timing=obs_timing,
        )

        config = DimeWindowFactorConfig(
            dt=dt,
            defect_mode="soft",
            defect_weight=100.0,
            control_variation_weight=0.01,
            control_magnitude_weight=0.001,
        )
        factor = DimeDynamicsWindowFactor(
            provider=provider,
            times=times,
            config=config,
            observation_factors=[obs_factor],
        )

        # Initial perturbed guess
        init_theta = noisy_theta.copy()
        init_omega = np.gradient(init_theta, dt)
        init_states = _make_pendulum_states(init_theta, init_omega, times)
        init_u = np.zeros_like(u_true)

        result = factor.solve(initial_states=init_states, initial_controls=init_u)

        assert result.success is True
        recovered_theta = np.array([float(s.q[0]) for s in result.trajectory_states])
        recovered_u = result.estimated_controls

        # Compare against frozen benchmark tolerances from dime_manifest.NumericAcceptanceThresholds
        thresholds = NumericAcceptanceThresholds()
        angular_drift = np.max(np.abs(recovered_theta - true_theta))
        assert angular_drift <= thresholds.max_angular_drift_rad, (
            f"Recovered angle drift {angular_drift:.4f} rad exceeds frozen limit {thresholds.max_angular_drift_rad}"
        )

        # Control alignment metric >= min_alignment (0.95)
        u_norm_true = np.linalg.norm(u_true)
        u_norm_rec = np.linalg.norm(recovered_u)
        alignment = float(
            np.dot(u_true.flat, recovered_u.flat) / (u_norm_true * u_norm_rec)
        )
        assert alignment >= thresholds.min_alignment, (
            f"Control alignment {alignment:.4f} is below frozen threshold {thresholds.min_alignment}"
        )

    @pytest.mark.unit
    def test_red_two_link_dynamic_coupling_and_acceleration_defects(self) -> None:
        """Two-link unactuated coupling produces severe acceleration defects on uncoordinated trajectories."""
        provider = UnderactuatedAnalyticProvider()
        dt = 0.01
        times = np.array([0.0, dt, 2 * dt])

        config = DimeWindowFactorConfig(dt=dt, defect_mode="soft", defect_weight=1.0)
        factor = DimeDynamicsWindowFactor(provider=provider, times=times, config=config)

        # State 1: consistent with passive dynamics
        q0 = np.array([0.1, 0.2])
        v0 = np.array([0.0, 0.0])
        # Suppose joint 0 undergoes massive acceleration while joint 1 is stationary and u=0
        q1_unphysical = np.array([1.5, 0.2])
        v1_unphysical = np.array([10.0, 0.0])
        q2_unphysical = np.array([3.0, 0.2])
        v2_unphysical = np.array([20.0, 0.0])

        states_unphysical = _make_twolink_states(
            np.array([q0, q1_unphysical, q2_unphysical]),
            np.array([v0, v1_unphysical, v2_unphysical]),
            times,
        )
        u_zero = np.zeros((2, 1), dtype=np.float64)

        components = factor.evaluate_components(states_unphysical, u_zero)
        defect_norm = np.linalg.norm(components.transition_defects)
        # Dynamic coupling dictates joint 0 cannot accelerate wildly without torques/gravity coupling
        assert defect_norm > 5.0, (
            f"Unphysical uncoordinated trajectory should have large defect, got {defect_norm}"
        )

    @pytest.mark.unit
    def test_red_arbitrary_root_wrench_shortcut_rejected(self) -> None:
        """Underactuated root constraints reject unphysical root wrenches on unactuated DOFs."""
        provider = UnderactuatedAnalyticProvider()
        dt = 0.01
        times = np.array([0.0, dt, 2 * dt])

        config = DimeWindowFactorConfig(
            dt=dt,
            underactuated_root_weight=1000.0,
        )
        factor = DimeDynamicsWindowFactor(provider=provider, times=times, config=config)

        states = _make_twolink_states(
            np.array([[0.1, 0.2], [0.11, 0.21], [0.12, 0.22]]),
            np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]),
            times,
        )

        # Attempting an arbitrary root wrench on unactuated DOF 0 (e.g. 50.0 Nm shortcut)
        unphysical_root_control = np.array([[50.0], [50.0]])
        with pytest.raises((ContractViolationError, PreconditionError)):
            factor.evaluate_components(
                states,
                controls=np.zeros((2, 1)),
                unactuated_root_wrenches=unphysical_root_control,
            )

    @pytest.mark.unit
    def test_red_discontinuous_controls_penalized_by_variation_regularizers(
        self,
    ) -> None:
        """Discontinuous controls are penalized heavily by control variation regularizers."""
        provider = AnalyticPendulumProvider()
        dt = 0.02
        n_steps = 4
        times = np.linspace(0.0, dt * n_steps, n_steps + 1)

        config = DimeWindowFactorConfig(
            dt=dt,
            control_variation_weight=10.0,
        )
        factor = DimeDynamicsWindowFactor(provider=provider, times=times, config=config)

        states = _make_pendulum_states(
            np.zeros(n_steps + 1), np.zeros(n_steps + 1), times
        )

        # Smooth controls: 1.0, 1.1, 1.2, 1.3
        u_smooth = np.array([[1.0], [1.1], [1.2], [1.3]])
        # Discontinuous controls with alternating sign jumps: 10.0, -10.0, 10.0, -10.0
        u_discontinuous = np.array([[10.0], [-10.0], [10.0], [-10.0]])

        comp_smooth = factor.evaluate_components(states, u_smooth)
        comp_disc = factor.evaluate_components(states, u_discontinuous)

        var_smooth = np.linalg.norm(comp_smooth.control_variation_residuals)
        var_disc = np.linalg.norm(comp_disc.control_variation_residuals)

        assert var_disc > 50.0 * var_smooth, (
            f"Discontinuous control variation ({var_disc}) must heavily exceed smooth ({var_smooth})"
        )

    @pytest.mark.unit
    def test_red_excessive_model_discrepancy_reports_slack_violations(self) -> None:
        """Bounded model discrepancy reports slack violations when slack exceeds configured limit."""
        provider = AnalyticPendulumProvider()
        dt = 0.02
        times = np.array([0.0, dt, 2 * dt])

        config = DimeWindowFactorConfig(
            dt=dt,
            enable_model_discrepancy=True,
            slack_bound=0.5,  # Max permitted slack norm = 0.5 m/s^2
            slack_weight=5.0,
        )
        factor = DimeDynamicsWindowFactor(provider=provider, times=times, config=config)

        states = _make_pendulum_states(np.zeros(3), np.zeros(3), times)
        controls = np.zeros((2, 1))

        # Slacks exceeding bound: [2.5, 3.0]
        slacks_excessive = np.array([[2.5], [3.0]])
        comp = factor.evaluate_components(states, controls, slacks=slacks_excessive)

        assert comp.is_slack_bounded is False
        assert len(comp.slack_violations) > 0
        assert comp.max_slack == pytest.approx(3.0)

    @pytest.mark.unit
    def test_red_contact_hooks_fail_closed_without_contact_provider(self) -> None:
        """Active contact asserted without a contact provider fails closed."""
        provider = AnalyticPendulumProvider()
        dt = 0.02
        times = np.array([0.0, dt, 2 * dt])

        # Asserting active contact while contact_provider is None must fail closed
        with pytest.raises((ContractViolationError, PreconditionError)):
            DimeContactHook(
                contact_provider=None,
                assert_active_contact=True,
            )

        # Claiming stance completeness on analytic model without physical contact engine must fail closed
        with pytest.raises((ContractViolationError, PreconditionError)):
            DimeContactHook(
                contact_provider=None,
                stance_completeness_claimed=True,
            )

    @pytest.mark.unit
    def test_red_failed_dynamics_is_not_converted_into_finite_best_fit_success(
        self,
    ) -> None:
        """Simulated dynamics failure is never converted into finite best-fit success."""
        provider = DeterministicFakeProvider()
        provider.inject_step_failure(True)  # Forces simulated engine failure
        dt = 0.05
        times = np.array([0.0, dt, 2 * dt])

        config = DimeWindowFactorConfig(dt=dt)
        factor = DimeDynamicsWindowFactor(provider=provider, times=times, config=config)

        states = tuple(
            DimeCompleteState(
                t=float(t),
                q=np.zeros(1),
                v=np.zeros(1),
                model_hash=provider.model_hash,
                units=dict(CANONICAL_DIME_UNITS),
            )
            for t in times
        )
        controls = np.zeros((2, 1))

        # Solving with a failing engine must not return success=True
        result = factor.solve(initial_states=states, initial_controls=controls)
        assert result.success is False
        assert "failure" in result.message.lower() or "error" in result.message.lower()


# ==============================================================================
# GREEN Test Suite
# ==============================================================================


class TestGreenDynamicsWindowFactors:
    """GREEN behavioral suite validating exported components, finite differences, and exact defects."""

    @pytest.mark.unit
    def test_green_exact_vs_soft_defect_selection_and_export(self) -> None:
        """Full integrated transition defects and all components are separately exported in exact and soft modes."""
        provider = AnalyticPendulumProvider()
        dt = 0.02
        times = np.array([0.0, dt, 2 * dt])

        # Soft defect mode
        config_soft = DimeWindowFactorConfig(
            dt=dt, defect_mode="soft", defect_weight=10.0
        )
        factor_soft = DimeDynamicsWindowFactor(
            provider=provider, times=times, config=config_soft
        )

        # Exact defect mode
        config_exact = DimeWindowFactorConfig(dt=dt, defect_mode="exact")
        factor_exact = DimeDynamicsWindowFactor(
            provider=provider, times=times, config=config_exact
        )

        states = _make_pendulum_states(
            np.array([0.1, 0.12, 0.14]), np.array([0.0, 0.1, 0.2]), times
        )
        controls = np.array([[0.5], [0.5]])

        comp_soft = factor_soft.evaluate_components(states, controls)
        comp_exact = factor_exact.evaluate_components(states, controls)

        # Both export transition defects of identical shape
        assert comp_soft.transition_defects.shape == (2, 2)
        assert comp_exact.transition_defects.shape == (2, 2)
        np.testing.assert_allclose(
            comp_soft.transition_defects, comp_exact.transition_defects
        )

        # Separately exported components exist and are finite
        for comp in (comp_soft, comp_exact):
            assert comp.control_variation_residuals is not None
            assert comp.control_magnitude_residuals is not None
            assert comp.actuator_bound_residuals is not None
            assert comp.underactuated_root_residuals is not None
            assert comp.model_discrepancy_residuals is not None
            assert np.all(np.isfinite(comp.transition_defects))

    @pytest.mark.unit
    def test_green_finite_difference_derivatives_match_analytical(self) -> None:
        """Residual derivatives match finite differences on smooth trajectory segments."""
        provider = AnalyticPendulumProvider()
        dt = 0.02
        times = np.array([0.0, dt, 2 * dt])

        config = DimeWindowFactorConfig(
            dt=dt,
            defect_weight=1.0,
            control_variation_weight=0.1,
            control_magnitude_weight=0.01,
        )
        factor = DimeDynamicsWindowFactor(provider=provider, times=times, config=config)

        states = _make_pendulum_states(
            np.array([0.2, 0.22, 0.23]), np.array([0.1, 0.12, 0.13]), times
        )
        controls = np.array([[1.5], [1.6]])

        param_vec = factor.layout.pack(states, controls)
        jac_analytic = factor.compute_jacobian(param_vec)
        jac_fd = factor.compute_finite_difference_jacobian(param_vec, eps=1e-6)

        np.testing.assert_allclose(jac_analytic, jac_fd, rtol=1e-4, atol=1e-4)

    @pytest.mark.unit
    def test_green_noiseless_pendulum_zero_defect_recovery(self) -> None:
        """Consistent ground-truth trajectory produces identically zero defect residuals."""
        provider = AnalyticPendulumProvider()
        dt = 0.01
        times = np.array([0.0, dt, 2 * dt])

        # Generate ground truth step
        s0 = provider.get_state()
        from src.shared.python.estimation.dime_contracts import DimeFullStepRequest

        res1 = provider.step(
            DimeFullStepRequest(
                state=s0,
                controls=np.array([1.0]),
                dt=dt,
                model_hash=provider.model_hash,
            )
        )
        res2 = provider.step(
            DimeFullStepRequest(
                state=res1.next_state,
                controls=np.array([1.0]),
                dt=dt,
                model_hash=provider.model_hash,
            )
        )

        gt_states = (s0, res1.next_state, res2.next_state)
        gt_controls = np.array([[1.0], [1.0]])

        config = DimeWindowFactorConfig(dt=dt, defect_mode="soft", defect_weight=1.0)
        factor = DimeDynamicsWindowFactor(provider=provider, times=times, config=config)

        comp = factor.evaluate_components(gt_states, gt_controls)
        np.testing.assert_allclose(comp.transition_defects, 0.0, atol=1e-12)

    @pytest.mark.unit
    def test_green_contact_hooks_clean_exposure_with_native_no_contact(self) -> None:
        """Contact hooks cleanly expose native no-contact semantics without claiming stance completeness."""
        hook = DimeContactHook(
            contact_provider=None,
            assert_active_contact=False,
            stance_completeness_claimed=False,
        )
        assert hook.has_active_contact is False
        assert hook.stance_completeness_claimed is False

    @pytest.mark.unit
    def test_green_actuator_limits_enforced(self) -> None:
        """Controls exceeding actuator channel limits produce actuator bound residuals."""
        provider = AnalyticPendulumProvider()  # limits are (-50.0, 50.0)
        dt = 0.02
        times = np.array([0.0, dt, 2 * dt])

        config = DimeWindowFactorConfig(
            dt=dt,
            actuator_bound_penalty_weight=10.0,
        )
        factor = DimeDynamicsWindowFactor(provider=provider, times=times, config=config)

        states = _make_pendulum_states(np.zeros(3), np.zeros(3), times)
        # Limit is 50.0; passing 65.0 should produce penalty of (65 - 50) * sqrt(10.0)
        controls_exceeding = np.array([[65.0], [-70.0]])

        comp = factor.evaluate_components(states, controls_exceeding)
        assert np.any(comp.actuator_bound_residuals > 0.0)
        # Residual on joint 0 for 65.0 Nm (excess 15.0)
        assert comp.actuator_bound_residuals[0, 0] == pytest.approx(
            15.0 * np.sqrt(10.0)
        )
        # Residual on joint 0 for -70.0 Nm (excess 20.0)
        assert comp.actuator_bound_residuals[1, 0] == pytest.approx(
            20.0 * np.sqrt(10.0)
        )
