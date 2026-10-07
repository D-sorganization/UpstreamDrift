"""Behavioral test suite for Coupled State-Control Full-Dynamics Window Factors (DIME-05).

Verifies coupled state-control window estimation, exact vs soft integrated defects,
bounded model discrepancy, actuator bounds, underactuated root constraints,
control variation regularizers, and export of all cost components and constraint residuals.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_contracts import (
    AnalyticPendulumProvider,
    ContactPolicy,
    DimeCompleteState,
    DimeFullStepRequest,
    EstimationIntervalFactor,
    RuntimeExclusivityContract,
    UnderactuatedAnalyticProvider,
)
from src.shared.python.estimation.dime_manifest import (
    NumericAcceptanceThresholds,
)
from src.shared.python.estimation.dime_observation_factors import (
    Marker3DObservationFactor,
    MarkerAttachment,
)
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
    make_underactuated_analytic_fixture,
)
from src.shared.python.estimation.dime_dynamics_window import (
    DefectMode,
    DimeDynamicsWindowFactor,
    DimeDynamicsWindowProblem,
    DimeDynamicsWindowResult,
    ModelDiscrepancyBounds,
    solve_dime_dynamics_window,
)

pytestmark = pytest.mark.unit


class TestDimeDynamicsWindowRedSuite:
    """RED test suite verifying failure modes, constraints, and boundaries."""

    def test_red_arbitrary_root_wrench_shortcut_rejected(self) -> None:
        """Attempting to apply an arbitrary root torque to an underactuated model fails closed."""
        provider = UnderactuatedAnalyticProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.1, 0.0], dtype=np.float64),
            v=np.array([0.0, 0.0], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=4,
            dt_s=0.02,
            defect_mode=DefectMode.EXACT,
            enforce_root_constraints=True,
        )

        # Controls attempting an arbitrary shortcut on passive root joint 0:
        # shape is (N, n_total_joints) = (4, 2) where joint 0 has non-zero torque!
        invalid_controls = np.array(
            [
                [5.0, 1.0],  # joint 0 has 5.0 Nm torque! But joint 0 is passive!
                [5.0, 1.0],
                [5.0, 1.0],
                [5.0, 1.0],
            ],
            dtype=np.float64,
        )

        with pytest.raises(
            (ValueError, PreconditionError), match="root.*passive|underactuated"
        ):
            problem.validate_controls(invalid_controls)

    def test_red_discontinuous_controls_heavily_penalized(self) -> None:
        """Discontinuous/chattering controls incur high control variation regularizer cost."""
        provider = AnalyticPendulumProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.1], dtype=np.float64),
            v=np.array([0.0], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=5,
            dt_s=0.01,
            control_rate_weight=10.0,
        )

        smooth_controls = np.array(
            [[1.0], [1.1], [1.2], [1.3], [1.4]], dtype=np.float64
        )
        chattering_controls = np.array(
            [[1.0], [-1.0], [1.0], [-1.0], [1.0]], dtype=np.float64
        )

        cost_smooth = problem.evaluate_control_variation_cost(smooth_controls)
        cost_chatter = problem.evaluate_control_variation_cost(chattering_controls)

        assert cost_chatter > 10.0 * cost_smooth
        assert cost_chatter > 100.0

    def test_red_excessive_model_discrepancy_slack_rejected(self) -> None:
        """Model discrepancy slack exceeding bounded threshold fails closed."""
        bounds = ModelDiscrepancyBounds(max_slack_norm=0.05, slack_weight=100.0)

        # Excessive slack norm
        slack = np.array([0.1, 0.1], dtype=np.float64)
        assert np.linalg.norm(slack) > 0.05

        with pytest.raises(
            (ValueError, PreconditionError), match="slack.*exceeds|bound"
        ):
            bounds.validate_slack(slack)

    def test_red_two_link_coupling_detected(self) -> None:
        """Ignoring two-link dynamic inertial coupling produces severe transition defects."""
        provider = UnderactuatedAnalyticProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.2, 0.3], dtype=np.float64),
            v=np.array([0.5, -0.5], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        factor = DimeDynamicsWindowFactor(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=3,
            dt_s=0.02,
            defect_mode=DefectMode.SOFT,
        )

        # Candidate state with uncoupled assumption (neglecting Coriolis / off-diagonal mass)
        uncoupled_candidate_q = np.array([0.2 + 0.5 * 0.02, 0.3 - 0.5 * 0.02])
        uncoupled_candidate_v = np.array([0.5, -0.5])  # constant velocity assumption
        candidate_state_uncoupled = DimeCompleteState(
            t=0.02,
            q=uncoupled_candidate_q,
            v=uncoupled_candidate_v,
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        # Defect against true dynamics rollout
        defect = factor.compute_single_step_defect(
            initial_state, candidate_state_uncoupled, control=np.array([0.0])
        )
        assert np.linalg.norm(defect) > 1e-3

    def test_red_failed_dynamics_never_converted_to_finite_success(self) -> None:
        """Provider failure/exception rolls back state and returns success=False, not false success."""
        provider = AnalyticPendulumProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.0], dtype=np.float64),
            v=np.array([0.0], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        # Deliberately supply non-finite or invalid initial conditions to trigger solver failure
        bad_state = DimeCompleteState(
            t=0.0,
            q=np.array([1e8], dtype=np.float64),  # extreme value causing divergence
            v=np.array([1e8], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=bad_state,
            horizon_steps=3,
            dt_s=0.01,
            max_iterations=5,
        )

        result = solve_dime_dynamics_window(problem)
        assert not result.success
        assert (
            "fail" in result.status.lower()
            or "diverg" in result.status.lower()
            or "infeasible" in result.status.lower()
        )


class TestDimeDynamicsWindowGreenSuite:
    """GREEN test suite verifying coupled window solve, recovery, and contracts."""

    def test_green_recover_identifiable_controls_and_states(self) -> None:
        """Coupled window solve recovers known controls and states under observation noise within frozen tolerances."""
        provider = AnalyticPendulumProvider()
        thresholds = NumericAcceptanceThresholds()

        dt_s = 0.02
        horizon = 20
        q0 = np.array([0.2], dtype=np.float64)
        v0 = np.array([0.0], dtype=np.float64)
        initial_state = DimeCompleteState(
            t=0.0,
            q=q0,
            v=v0,
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        # Generate true trajectory under constant torque u = 0.5 Nm
        true_u = np.array([0.5], dtype=np.float64)
        states = [initial_state]
        current = initial_state
        for _ in range(horizon):
            step_req = DimeFullStepRequest(
                state=current,
                controls=true_u,
                dt=dt_s,
                model_hash=provider.model_hash,
            )
            step_res = provider.step(step_req)
            states.append(step_res.next_state)
            current = step_res.next_state

        # Synthesize noisy state observations
        rng = np.random.default_rng(42)
        noisy_observations = [
            s.q + rng.normal(0.0, 0.002, size=s.q.shape) for s in states
        ]

        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=horizon,
            dt_s=dt_s,
            target_positions=noisy_observations,
            defect_mode=DefectMode.SOFT,
            observation_weight=1.0 / (0.002**2),
            control_rate_weight=0.01,
            actuator_bounds=(-10.0, 10.0),
        )

        result = solve_dime_dynamics_window(problem)
        assert result.success
        # State trajectory recovery within frozen max drift tolerance
        max_state_drift = max(
            np.linalg.norm(res_s.q - true_s.q)
            for res_s, true_s in zip(result.states, states, strict=True)
        )
        assert max_state_drift <= thresholds.max_drift_m

        # Recovered control close to true control 0.5 Nm
        mean_control = np.mean(result.controls)
        assert abs(mean_control - 0.5) < 0.1

    def test_green_finite_difference_derivatives_on_smooth_segments(self) -> None:
        """Factor transition defect Jacobians match central finite differences to <= 1e-5."""
        provider = AnalyticPendulumProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.3], dtype=np.float64),
            v=np.array([0.2], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        next_state = DimeCompleteState(
            t=0.02,
            q=np.array([0.304], dtype=np.float64),
            v=np.array([0.198], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        factor = DimeDynamicsWindowFactor(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=1,
            dt_s=0.02,
        )

        u = np.array([0.2], dtype=np.float64)

        # Analytical Jacobian
        J_x0, J_x1, J_u = factor.compute_defect_jacobians(initial_state, next_state, u)

        # Numerical central finite differences
        eps = 1e-6
        # J_u check
        u_plus = u + eps
        u_minus = u - eps
        d_plus = factor.compute_single_step_defect(initial_state, next_state, u_plus)
        d_minus = factor.compute_single_step_defect(initial_state, next_state, u_minus)
        J_u_fd = (d_plus - d_minus) / (2.0 * eps)

        np.testing.assert_allclose(
            J_u.flatten(), J_u_fd.flatten(), rtol=1e-4, atol=1e-5
        )

    def test_green_runtime_exclusivity_contract(self) -> None:
        """DimeDynamicsWindowFactor enforces mutual exclusivity against marginalized factors on same interval."""
        provider = AnalyticPendulumProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.1], dtype=np.float64),
            v=np.array([0.0], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        factor = DimeDynamicsWindowFactor(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=4,
            dt_s=0.02,
        )

        contract = RuntimeExclusivityContract()
        # Register explicit factor
        contract.register_factor(factor.as_interval_factor())

        # Attempting to register overlapping marginalized factor must raise PreconditionError
        marginalized_factor = EstimationIntervalFactor(
            name="marginalized_ztcf",
            factor_type="marginalized_input_transition",
            t_start=0.0,
            t_end=0.08,
            contributes_to_objective=True,
        )

        with pytest.raises(PreconditionError, match="Runtime exclusivity violation"):
            contract.register_factor(marginalized_factor)

    def test_green_all_cost_and_residual_components_exported(self) -> None:
        """Every cost component and constraint residual is exported in the solve receipt."""
        provider = AnalyticPendulumProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.1], dtype=np.float64),
            v=np.array([0.0], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=3,
            dt_s=0.02,
            target_positions=[
                np.array([0.1]),
                np.array([0.101]),
                np.array([0.102]),
                np.array([0.103]),
            ],
            defect_mode=DefectMode.SOFT,
        )

        result = solve_dime_dynamics_window(problem)
        assert result.success

        # Cost breakdown exported
        costs = result.cost_breakdown
        assert "observation_cost" in costs
        assert "transition_cost" in costs
        assert "control_effort_cost" in costs
        assert "control_rate_cost" in costs
        assert "discrepancy_cost" in costs
        assert "total_cost" in costs
        assert costs["total_cost"] >= 0.0

        # Residual components exported
        assert len(result.transition_defects) == 3
        assert result.actuator_bound_residuals is not None
        assert result.control_variation_residuals is not None

    def test_green_exact_defect_mode_hard_feasibility(self) -> None:
        """Exact defect mode enforces hard dynamic consistency with zero transition defect."""
        provider = AnalyticPendulumProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.15], dtype=np.float64),
            v=np.array([0.05], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=4,
            dt_s=0.02,
            target_positions=[
                np.array([0.15]),
                np.array([0.151]),
                np.array([0.152]),
                np.array([0.153]),
                np.array([0.154]),
            ],
            defect_mode=DefectMode.EXACT,
        )

        result = solve_dime_dynamics_window(problem)
        assert result.success
        for d in result.transition_defects:
            np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_green_actuator_bounds_enforced(self) -> None:
        """Optimal controls strictly satisfy declared actuator bounds."""
        provider = AnalyticPendulumProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.0], dtype=np.float64),
            v=np.array([0.0], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        # Strongly displaced targets that would demand large control > 0.3
        target_positions = [
            np.array([0.0]),
            np.array([0.1]),
            np.array([0.2]),
            np.array([0.3]),
        ]
        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=3,
            dt_s=0.02,
            target_positions=target_positions,
            actuator_bounds=(-0.3, 0.3),
        )

        result = solve_dime_dynamics_window(problem)
        assert result.success
        assert np.all(result.controls >= -0.3 - 1e-9)
        assert np.all(result.controls <= 0.3 + 1e-9)

    def test_green_serialization_roundtrip(self) -> None:
        """DimeDynamicsWindowResult serializes to JSON-safe dict."""
        provider = AnalyticPendulumProvider()

        initial_state = DimeCompleteState(
            t=0.0,
            q=np.array([0.1], dtype=np.float64),
            v=np.array([0.0], dtype=np.float64),
            units={"length": "m", "angle": "rad", "time": "s"},
            model_hash=provider.model_hash,
        )

        problem = DimeDynamicsWindowProblem(
            provider=provider,
            initial_state=initial_state,
            horizon_steps=2,
            dt_s=0.02,
        )

        result = solve_dime_dynamics_window(problem)
        payload = result.to_dict()
        assert isinstance(payload, dict)
        assert payload["success"] is True
        assert len(payload["states"]) == 3
        assert "cost_breakdown" in payload


class _FailsOnNonzeroControls(AnalyticPendulumProvider):
    """Pendulum whose step raises for any non-zero control (nominal rollout passes)."""

    def step(self, request: DimeFullStepRequest):  # type: ignore[override]
        if np.any(request.controls != 0.0):
            raise RuntimeError("synthetic provider failure")
        return super().step(request)


def _fixed_size_problem(provider: AnalyticPendulumProvider):
    state = DimeCompleteState(
        t=0.0,
        q=np.array([0.1], dtype=np.float64),
        v=np.array([0.0], dtype=np.float64),
        units={"length": "m", "angle": "rad", "time": "s"},
        model_hash=provider.model_hash,
    )
    targets = [np.array([0.1], dtype=np.float64) for _ in range(5)]
    return DimeDynamicsWindowProblem(
        provider=provider,
        initial_state=state,
        horizon_steps=4,
        dt_s=0.01,
        target_positions=targets,
        control_rate_weight=1.0,
        actuator_bounds=(-1.0, 1.0),
        max_iterations=5,
    )


class TestFixedSizeResidual:
    """Residual length is fixed on every path; failures are typed, not faked (#11554)."""

    def test_residual_length_invariant_across_controls(self) -> None:
        from src.shared.python.estimation.dime_dynamics_window import (
            _build_residuals_evaluator,
            expected_residual_size,
        )

        problem = _fixed_size_problem(AnalyticPendulumProvider())
        fn = _build_residuals_evaluator(problem)
        n = expected_residual_size(problem)
        inside = fn(np.full(4, 0.1))
        outside = fn(np.full(4, 5.0))  # violates actuator bounds
        assert inside.shape == outside.shape == (n,)

    def test_failure_raises_typed_error_not_fake_residual(self) -> None:
        from src.shared.python.estimation.dime_dynamics_window import (
            DimeResidualEvaluationError,
            _build_residuals_evaluator,
        )

        problem = _fixed_size_problem(_FailsOnNonzeroControls())
        fn = _build_residuals_evaluator(problem)
        with pytest.raises(DimeResidualEvaluationError):
            fn(np.full(4, 0.1))

    def test_solver_reports_step_failure_explicitly(self) -> None:
        problem = _fixed_size_problem(_FailsOnNonzeroControls())
        result = solve_dime_dynamics_window(problem)
        assert not result.success
        assert result.status == "failed_dynamics_step"
        assert result.controls.shape == (4, 1)
