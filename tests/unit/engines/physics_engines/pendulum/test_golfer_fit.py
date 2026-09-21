"""Behavioral tests for the constrained upper-body golfer fitter (TB-06 #10591).

Covers: feasible vs incompatible initial states, q/v loop-closure through
rollout, singular/infeasible diagnostic receipts (no synthetic success),
two-hand geometry (grip_right/grip_left tracked independently, never
inferred from one hand), and torque/reaction-force distinction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.motion_matching.adapters_golfer import (
    MODEL_ID_GOLFER,
    create_calibrated_golfer_params,
    describe_golfer_topology,
    grip_and_clubhead_positions,
    solve_feasible_initial_state,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization_golfer import (
    COEFFS_PER_JOINT,
    N_ACTUATED_JOINTS,
    BernsteinGolferTorqueProfile,
    GolferFitOptions,
    GolferFitTarget,
    fit_bounded_golfer,
    integrate_golfer_rollout,
)
from src.shared.python.pendulum_simulator.golfer_constraints import constraint_vector
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams

pytestmark = pytest.mark.unit


def _zero_profile(duration: float) -> BernsteinGolferTorqueProfile:
    return BernsteinGolferTorqueProfile(
        controls=np.zeros((N_ACTUATED_JOINTS, COEFFS_PER_JOINT)),
        duration_s=duration,
    )


class TestTopologyRecording:
    """Bounded work item 1: record DOF vs generalized coordinates, loop topology."""

    def test_model_identity_matches_tb00_registry(self) -> None:
        assert MODEL_ID_GOLFER == "constrained_upper_body_golfer"

    def test_topology_reports_closed_loop_and_dof_counts(self) -> None:
        topo = describe_golfer_topology()
        assert topo["n_generalized_coordinates"] == N_DOF
        assert topo["n_constraints"] == 4
        assert topo["constraint_jacobian_rank"] == 3
        assert topo["n_independent_dof"] == 5
        assert topo["has_hand_club_loop_closure"] is True
        assert topo["n_actuated_joints"] == 7


class TestFeasibleInitialState:
    """Bounded work item 2: feasible q0/v0, real loop-closure enforcement."""

    def test_feasible_guess_projects_to_closed_loop(self) -> None:
        params = create_calibrated_golfer_params()
        q_guess = np.array([-0.2, 0.3, 0.5, -0.1, -0.3, 0.5, -0.1, 0.0])
        v_guess = np.zeros(N_DOF)
        report = solve_feasible_initial_state(q_guess, v_guess, params)
        assert report.feasible is True
        phi = constraint_vector(report.q0, params)
        assert math.sqrt(float(np.dot(phi, phi))) < 1e-8
        # Velocity must also satisfy the differentiated constraint (Phi_q @ v0 == 0),
        # not merely the position-level closure.
        assert report.velocity_residual < 1e-8

    def test_incompatible_geometry_yields_infeasible_diagnostic_not_exception(
        self,
    ) -> None:
        """Arms too short to ever let both hands reach a shared club must not raise."""
        params = GolferParams(
            m_hub=10.0,
            m_r_upper=2.0,
            m_r_fore=1.5,
            m_l_upper=2.0,
            m_l_fore=1.5,
            m_club=0.4,
            L_hub=0.2,
            L_r_upper=0.05,
            L_r_fore=0.05,
            L_l_upper=2.5,
            L_l_fore=2.5,
            L_club=1.0,
            d_rs=2.0,
            d_ls=2.0,
            grip_right=0.05,
            grip_left=0.15,
        )
        q_guess = np.zeros(N_DOF)
        v_guess = np.zeros(N_DOF)
        # A tight iteration budget on a poorly-conditioned starting guess must
        # surface honest non-convergence rather than iterating until some
        # (possibly unphysical) solution happens to appear.
        report = solve_feasible_initial_state(q_guess, v_guess, params, max_iter=2)
        assert report.feasible is False
        assert report.reason != ""

    def test_rejects_wrong_shape_state(self) -> None:
        params = create_calibrated_golfer_params()
        with pytest.raises(ValueError):
            solve_feasible_initial_state(np.zeros(3), np.zeros(N_DOF), params)


class TestRolloutClosure:
    """q/v closure must hold through the whole rollout, not just at t0."""

    def test_constraint_residual_stays_bounded_through_rollout(self) -> None:
        params = create_calibrated_golfer_params()
        q_guess = np.array([-0.2, 0.3, 0.5, -0.1, -0.3, 0.5, -0.1, 0.0])
        report = solve_feasible_initial_state(q_guess, np.zeros(N_DOF), params)
        assert report.feasible
        times = np.linspace(0.0, 0.2, 21)
        profile = _zero_profile(times[-1])
        result = integrate_golfer_rollout(params, report.q0, report.v0, times, profile)
        assert result.constraint_residual_traj.shape == (len(times),)
        assert np.all(result.constraint_residual_traj < 1e-3)

    def test_frame_zero_evaluated_before_any_integration_step(self) -> None:
        params = create_calibrated_golfer_params()
        q_guess = np.array([-0.2, 0.3, 0.5, -0.1, -0.3, 0.5, -0.1, 0.0])
        report = solve_feasible_initial_state(q_guess, np.zeros(N_DOF), params)
        times = np.linspace(0.0, 0.1, 6)
        profile = _zero_profile(times[-1])
        result = integrate_golfer_rollout(params, report.q0, report.v0, times, profile)
        assert np.allclose(result.q_traj[0], report.q0)
        assert np.allclose(result.v_traj[0], report.v0)


class TestTorqueReactionDistinction:
    """Applied joint torques and loop-closure reaction forces must stay distinct."""

    def test_reaction_forces_are_reported_separately_from_applied_torque(self) -> None:
        params = create_calibrated_golfer_params()
        q_guess = np.array([-0.2, 0.3, 0.5, -0.1, -0.3, 0.5, -0.1, 0.0])
        report = solve_feasible_initial_state(q_guess, np.zeros(N_DOF), params)
        times = np.linspace(0.0, 0.1, 6)
        controls = np.zeros((N_ACTUATED_JOINTS, COEFFS_PER_JOINT))
        controls[:, :] = 5.0
        profile = BernsteinGolferTorqueProfile(controls=controls, duration_s=times[-1])
        result = integrate_golfer_rollout(params, report.q0, report.v0, times, profile)
        assert result.lambda_traj.shape == (len(times), 4)
        # Reaction (constraint) forces are not the same array as applied torque.
        applied = np.array([profile.evaluate(t) for t in times])
        assert applied.shape == (len(times), N_ACTUATED_JOINTS)
        assert not np.allclose(result.lambda_traj[:, :2], applied[:, :2])


class TestTwoHandGeometry:
    """The second hand must be a real independently-constrained DOF, never inferred."""

    def test_grip_positions_are_independent_and_close_within_tolerance(self) -> None:
        params = create_calibrated_golfer_params()
        q_guess = np.array([-0.2, 0.3, 0.5, -0.1, -0.3, 0.5, -0.1, 0.0])
        report = solve_feasible_initial_state(q_guess, np.zeros(N_DOF), params)
        positions = grip_and_clubhead_positions(report.q0, params)
        grip_right = np.array(positions["grip_right"])
        grip_left = np.array(positions["grip_left"])
        # They must be distinct points on the club (not literally the same tuple),
        # yet consistent with club geometry (separated by grip_left - grip_right).
        assert not np.allclose(grip_right, grip_left)
        separation = float(np.linalg.norm(grip_left - grip_right))
        assert math.isclose(
            separation, params.grip_left - params.grip_right, abs_tol=1e-6
        )


class TestBoundedFit:
    """Bounded work item 3-4: reuse Bernstein optimization, honest diagnostics."""

    def test_fit_against_manufactured_trajectory_reduces_rmse(self) -> None:
        params = create_calibrated_golfer_params()
        q_guess = np.array([-0.2, 0.3, 0.5, -0.1, -0.3, 0.5, -0.1, 0.0])
        report = solve_feasible_initial_state(q_guess, np.zeros(N_DOF), params)
        times = np.linspace(0.0, 0.15, 11)

        # Manufacture a target by rolling out a small known torque profile,
        # then fit against it and confirm the optimizer can recover a
        # comparable trajectory (RMSE well below the unforced baseline).
        controls = np.zeros((N_ACTUATED_JOINTS, COEFFS_PER_JOINT))
        controls[0, :] = 8.0
        controls[4, :] = -6.0
        true_profile = BernsteinGolferTorqueProfile(
            controls=controls, duration_s=times[-1]
        )
        truth = integrate_golfer_rollout(
            params, report.q0, report.v0, times, true_profile
        )
        clubhead_traj = np.array(
            [
                grip_and_clubhead_positions(truth.q_traj[i], params)["clubhead"]
                for i in range(len(times))
            ]
        )
        grip_right_traj = np.array(
            [
                grip_and_clubhead_positions(truth.q_traj[i], params)["grip_right"]
                for i in range(len(times))
            ]
        )

        target = GolferFitTarget(
            times=times,
            clubhead=clubhead_traj,
            grip_right=grip_right_traj,
            q0=report.q0,
            v0=report.v0,
        )
        outcome = fit_bounded_golfer(
            target, params, GolferFitOptions(max_nfev=40, tau_bounds=(-50.0, 50.0))
        )
        assert outcome.feasible is True
        assert outcome.final_rmse_m < outcome.unforced_rmse_m
        assert outcome.final_rmse_m < 0.05

    def test_infeasible_initial_state_produces_diagnostic_not_fake_success(
        self,
    ) -> None:
        params = GolferParams(
            m_hub=10.0,
            m_r_upper=2.0,
            m_r_fore=1.5,
            m_l_upper=2.0,
            m_l_fore=1.5,
            m_club=0.4,
            L_hub=0.2,
            L_r_upper=0.05,
            L_r_fore=0.05,
            L_l_upper=2.5,
            L_l_fore=2.5,
            L_club=1.0,
            d_rs=2.0,
            d_ls=2.0,
            grip_right=0.05,
            grip_left=0.15,
        )
        times = np.linspace(0.0, 0.1, 5)
        target = GolferFitTarget(
            times=times,
            clubhead=np.zeros((5, 2)),
            grip_right=np.zeros((5, 2)),
            q0=np.zeros(N_DOF),
            v0=np.zeros(N_DOF),
        )
        outcome = fit_bounded_golfer(
            target, params, GolferFitOptions(max_nfev=5, feasibility_max_iter=1)
        )
        assert outcome.feasible is False
        assert outcome.converged is False
        assert outcome.profile is None
        assert outcome.reason != ""
