"""Synthetic-truth and independent-replay tests for the bounded F03 spike."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.sparse_collocation_spike import (
    BenchmarkBackend,
    BenchmarkGate,
    BenchmarkStart,
    SecondOrderFixture,
    SparseCollocationProblem,
    benchmark_backends,
    solve_existing_shooting_fixture,
    solve_sparse_collocation,
)

pytestmark = pytest.mark.unit


def _problem() -> SparseCollocationProblem:
    times = np.linspace(0.0, 0.5, 11)
    return SparseCollocationProblem(
        fixture=SecondOrderFixture(
            inertia_kg_m2=2.0, damping_nm_s_rad=0.3, stiffness_nm_rad=1.0
        ),
        times_s=times,
        initial_state=np.array([0.2, 0.0]),
        target_q_rad=np.zeros(len(times)),
        torque_lower_nm=-4.0,
        torque_upper_nm=4.0,
        rate_limit_nm_s=30.0,
        position_weight=10.0,
        effort_weight=0.01,
        rate_weight=0.001,
    )


def test_sparse_analytic_jacobian_matches_independent_difference() -> None:
    problem = _problem()
    candidate = problem.initial_guess(np.zeros(problem.intervals))
    direction = np.sin(np.arange(candidate.size, dtype=float) + 1.0)
    epsilon = 1e-6
    finite_difference = (
        problem.defects(candidate + epsilon * direction)
        - problem.defects(candidate - epsilon * direction)
    ) / (2 * epsilon)
    analytic = problem.defect_jacobian() @ direction
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-8, atol=1e-9)
    assert (
        problem.defect_jacobian().nnz
        < candidate.size * problem.defects(candidate).size / 3
    )


def test_collocation_solve_reports_fresh_zoh_replay_separately() -> None:
    problem = _problem()
    result = solve_sparse_collocation(
        problem, initial_torque=np.zeros(problem.intervals)
    )
    assert result.optimizer_converged
    assert result.max_dynamics_defect < 1e-7
    assert result.max_torque_violation == 0.0
    assert result.max_rate_violation == 0.0
    assert result.replay is not None
    assert result.replay.initial_state_equal
    assert result.replay.state_resets == 0
    assert result.replay.max_state_gap > 0.0  # Collocation nodes are not exact replay.
    assert result.replay.position_rmse_rad < 0.2
    assert result.objective >= 0.0


def test_coarse_collocation_does_not_pass_independent_replay_by_node_cost() -> None:
    problem = SparseCollocationProblem(
        fixture=SecondOrderFixture(1.0, 0.0, 80.0),
        times_s=np.array([0.0, 0.2, 0.4]),
        initial_state=np.array([0.2, 0.0]),
        target_q_rad=np.zeros(3),
        torque_lower_nm=-1.0,
        torque_upper_nm=1.0,
        rate_limit_nm_s=100.0,
        position_weight=1.0,
        effort_weight=0.01,
        rate_weight=0.0,
    )
    result = solve_sparse_collocation(problem, initial_torque=np.zeros(2))
    assert result.max_dynamics_defect < 1e-6
    assert result.replay is not None
    assert result.replay.max_state_gap > 0.05
    assert not result.replay_accepted(
        max_state_gap=0.05,
        max_constraint_defect=1e-6,
        max_bound_violation=1e-8,
    )


def test_rate_limit_and_invalid_contract_fail_closed() -> None:
    problem = _problem()
    result = solve_sparse_collocation(
        problem, initial_torque=np.full(problem.intervals, 3.0)
    )
    assert result.max_rate_violation < 1e-8
    with pytest.raises(ValueError, match="strictly increasing"):
        SparseCollocationProblem(
            fixture=problem.fixture,
            times_s=np.array([0.0, 0.0]),
            initial_state=problem.initial_state,
            target_q_rad=np.zeros(2),
            torque_lower_nm=-1.0,
            torque_upper_nm=1.0,
            rate_limit_nm_s=1.0,
            position_weight=1.0,
            effort_weight=0.1,
            rate_weight=0.1,
        )


def test_benchmark_keeps_predeclared_cold_warm_failures_and_costs_separate() -> None:
    problem = _problem()

    def fails(problem: SparseCollocationProblem, torque: np.ndarray):
        raise RuntimeError("infeasible start")

    report = benchmark_backends(
        problem,
        backends=(
            BenchmarkBackend(
                "sparse_implicit",
                lambda p, torque: solve_sparse_collocation(p, initial_torque=torque),
            ),
            BenchmarkBackend("failing_candidate", fails),
        ),
        starts=(
            BenchmarkStart("cold_zero", np.zeros(problem.intervals), warm=False),
            BenchmarkStart("warm_bounded", np.full(problem.intervals, 3.0), warm=True),
        ),
        gate=BenchmarkGate(
            max_replay_gap=0.01,
            max_observation_rmse_rad=0.2,
            max_constraint_defect_by_kind={
                "midpoint_inverse_dynamics": 1e-6,
                "shooting_state": 1e-6,
            },
            max_bound_violation=1e-8,
        ),
    )
    assert report.hardware.host
    assert len(report.attempts) == 4
    successes = [
        attempt for attempt in report.attempts if attempt.backend == "sparse_implicit"
    ]
    failures = [
        attempt for attempt in report.attempts if attempt.backend == "failing_candidate"
    ]
    assert all(attempt.optimizer_converged for attempt in successes)
    assert all(attempt.failure_reason == "infeasible start" for attempt in failures)
    assert all(attempt.total_seconds > 0.0 for attempt in report.attempts)
    assert all(attempt.solve_seconds is not None for attempt in successes)
    assert all(attempt.replay_seconds is not None for attempt in successes)
    assert all(attempt.peak_python_bytes >= 0 for attempt in report.attempts)
    assert report.summary("sparse_implicit", warm=False).p50_total_seconds > 0.0
    assert report.summary("sparse_implicit", warm=True).p95_total_seconds > 0.0
    assert report.summary("sparse_implicit", warm=False).p50_solve_seconds > 0.0
    assert report.summary("sparse_implicit", warm=True).p95_replay_seconds > 0.0
    assert report.summary("sparse_implicit", warm=True).p95_peak_python_bytes >= 0
    assert (
        report.summary("failing_candidate", warm=False).time_to_first_accepted_s is None
    )


def test_existing_multiple_shooting_runs_on_same_synthetic_truth_fixture() -> None:
    problem = _problem()
    shooting = solve_existing_shooting_fixture(
        problem, initial_torque=np.zeros(problem.intervals)
    )
    collocation = solve_sparse_collocation(
        problem, initial_torque=np.zeros(problem.intervals)
    )
    assert shooting.optimizer_converged
    assert shooting.replay is not None
    assert shooting.replay.state_resets == 0
    assert shooting.input_degrees_of_freedom == 1
    assert collocation.input_degrees_of_freedom == problem.intervals
    assert shooting.defect_kind == "shooting_state"
    assert collocation.defect_kind == "midpoint_inverse_dynamics"
    assert shooting.replay.position_rmse_rad > collocation.replay.position_rmse_rad
    report = benchmark_backends(
        problem,
        backends=(
            BenchmarkBackend(
                "existing_shooting",
                lambda p, torque: solve_existing_shooting_fixture(
                    p, initial_torque=torque
                ),
            ),
            BenchmarkBackend(
                "sparse_implicit",
                lambda p, torque: solve_sparse_collocation(p, initial_torque=torque),
            ),
        ),
        starts=(BenchmarkStart("cold_zero", np.zeros(problem.intervals), warm=False),),
        gate=BenchmarkGate(
            max_replay_gap=0.01,
            max_observation_rmse_rad=0.2,
            max_constraint_defect_by_kind={
                "midpoint_inverse_dynamics": 1e-6,
                "shooting_state": 1e-6,
            },
            max_bound_violation=1e-8,
        ),
    )
    assert {attempt.input_degrees_of_freedom for attempt in report.attempts} == {
        1,
        problem.intervals,
    }
    assert {attempt.defect_kind for attempt in report.attempts} == {
        "shooting_state",
        "midpoint_inverse_dynamics",
    }
