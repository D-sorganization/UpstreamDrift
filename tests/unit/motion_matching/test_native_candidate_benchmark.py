"""Independent analytic and native acceptance tests for F03b candidates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.native_candidate_benchmark import (
    NativeCandidateGate,
    benchmark_native_candidates,
    exact_held_solution,
    validate_candidate_native,
)
from src.shared.python.motion_matching.sparse_collocation_spike import (
    BenchmarkBackend,
    BenchmarkStart,
    SecondOrderFixture,
    SparseCollocationProblem,
    solve_existing_shooting_fixture,
    solve_sparse_collocation,
)

pytestmark = pytest.mark.unit


def _problem(*, coarse: bool = False) -> SparseCollocationProblem:
    times = np.array([0.0, 0.2, 0.4]) if coarse else np.linspace(0.0, 0.1, 11)
    fixture = (
        SecondOrderFixture(1.0, 0.0, 80.0)
        if coarse
        else SecondOrderFixture(2.0, 0.3, 1.0)
    )
    return SparseCollocationProblem(
        fixture=fixture,
        times_s=times,
        initial_state=np.array([0.2, 0.0]),
        target_q_rad=np.zeros(len(times)),
        torque_lower_nm=-1.0 if coarse else -4.0,
        torque_upper_nm=1.0 if coarse else 4.0,
        rate_limit_nm_s=100.0,
        position_weight=1.0,
        effort_weight=0.01,
        rate_weight=0.0,
    )


def _gate(*, gap: float = 0.01) -> NativeCandidateGate:
    return NativeCandidateGate(
        max_native_node_gap=gap,
        max_native_exact_gap=1e-5,
        max_observation_rmse_rad=0.3,
        max_torque_violation_nm=1e-8,
        max_rate_violation_nm_s=1e-8,
        max_constraint_defect_by_kind={
            "midpoint_inverse_dynamics": 1e-6,
            "shooting_state": 1e-6,
        },
    )


def test_exact_held_truth_matches_closed_form_double_integrator() -> None:
    fixture = SecondOrderFixture(2.0, 0.0, 0.0)
    states = exact_held_solution(
        fixture,
        np.array([0.0, 0.1, 0.2]),
        np.array([0.2, 0.1]),
        np.array([0.5, -0.5]),
    )
    np.testing.assert_allclose(states[1], [0.21125, 0.125], atol=1e-14)
    np.testing.assert_allclose(states[2], [0.2225, 0.1], atol=1e-14)


def test_native_held_candidate_matches_exact_dynamics_and_binds_input(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    problem = _problem()
    candidate = solve_sparse_collocation(
        problem, initial_torque=np.zeros(problem.intervals)
    )
    receipt = validate_candidate_native(
        problem,
        candidate,
        model_path=tmp_path / "rotary.xml",
        substeps=4,
        gate=_gate(),
    )
    assert receipt.accepted
    assert receipt.max_native_exact_gap < 1e-5
    assert receipt.max_native_node_gap < 0.01
    assert receipt.input_sha256 and receipt.policy_sha256 and receipt.model_sha256
    assert receipt.applied_torque_rows == problem.intervals * 4
    assert receipt.native_step_seconds == pytest.approx(0.0025)
    assert receipt.native_replay_mode == "native_own_contact"
    assert receipt.native_contact_present is False


def test_coarse_midpoint_nodes_cannot_pass_native_rollout(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    problem = _problem(coarse=True)
    candidate = solve_sparse_collocation(
        problem, initial_torque=np.zeros(problem.intervals)
    )
    assert candidate.max_dynamics_defect < 1e-6
    receipt = validate_candidate_native(
        problem,
        candidate,
        model_path=tmp_path / "coarse.xml",
        substeps=20,
        gate=_gate(gap=0.05),
    )
    assert receipt.max_native_exact_gap < 1e-5
    assert receipt.max_native_node_gap > 0.05
    assert not receipt.accepted
    assert receipt.failure_reason == "native_node_gap"


def test_invalid_refinement_is_rejected_before_native_export(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    problem = _problem()
    candidate = solve_sparse_collocation(
        problem, initial_torque=np.zeros(problem.intervals)
    )
    with pytest.raises(ValueError, match="positive"):
        validate_candidate_native(
            problem,
            candidate,
            model_path=tmp_path / "bad.xml",
            substeps=0,
            gate=_gate(),
        )


def test_native_refinement_reduces_integrator_error_without_changing_held_input(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    problem = _problem(coarse=True)
    candidate = solve_sparse_collocation(
        problem, initial_torque=np.zeros(problem.intervals)
    )
    coarse = validate_candidate_native(
        problem,
        candidate,
        model_path=tmp_path / "native-coarse.xml",
        substeps=1,
        gate=_gate(gap=1.0),
    )
    refined = validate_candidate_native(
        problem,
        candidate,
        model_path=tmp_path / "native-refined.xml",
        substeps=20,
        gate=_gate(gap=1.0),
    )
    assert refined.max_native_exact_gap < coarse.max_native_exact_gap
    assert refined.input_sha256 != coarse.input_sha256  # Different executed grids.
    assert refined.applied_torque_rows == 20 * coarse.applied_torque_rows


def test_existing_shooting_candidate_uses_same_native_validator(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    problem = _problem()
    shooting = solve_existing_shooting_fixture(
        problem, initial_torque=np.zeros(problem.intervals)
    )
    receipt = validate_candidate_native(
        problem,
        shooting,
        model_path=tmp_path / "shooting.xml",
        substeps=4,
        gate=_gate(),
    )
    assert receipt.defect_kind == "shooting_state"
    assert receipt.max_native_exact_gap < 1e-5
    assert receipt.max_native_node_gap >= 0
    assert receipt.input_sha256


def test_total_accepted_cost_includes_failed_start_and_export(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    problem = _problem()

    def solver(p: SparseCollocationProblem, torque: np.ndarray):
        if np.any(torque):
            raise RuntimeError("declared failed start")
        return solve_sparse_collocation(p, initial_torque=torque)

    report = benchmark_native_candidates(
        problem,
        backends=(BenchmarkBackend("sparse", solver),),
        starts=(
            BenchmarkStart("failed", np.ones(problem.intervals), warm=False),
            BenchmarkStart("accepted", np.zeros(problem.intervals), warm=True),
        ),
        gate=_gate(),
        substeps=4,
        output_dir=tmp_path,
    )
    first, second = report.attempts
    assert first.failure_reason == "declared failed start"
    assert not first.accepted and second.accepted
    assert first.solve_seconds > 0 and first.native_seconds is None
    assert second.solve_seconds > 0 and second.native_seconds > 0
    assert first.export_seconds > 0 and second.export_seconds > 0
    assert first.receipt_path.is_file() and second.receipt_path.is_file()
    exported = json.loads(second.receipt_path.read_text(encoding="utf-8"))
    assert exported["hardware"]["host"] == report.hardware.host
    assert (
        report.time_to_first_accepted_s("sparse")
        >= sum(attempt.total_seconds for attempt in report.attempts) - 1e-6
    )
    assert report.summary("sparse", warm=True).p95_total_seconds > 0


def test_native_gate_recomputes_rate_limit_from_executed_torque(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    problem = _problem()
    candidate = solve_sparse_collocation(
        problem, initial_torque=np.zeros(problem.intervals)
    )
    alternating = np.array([4.0, -4.0] * (problem.intervals // 2))
    dishonest = replace(
        candidate,
        torque_nm=alternating,
        max_torque_violation=0.0,
        max_rate_violation=0.0,
    )
    receipt = validate_candidate_native(
        problem,
        dishonest,
        model_path=tmp_path / "rate.xml",
        substeps=4,
        gate=_gate(gap=100.0),
    )
    assert not receipt.accepted
    assert receipt.failure_reason == "rate_bound"
    assert receipt.max_rate_violation_nm_s > 0
