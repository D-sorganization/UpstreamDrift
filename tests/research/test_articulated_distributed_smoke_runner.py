"""Tests for the distributed event-attribution smoke study runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_distributed_smoke_registration import (
    EVALUATOR_REVISION,
    REGISTRATION_PATH,
    build_registration,
    registered_smoke_cases,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    require_robotics_pinocchio,
)
from scripts.research.proximal_distal_energy.run_articulated_distributed_smoke import (
    RESULTS_PATH,
    CaseExecutionResult,
    DistributedSmokeRunner,
    DistributedSmokeSummary,
    execute_smoke_case,
    qualify_smoke_results,
    run_distributed_smoke,
)


pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]


def _robotics_pinocchio_available() -> bool:
    try:
        import pinocchio as pin

        require_robotics_pinocchio(pin)
    except (ImportError, RuntimeError):
        return False
    return True


requires_native_pinocchio = pytest.mark.skipif(
    not _robotics_pinocchio_available(),
    reason="robotics Pinocchio is exercised in the optional-stack CI lane",
)


def test_smoke_runner_verifies_registration_identity() -> None:
    runner = DistributedSmokeRunner(root=ROOT)
    registration = runner.load_and_validate_registration()
    assert registration["evaluator_authority"]["revision"] == EVALUATOR_REVISION
    assert registration["execution_status"] == "not_started"
    assert len(registration["study_design"]["cases"]) == 6


def test_smoke_runner_rejects_corrupted_registration(tmp_path: Path) -> None:
    runner = DistributedSmokeRunner(root=ROOT)
    reg = build_registration(ROOT)
    reg["evaluator_authority"]["revision"] = "0000000000000000000000000000000000000000"
    corrupt_file = tmp_path / "corrupt_registration.json"
    corrupt_file.write_text(json.dumps(reg), encoding="utf-8")

    with pytest.raises(
        ValueError, match="registration differs from deterministic authority"
    ):
        runner.load_and_validate_registration(corrupt_file)


def test_smoke_runner_enumerates_all_six_cases() -> None:
    runner = DistributedSmokeRunner(root=ROOT)
    cases = runner.cases
    assert len(cases) == 6
    case_ids = [c["case_id"] for c in cases]
    assert case_ids == [
        "event_probe_mujoco_dt_1000us",
        "event_probe_mujoco_dt_0500us",
        "event_probe_mujoco_dt_0250us",
        "event_probe_pinocchio_dt_1000us",
        "event_probe_pinocchio_dt_0500us",
        "event_probe_pinocchio_dt_0250us",
    ]


def test_smoke_runner_enforces_thread_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = DistributedSmokeRunner(root=ROOT)
    runner.enforce_thread_limits()
    import os

    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert os.environ.get("MKL_NUM_THREADS") == "1"


def test_execute_single_smoke_case_mujoco() -> None:
    cases = registered_smoke_cases(build_registration(ROOT), ROOT)
    case_0 = cases[0]
    result = execute_smoke_case(case_0, root=ROOT)

    assert isinstance(result, CaseExecutionResult)
    assert result.status == "completed"
    assert result.case_id == "event_probe_mujoco_dt_1000us"
    assert result.engine == "mujoco"
    assert result.time_step_s == 0.001
    assert result.event_count >= 2
    assert "opening" in result.event_kinds
    assert "reattachment" in result.event_kinds
    assert result.maximum_absolute_gap_residual_m <= 1.0e-10
    assert result.maximum_final_bracket_width_s <= 1.0e-12
    assert result.maximum_force_closure_residual <= 1.0e-12
    assert result.total_discrete_event_impulse == pytest.approx(0.0, abs=1.0e-15)
    assert result.total_discrete_event_work_j == pytest.approx(0.0, abs=1.0e-15)


@requires_native_pinocchio
def test_execute_single_smoke_case_pinocchio() -> None:
    cases = registered_smoke_cases(build_registration(ROOT), ROOT)
    case_3 = cases[3]
    result = execute_smoke_case(case_3, root=ROOT)

    assert isinstance(result, CaseExecutionResult)
    assert result.status == "completed"
    assert result.case_id == "event_probe_pinocchio_dt_1000us"
    assert result.engine == "pinocchio"
    assert result.time_step_s == 0.001
    assert result.event_count >= 2
    assert "opening" in result.event_kinds
    assert "reattachment" in result.event_kinds
    assert result.maximum_absolute_gap_residual_m <= 1.0e-10
    assert result.maximum_final_bracket_width_s <= 1.0e-12
    assert result.maximum_force_closure_residual <= 1.0e-12
    assert result.total_discrete_event_impulse == pytest.approx(0.0, abs=1.0e-15)
    assert result.total_discrete_event_work_j == pytest.approx(0.0, abs=1.0e-15)


def test_smoke_runner_atomic_checkpoint_and_resume(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    runner = DistributedSmokeRunner(root=ROOT, checkpoint_dir=checkpoint_dir)

    # Use mujoco cases that can run in any environment
    cases = [c for c in runner.cases if c["engine"] == "mujoco"][:2]
    res1 = runner.run_cases(cases=cases)
    assert len(res1.case_results) == 2
    assert (checkpoint_dir / "event_probe_mujoco_dt_1000us.json").exists()
    assert (checkpoint_dir / "event_probe_mujoco_dt_0500us.json").exists()

    # Second run should resume and load from existing checkpoints
    res2 = runner.run_cases(cases=cases)
    assert len(res2.case_results) == 2
    assert res2.case_results[0].case_id == res1.case_results[0].case_id


def test_smoke_runner_handles_typed_failure() -> None:
    bad_case = {
        "case_id": "case_bad_test",
        "engine": "unknown_engine",
        "time_step_s": 0.001,
        "source_case_index": 0,
        "source_sample_index": 6,
        "initial_club_displacement_m": 0.001,
        "initial_club_velocity_m_s": -0.8,
        "slack_distance_m": 0.0015,
        "checkpoint_policy": "atomic_per_case",
    }
    result = execute_smoke_case(bad_case, root=ROOT)
    assert result.status == "failed"
    assert result.error_type is not None
    assert (
        "unknown_engine" in (result.error_message or "").lower()
        or "engine" in (result.error_message or "").lower()
    )


def test_qualify_smoke_results_acceptance_gates() -> None:
    cases = registered_smoke_cases(build_registration(ROOT), ROOT)
    # Execute mujoco cases
    mujoco_cases = [c for c in cases if c["engine"] == "mujoco"]
    mujoco_results = [execute_smoke_case(c, root=ROOT) for c in mujoco_cases]

    # Synthesize corresponding pinocchio results if pinocchio is not available in local test env
    if _robotics_pinocchio_available():
        pin_cases = [c for c in cases if c["engine"] == "pinocchio"]
        pin_results = [execute_smoke_case(c, root=ROOT) for c in pin_cases]
    else:
        # Clone matching mujoco case execution results for qualifying acceptance gate logic
        pin_results = [
            CaseExecutionResult(
                case_id=c["case_id"],
                engine="pinocchio",
                time_step_s=c["time_step_s"],
                status="completed",
                event_count=mujoco_results[i].event_count,
                event_kinds=mujoco_results[i].event_kinds,
                events_detail=mujoco_results[i].events_detail,
                maximum_absolute_gap_residual_m=mujoco_results[
                    i
                ].maximum_absolute_gap_residual_m,
                maximum_final_bracket_width_s=mujoco_results[
                    i
                ].maximum_final_bracket_width_s,
                maximum_force_closure_residual=mujoco_results[
                    i
                ].maximum_force_closure_residual,
                total_discrete_event_impulse=mujoco_results[
                    i
                ].total_discrete_event_impulse,
                total_discrete_event_work_j=mujoco_results[
                    i
                ].total_discrete_event_work_j,
                momentum_change=mujoco_results[i].momentum_change,
                kinetic_energy_change_j=mujoco_results[i].kinetic_energy_change_j,
                continuous_work_j=mujoco_results[i].continuous_work_j,
                work_closure_residual_j=mujoco_results[i].work_closure_residual_j,
                generalized_work_j=mujoco_results[i].generalized_work_j,
                impulse_shares=mujoco_results[i].impulse_shares,
                work_shares=mujoco_results[i].work_shares,
            )
            for i, c in enumerate([c for c in cases if c["engine"] == "pinocchio"])
        ]

    summary = DistributedSmokeSummary(
        registration_path=REGISTRATION_PATH.relative_to(ROOT).as_posix(),
        evaluator_revision=EVALUATOR_REVISION,
        case_results=mujoco_results + pin_results,
    )
    qualification = qualify_smoke_results(summary, root=ROOT)

    assert qualification["qualified"] is True
    assert qualification["case_count"] == 6
    assert qualification["completed_count"] == 6
    assert qualification["failed_count"] == 0
    assert qualification["all_events_bracketed"] is True
    assert qualification["all_force_closures_pass"] is True
    assert qualification["all_discrete_impulses_zero"] is True
    assert qualification["all_discrete_works_zero"] is True
    assert qualification["promotion_eligible"] is False
