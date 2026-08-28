"""Serial execution contracts for the structural-factorial study."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    NativeEngineUnavailable,
    StructuralCase,
    build_launch_manifest,
    build_registered_cases,
    load_registered_checkpoints,
    run_serial_cases,
)

PLAN_REVISION = "a" * 40
EXECUTION_REVISION = "b" * 40
HASHES = {
    "closed_state_npz": "1" * 64,
    "shaft_atlas_json": "2" * 64,
    "shaft_atlas_npz": "3" * 64,
    "ground_atlas_json": "4" * 64,
    "ground_atlas_npz": "5" * 64,
}
pytestmark = pytest.mark.scientific


def _plan() -> dict[str, object]:
    return StructuralFactorialPlan(
        design_authority_revision=PLAN_REVISION,
        authority_sha256=HASHES,
    ).to_manifest()


def _launch(plan: dict[str, object]) -> dict[str, object]:
    return build_launch_manifest(plan=plan, execution_revision=EXECUTION_REVISION)


def _tiny_plan() -> dict[str, object]:
    """Retain the same schema while bounding checkpoint-I/O unit tests."""

    plan = _plan()
    design = dict(plan["design"])  # type: ignore[arg-type]
    design["states"] = design["states"][:1]
    design["factorial_cells"] = design["factorial_cells"][:2]
    design["velocity_factors"] = design["velocity_factors"][:1]
    design["time_steps_s"] = design["time_steps_s"][:1]
    design["registered_engine_attempt_count"] = 4
    design["expected_native_attempt_count"] = 2
    plan["design"] = design
    return plan


def test_runner_expands_all_cells_in_stable_order() -> None:
    cases = build_registered_cases(_plan())

    assert len(cases) == 2304
    assert len({case.case_key for case in cases}) == 2304
    assert cases[0] == StructuralCase(
        source_case_index=0,
        source_sample_index=0,
        source_time_s=0.0,
        cell_id="0000",
        shaft_activation="rigid",
        ground_activation="fixed",
        velocity_factor=1.0,
        engine="mujoco",
        time_step_s=0.0002,
        case_key="state=case0-sample0/cell=0000/v=+1/mujoco/dt=0.0002",
    )
    assert cases[-1].case_key == (
        "state=case17-sample12/cell=1111/v=-1/pinocchio/dt=5e-05"
    )


def test_serial_runner_is_atomic_resumable_and_identity_bound(tmp_path: Path) -> None:
    plan = _tiny_plan()
    launch = _launch(plan)
    calls: list[str] = []

    def evaluate(case: StructuralCase) -> dict[str, object]:
        calls.append(case.case_key)
        return {"horizons": [{"horizon_s": 0.05, "retained": True}]}

    first = run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
        evaluator=evaluate,
    )
    second = run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
        evaluator=evaluate,
    )

    assert len(first) == len(second) == 4
    assert len(calls) == 4
    assert all(checkpoint.status == "completed" for checkpoint in first)
    assert all(checkpoint.resumed for checkpoint in second)
    assert not tuple(tmp_path.glob("*.tmp"))
    payload = json.loads(first[0].path.read_text(encoding="utf-8"))
    assert payload["identity"]["execution_revision"] == EXECUTION_REVISION
    assert payload["case"]["cell_id"] == "0000"


def test_native_absence_is_typed_and_unexpected_failure_is_not_retained(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    launch = _launch(plan)

    def unavailable(case: StructuralCase) -> dict[str, object]:
        raise NativeEngineUnavailable(engine=case.engine, detail="not installed")

    checkpoints = run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path / "unavailable",
        evaluator=unavailable,
    )
    assert all(checkpoint.status == "unavailable" for checkpoint in checkpoints)

    def broken(_case: StructuralCase) -> dict[str, object]:
        raise RuntimeError("planted integration failure")

    with pytest.raises(RuntimeError, match="planted integration failure"):
        run_serial_cases(
            plan=plan,
            launch=launch,
            checkpoint_dir=tmp_path / "broken",
            evaluator=broken,
        )
    assert not tuple((tmp_path / "broken").glob("*.json"))


def test_launch_and_checkpoint_loading_fail_closed_on_identity_drift(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    launch = _launch(plan)
    run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=tmp_path,
        evaluator=lambda _case: {"retained": True},
    )

    drifted = dict(launch)
    drifted["execution_revision"] = "c" * 40
    with pytest.raises(ValueError, match="launch identity"):
        load_registered_checkpoints(
            plan=plan,
            launch=drifted,
            checkpoint_dir=tmp_path,
        )


def test_launch_refuses_parallel_or_outcome_matched_design() -> None:
    plan = _plan()
    plan["execution"] = {**plan["execution"], "worker_count": 2}  # type: ignore[index]
    with pytest.raises(ValueError, match="worker_count"):
        build_launch_manifest(plan=plan, execution_revision=EXECUTION_REVISION)

    plan = _plan()
    plan["analysis"] = {**plan["analysis"], "outcome_matching": "allowed"}  # type: ignore[index]
    with pytest.raises(ValueError, match="outcome matching"):
        build_launch_manifest(plan=plan, execution_revision=EXECUTION_REVISION)
