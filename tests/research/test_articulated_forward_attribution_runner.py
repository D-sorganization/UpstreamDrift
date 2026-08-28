"""Atomic serial execution contract for the #9153 attribution study."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    NativeEngineUnavailable,
    StudyCase,
    build_registered_cases,
    load_registered_checkpoints,
    run_serial_cases,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_study import (
    ForwardAttributionStudyPlan,
)


SOURCE_REVISION = "a" * 40
SOURCE_DATA_SHA256 = "b" * 64
EXECUTION_REVISION = "c" * 40


def _manifest() -> dict[str, object]:
    return ForwardAttributionStudyPlan(
        source_revision=SOURCE_REVISION,
        source_data_sha256=SOURCE_DATA_SHA256,
    ).to_manifest()


def test_registered_cases_are_complete_unique_and_stably_ordered() -> None:
    cases = build_registered_cases(_manifest())

    assert len(cases) == 42
    assert len({case.case_key for case in cases}) == 42
    assert cases[0].case_key == "state=case4-sample6/mujoco/nominal/dt=0.001"
    assert cases[-1].case_key == (
        "state=case4-sample6/pinocchio/zero_preload/dt=0.00025"
    )
    assert cases[0].source_case_index == 4
    assert cases[0].source_sample_index == 6


def test_serial_runner_writes_atomic_checkpoints_and_resumes(tmp_path: Path) -> None:
    calls: list[str] = []

    def evaluate(case: StudyCase) -> dict[str, object]:
        calls.append(case.case_key)
        return {"momentum_relative_residual": 0.001, "retained": True}

    first = run_serial_cases(
        manifest=_manifest(),
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
        evaluator=evaluate,
    )
    second = run_serial_cases(
        manifest=_manifest(),
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
        evaluator=evaluate,
    )

    assert len(first) == len(second) == 42
    assert len(calls) == 42
    assert all(item.status == "completed" for item in first)
    assert not tuple(tmp_path.glob("*.tmp"))
    payload = json.loads(first[0].path.read_text(encoding="utf-8"))
    assert payload["identity"]["execution_revision"] == EXECUTION_REVISION
    assert payload["identity"]["case_key"] == first[0].case.case_key
    assert payload["outcome"]["status"] == "completed"
    loaded = load_registered_checkpoints(
        manifest=_manifest(),
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
    )
    assert len(loaded) == 42
    assert all(item.resumed for item in loaded)


def test_native_unavailability_is_typed_and_retained(tmp_path: Path) -> None:
    def evaluate(case: StudyCase) -> dict[str, object]:
        if case.engine == "pinocchio":
            raise NativeEngineUnavailable(
                engine="pinocchio",
                detail="qualified robotics Pinocchio is not installed",
            )
        return {"retained": True}

    checkpoints = run_serial_cases(
        manifest=_manifest(),
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
        evaluator=evaluate,
    )

    unavailable = [item for item in checkpoints if item.status == "unavailable"]
    assert len(unavailable) == 21
    payload = json.loads(unavailable[0].path.read_text(encoding="utf-8"))
    assert payload["outcome"] == {
        "status": "unavailable",
        "failure": {
            "code": "native_engine_unavailable",
            "detail": "qualified robotics Pinocchio is not installed",
            "engine": "pinocchio",
        },
    }


def test_resume_refuses_identity_drift(tmp_path: Path) -> None:
    run_serial_cases(
        manifest=_manifest(),
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
        evaluator=lambda _case: {"retained": True},
    )

    with pytest.raises(ValueError, match="execution_revision"):
        run_serial_cases(
            manifest=_manifest(),
            execution_revision="d" * 40,
            checkpoint_dir=tmp_path,
            evaluator=lambda _case: {"retained": True},
        )


def test_unexpected_evaluator_error_propagates_without_checkpoint(
    tmp_path: Path,
) -> None:
    def evaluate(_case: object) -> dict[str, object]:
        raise ValueError("planted evaluator corruption")

    with pytest.raises(ValueError, match="planted evaluator corruption"):
        run_serial_cases(
            manifest=_manifest(),
            execution_revision=EXECUTION_REVISION,
            checkpoint_dir=tmp_path,
            evaluator=evaluate,
        )

    assert not tuple(tmp_path.glob("case-*.json"))


def test_checkpoint_loader_fails_closed_on_incomplete_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="registered checkpoint is missing"):
        load_registered_checkpoints(
            manifest=_manifest(),
            execution_revision=EXECUTION_REVISION,
            checkpoint_dir=tmp_path,
        )


def test_checkpoint_loader_fails_closed_on_schema_drift(tmp_path: Path) -> None:
    checkpoints = run_serial_cases(
        manifest=_manifest(),
        execution_revision=EXECUTION_REVISION,
        checkpoint_dir=tmp_path,
        evaluator=lambda _case: {"retained": True},
    )
    path = checkpoints[0].path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint_schema_version"] = "unexpected"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint schema version"):
        load_registered_checkpoints(
            manifest=_manifest(),
            execution_revision=EXECUTION_REVISION,
            checkpoint_dir=tmp_path,
        )
