"""Unit tests for simulation run durable store (ADR 0001 / Epic #14085 / Issue #14087)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.models.responses import SimulationErrorInfo
from src.sim_run.models import Provenance, ResultEnvelope, RunRecord, RunRequest
from src.sim_run.store import (
    InvalidStateTransitionError,
    RunNotFoundError,
    SimulationRunStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def run_request() -> RunRequest:
    return RunRequest(
        engine_type="mujoco",
        duration=1.5,
        timestep=0.002,
        tags={"caller": "pytest", "env": "ci"},
    )


@pytest.fixture
def result_envelope() -> ResultEnvelope:
    return ResultEnvelope(
        run_id="placeholder",
        provenance=Provenance(
            repo_sha="abcdef1234567890",
            engine_type="mujoco",
            engine_version="3.2.0",
            input_sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            created_at="2026-10-29T23:00:00Z",
            completed_at="2026-10-29T23:00:02Z",
            host="desk-worker-1",
        ),
        validation_status="valid",
        validation_errors=[],
        success=True,
        data={"time": [0.0, 0.002], "qpos": [0.0, 0.1]},
        analysis_results={"max_speed": 45.2},
        artifacts=["trajectory.npz"],
    )


def test_create_and_get_run(tmp_path: Path, run_request: RunRequest) -> None:
    store = SimulationRunStore(results_root=tmp_path)
    record = store.create_run(run_request)
    assert record.run_id.startswith("run_")
    assert record.status == "queued"
    assert record.result is None
    assert (store.get_run_dir(record.run_id) / "record.json").is_file()
    assert store.get_run(record.run_id) == record


def test_custom_run_id_and_validation(tmp_path: Path, run_request: RunRequest) -> None:
    store = SimulationRunStore(results_root=tmp_path)
    rec = store.create_run(run_request, run_id="custom_run_1")
    assert rec.run_id == "custom_run_1"
    with pytest.raises(ValueError, match="Invalid run_id"):
        store.create_run(run_request, run_id="../escape_root")
    with pytest.raises(RunNotFoundError):
        store.get_run("run_missing")


def test_lifecycle_success_path(
    tmp_path: Path, run_request: RunRequest, result_envelope: ResultEnvelope
) -> None:
    store = SimulationRunStore(results_root=tmp_path)
    rec = store.create_run(run_request)
    assert store.start_run(rec.run_id).status == "running"
    result_envelope.run_id = rec.run_id
    succeeded = store.complete_run(rec.run_id, result=result_envelope)
    assert succeeded.status == "succeeded"
    assert succeeded.result is not None
    assert store.get_run(rec.run_id).status == "succeeded"


def test_lifecycle_failure_path(tmp_path: Path, run_request: RunRequest) -> None:
    store = SimulationRunStore(results_root=tmp_path)
    rec = store.create_run(run_request)
    store.start_run(rec.run_id)
    error = SimulationErrorInfo(
        code="DIVERGENCE", message="Diverged", stage="simulation", retriable=False
    )
    failed = store.fail_run(rec.run_id, error=error)
    assert failed.status == "failed"
    assert failed.error is not None and failed.error.code == "DIVERGENCE"
    assert store.get_run(rec.run_id).status == "failed"


def test_illegal_state_transition_succeeded_to_running_raises(
    tmp_path: Path, run_request: RunRequest, result_envelope: ResultEnvelope
) -> None:
    """Acceptance criterion: a succeeded->running transition raises."""
    store = SimulationRunStore(results_root=tmp_path)
    rec = store.create_run(run_request)
    store.start_run(rec.run_id)
    result_envelope.run_id = rec.run_id
    store.complete_run(rec.run_id, result=result_envelope)

    with pytest.raises(
        InvalidStateTransitionError,
        match="Illegal state transition from 'succeeded' to 'running'",
    ):
        store.start_run(rec.run_id)
    with pytest.raises(InvalidStateTransitionError):
        store.transition(rec.run_id, "queued")


def test_illegal_state_transition_from_expired_raises(
    tmp_path: Path, run_request: RunRequest
) -> None:
    store = SimulationRunStore(results_root=tmp_path)
    rec = store.create_run(run_request)
    store.prune_run(rec.run_id)
    for target in ("queued", "running", "succeeded", "failed"):
        with pytest.raises(InvalidStateTransitionError):
            store.transition(rec.run_id, target)  # type: ignore[arg-type]


def test_pruned_run_returns_status_expired_not_succeeded_with_empty_result(
    tmp_path: Path, run_request: RunRequest, result_envelope: ResultEnvelope
) -> None:
    """Acceptance criterion: pruned run returns status expired (not succeeded with empty result)."""
    store = SimulationRunStore(results_root=tmp_path)
    rec = store.create_run(run_request)
    store.start_run(rec.run_id)
    result_envelope.run_id = rec.run_id
    store.complete_run(rec.run_id, result=result_envelope)
    art = store.save_artifact(rec.run_id, "traj.bin", b"DATA")
    assert art.is_file()

    pruned = store.prune_run(rec.run_id, reason="quota")
    assert pruned.status == "expired"
    assert pruned.expires_at is not None
    assert pruned.result is None
    assert not art.exists()

    loaded = store.get_run(rec.run_id)
    assert loaded.status == "expired"
    assert loaded.status != "succeeded"
    assert loaded.result is None


def test_apply_quota_by_max_runs(
    tmp_path: Path, run_request: RunRequest, result_envelope: ResultEnvelope
) -> None:
    store = SimulationRunStore(results_root=tmp_path, max_runs=2)
    for rid in ("r1", "r2", "r3"):
        store.create_run(run_request, run_id=rid)
    pruned = store.apply_quota()
    assert pruned == ["r1"]
    assert store.get_run("r1").status == "expired"
    assert store.get_run("r2").status == "queued"
    assert store.get_run("r3").status == "queued"


def test_apply_quota_by_max_bytes(
    tmp_path: Path, run_request: RunRequest, result_envelope: ResultEnvelope
) -> None:
    store = SimulationRunStore(results_root=tmp_path)
    r1 = store.create_run(run_request, run_id="r1")
    store.start_run(r1.run_id)
    result_envelope.run_id = r1.run_id
    store.complete_run(r1.run_id, result=result_envelope)
    store.save_artifact(r1.run_id, "blob.bin", b"X" * 10_000)

    r2 = store.create_run(run_request, run_id="r2")
    store.start_run(r2.run_id)
    result_envelope.run_id = r2.run_id
    store.complete_run(r2.run_id, result=result_envelope)
    store.save_artifact(r2.run_id, "blob.bin", b"Y" * 10_000)

    assert store.get_total_disk_usage() > 20_000
    pruned = store.apply_quota(max_bytes=15_000)
    assert "r1" in pruned
    assert store.get_run("r1").status == "expired"
    assert store.get_run("r2").status == "succeeded"
    assert store.get_total_disk_usage() < 20_000


def test_list_runs_filtering_and_order(tmp_path: Path, run_request: RunRequest) -> None:
    store = SimulationRunStore(results_root=tmp_path)
    r1 = store.create_run(run_request, run_id="run_a")
    r2 = store.create_run(run_request, run_id="run_b")
    store.start_run(r2.run_id)
    assert [r.run_id for r in store.list_runs()] == ["run_a", "run_b"]
    assert [r.run_id for r in store.list_runs(status="queued")] == ["run_a"]
    assert [r.run_id for r in store.list_runs(status="running")] == ["run_b"]
