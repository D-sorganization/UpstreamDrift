"""Tests for :mod:`training.job` — :class:`TrainingJob` and :class:`RunResult`."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from training import (
    InvalidStatusTransitionError,
    MetricKind,
    RunResult,
    TrainingConfig,
    TrainingConfigError,
    TrainingFramework,
    TrainingJob,
    TrainingMetric,
    TrainingStatus,
    new_job_id,
    new_run_id,
)

pytestmark = pytest.mark.unit


def _config() -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="m:train",
        output_dir=Path("/tmp/out"),
    )


def _pending_job(now: float = 100.0) -> TrainingJob:
    return TrainingJob(
        job_id=new_job_id(),
        config=_config(),
        status=TrainingStatus.PENDING,
        created_at=now,
    )


class TestTrainingJobConstruction:
    def test_minimal_pending(self) -> None:
        job = _pending_job()
        assert job.status is TrainingStatus.PENDING
        assert job.started_at is None
        assert job.completed_at is None
        assert job.error_message is None
        assert job.run_id is None

    def test_rejects_non_jobid(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id="raw-string",  # type: ignore[arg-type]
                config=_config(),
                status=TrainingStatus.PENDING,
                created_at=0.0,
            )

    def test_rejects_non_config(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id=new_job_id(),
                config={"framework": "pytorch"},  # type: ignore[arg-type]
                status=TrainingStatus.PENDING,
                created_at=0.0,
            )

    def test_rejects_non_status(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id=new_job_id(),
                config=_config(),
                status="pending",  # type: ignore[arg-type]
                created_at=0.0,
            )

    def test_rejects_negative_created_at(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id=new_job_id(),
                config=_config(),
                status=TrainingStatus.PENDING,
                created_at=-1.0,
            )

    def test_rejects_started_before_created(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id=new_job_id(),
                config=_config(),
                status=TrainingStatus.RUNNING,
                created_at=100.0,
                started_at=50.0,
                run_id=new_run_id(),
            )

    def test_rejects_completed_without_started(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id=new_job_id(),
                config=_config(),
                status=TrainingStatus.COMPLETED,
                created_at=100.0,
                completed_at=200.0,
            )

    def test_rejects_completed_before_started(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id=new_job_id(),
                config=_config(),
                status=TrainingStatus.COMPLETED,
                created_at=100.0,
                started_at=200.0,
                completed_at=150.0,
            )

    def test_rejects_terminal_without_completed_at(self) -> None:
        for terminal in (TrainingStatus.COMPLETED, TrainingStatus.CANCELLED):
            with pytest.raises(TrainingConfigError):
                TrainingJob(
                    job_id=new_job_id(),
                    config=_config(),
                    status=terminal,
                    created_at=100.0,
                    started_at=110.0,
                )

    def test_rejects_error_on_non_failed(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id=new_job_id(),
                config=_config(),
                status=TrainingStatus.RUNNING,
                created_at=100.0,
                started_at=110.0,
                run_id=new_run_id(),
                error_message="should not be set",
            )

    def test_rejects_failed_without_error_message(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id=new_job_id(),
                config=_config(),
                status=TrainingStatus.FAILED,
                created_at=100.0,
                started_at=110.0,
                completed_at=120.0,
                error_message="",
            )

    def test_rejects_non_runid_run_id(self) -> None:
        with pytest.raises(TrainingConfigError):
            TrainingJob(
                job_id=new_job_id(),
                config=_config(),
                status=TrainingStatus.PENDING,
                created_at=0.0,
                run_id="raw",  # type: ignore[arg-type]
            )


class TestTrainingJobWithStatus:
    def test_pending_to_queued(self) -> None:
        job = _pending_job(now=100.0)
        next_job = job.with_status(TrainingStatus.QUEUED, now=110.0)
        assert next_job.status is TrainingStatus.QUEUED
        assert next_job.started_at is None  # not yet running
        assert next_job.completed_at is None

    def test_queued_to_running_sets_started_at(self) -> None:
        job = _pending_job(now=100.0).with_status(TrainingStatus.QUEUED, now=110.0)
        run_id = new_run_id()
        running = job.with_status(TrainingStatus.RUNNING, now=120.0, run_id=run_id)
        assert running.status is TrainingStatus.RUNNING
        assert running.started_at == 120.0
        assert running.run_id == run_id

    def test_running_requires_run_id_on_first_transition(self) -> None:
        job = _pending_job(now=100.0).with_status(TrainingStatus.QUEUED, now=110.0)
        with pytest.raises(TrainingConfigError):
            job.with_status(TrainingStatus.RUNNING, now=120.0)

    def test_running_to_completed_sets_completed_at(self) -> None:
        job = (
            _pending_job(now=100.0)
            .with_status(TrainingStatus.QUEUED, now=110.0)
            .with_status(TrainingStatus.RUNNING, now=120.0, run_id=new_run_id())
        )
        done = job.with_status(TrainingStatus.COMPLETED, now=200.0)
        assert done.completed_at == 200.0
        assert done.status.is_terminal

    def test_running_to_failed_requires_error_message(self) -> None:
        job = (
            _pending_job(now=100.0)
            .with_status(TrainingStatus.QUEUED, now=110.0)
            .with_status(TrainingStatus.RUNNING, now=120.0, run_id=new_run_id())
        )
        with pytest.raises(TrainingConfigError):
            job.with_status(TrainingStatus.FAILED, now=200.0)
        failed = job.with_status(TrainingStatus.FAILED, now=200.0, error_message="OOM")
        assert failed.status is TrainingStatus.FAILED
        assert failed.error_message == "OOM"

    def test_error_message_forbidden_for_non_failed_transition(self) -> None:
        job = _pending_job(now=100.0)
        with pytest.raises(TrainingConfigError):
            job.with_status(TrainingStatus.QUEUED, now=110.0, error_message="nope")

    def test_invalid_transition_raises_specific_error(self) -> None:
        job = _pending_job(now=100.0)
        with pytest.raises(InvalidStatusTransitionError):
            job.with_status(TrainingStatus.RUNNING, now=110.0)

    def test_now_must_be_at_least_created_at(self) -> None:
        job = _pending_job(now=100.0)
        with pytest.raises(TrainingConfigError):
            job.with_status(TrainingStatus.QUEUED, now=50.0)

    def test_pause_resume_round_trip(self) -> None:
        job = (
            _pending_job(now=100.0)
            .with_status(TrainingStatus.QUEUED, now=110.0)
            .with_status(TrainingStatus.RUNNING, now=120.0, run_id=new_run_id())
        )
        paused = job.with_status(TrainingStatus.PAUSED, now=150.0)
        resumed = paused.with_status(TrainingStatus.RUNNING, now=160.0)
        assert resumed.status is TrainingStatus.RUNNING
        assert resumed.started_at == 120.0  # original start preserved


class TestRunResult:
    def test_minimal_completed_result(self) -> None:
        rid = new_run_id()
        result = RunResult(run_id=rid, status=TrainingStatus.COMPLETED, duration_s=12.5)
        assert result.run_id == rid
        assert result.final_metrics == ()
        assert result.artifacts == ()

    def test_failed_result_requires_error(self) -> None:
        with pytest.raises(TrainingConfigError):
            RunResult(
                run_id=new_run_id(),
                status=TrainingStatus.FAILED,
                duration_s=1.0,
            )

    def test_failed_result_with_error(self) -> None:
        result = RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.FAILED,
            duration_s=1.0,
            error="OOM",
        )
        assert result.error == "OOM"

    def test_completed_with_error_rejected(self) -> None:
        with pytest.raises(TrainingConfigError):
            RunResult(
                run_id=new_run_id(),
                status=TrainingStatus.COMPLETED,
                duration_s=1.0,
                error="should-not-be-set",
            )

    def test_non_terminal_status_rejected(self) -> None:
        with pytest.raises(TrainingConfigError):
            RunResult(
                run_id=new_run_id(),
                status=TrainingStatus.RUNNING,
                duration_s=1.0,
            )

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(TrainingConfigError):
            RunResult(
                run_id=new_run_id(),
                status=TrainingStatus.COMPLETED,
                duration_s=-0.1,
            )

    def test_final_metrics_must_be_tuple_of_metric(self) -> None:
        m = TrainingMetric(
            name="loss",
            value=0.1,
            step=0,
            timestamp=0.0,
            kind=MetricKind.LOSS,
        )
        result = RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=1.0,
            final_metrics=(m,),
        )
        assert result.final_metrics == (m,)
        with pytest.raises(TrainingConfigError):
            RunResult(
                run_id=new_run_id(),
                status=TrainingStatus.COMPLETED,
                duration_s=1.0,
                final_metrics=[m],  # type: ignore[arg-type]
            )

    def test_artifacts_must_be_tuple_of_paths(self) -> None:
        result = RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=1.0,
            artifacts=(Path("/tmp/checkpoint.pt"),),
        )
        assert result.artifacts == (Path("/tmp/checkpoint.pt"),)
        with pytest.raises(TrainingConfigError):
            RunResult(
                run_id=new_run_id(),
                status=TrainingStatus.COMPLETED,
                duration_s=1.0,
                artifacts=("/tmp/checkpoint.pt",),  # type: ignore[arg-type]
            )


class TestImmutability:
    def test_training_job_is_frozen(self) -> None:
        job = _pending_job()
        with pytest.raises(dataclasses.FrozenInstanceError):
            job.status = TrainingStatus.QUEUED  # type: ignore[misc]

    def test_run_result_is_frozen(self) -> None:
        result = RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.duration_s = 2.0  # type: ignore[misc]
