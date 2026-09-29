"""Tests for :mod:`training.registry`."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import (
    DuplicateJobError,
    JobNotFoundError,
    JobRegistry,
    TrainingConfig,
    TrainingFramework,
    TrainingJob,
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


def _job(status: TrainingStatus = TrainingStatus.PENDING) -> TrainingJob:
    if status is TrainingStatus.RUNNING:
        return TrainingJob(
            job_id=new_job_id(),
            config=_config(),
            status=TrainingStatus.RUNNING,
            created_at=100.0,
            started_at=110.0,
            run_id=new_run_id(),
        )
    return TrainingJob(
        job_id=new_job_id(),
        config=_config(),
        status=status,
        created_at=100.0,
    )


class TestJobRegistryAdd:
    def test_add_and_get(self) -> None:
        registry = JobRegistry()
        job = _job()
        registry.add(job)
        assert registry.get(job.job_id) == job

    def test_initial_iterable(self) -> None:
        job = _job()
        registry = JobRegistry(initial=(job,))
        assert registry.has(job.job_id)

    def test_duplicate_raises(self) -> None:
        job = _job()
        registry = JobRegistry(initial=(job,))
        with pytest.raises(DuplicateJobError):
            registry.add(job)

    def test_add_rejects_non_job(self) -> None:
        with pytest.raises(TypeError):
            JobRegistry().add("not a job")  # type: ignore[arg-type]


class TestJobRegistryGet:
    def test_missing_raises(self) -> None:
        registry = JobRegistry()
        with pytest.raises(JobNotFoundError):
            registry.get(new_job_id())

    def test_get_rejects_non_jobid(self) -> None:
        with pytest.raises(TypeError):
            JobRegistry().get("raw-string")  # type: ignore[arg-type]

    def test_has_does_not_raise(self) -> None:
        assert JobRegistry().has(new_job_id()) is False


class TestJobRegistryReplace:
    def test_replace_atomic(self) -> None:
        job = _job(status=TrainingStatus.PENDING)
        registry = JobRegistry(initial=(job,))
        queued = job.with_status(TrainingStatus.QUEUED, now=110.0)
        previous = registry.replace(queued)
        assert previous == job
        assert registry.get(job.job_id).status is TrainingStatus.QUEUED

    def test_replace_missing_raises(self) -> None:
        job = _job()
        registry = JobRegistry()
        with pytest.raises(JobNotFoundError):
            registry.replace(job)

    def test_replace_rejects_non_job(self) -> None:
        with pytest.raises(TypeError):
            JobRegistry().replace({"job_id": "x"})  # type: ignore[arg-type]


class TestJobRegistryRemove:
    def test_remove(self) -> None:
        job = _job()
        registry = JobRegistry(initial=(job,))
        removed = registry.remove(job.job_id)
        assert removed == job
        assert not registry.has(job.job_id)

    def test_remove_missing_raises(self) -> None:
        with pytest.raises(JobNotFoundError):
            JobRegistry().remove(new_job_id())


class TestJobRegistryList:
    def test_unfiltered(self) -> None:
        a = _job()
        b = _job(status=TrainingStatus.RUNNING)
        registry = JobRegistry(initial=(a, b))
        listed = registry.list()
        assert len(listed) == 2
        assert a in listed
        assert b in listed

    def test_filter_by_status(self) -> None:
        a = _job()
        b = _job(status=TrainingStatus.RUNNING)
        registry = JobRegistry(initial=(a, b))
        running = registry.list(status=TrainingStatus.RUNNING)
        assert running == (b,)

    def test_predicate_filter(self) -> None:
        a = _job()
        b = _job(status=TrainingStatus.RUNNING)
        registry = JobRegistry(initial=(a, b))
        filtered = registry.list(predicate=lambda j: j.status is TrainingStatus.PENDING)
        assert filtered == (a,)

    def test_combined_filters_are_and(self) -> None:
        a = _job(status=TrainingStatus.PENDING)
        registry = JobRegistry(initial=(a,))
        result = registry.list(
            status=TrainingStatus.PENDING,
            predicate=lambda j: j.status is TrainingStatus.RUNNING,
        )
        assert result == ()


class TestJobRegistryThreadSafety:
    def test_concurrent_adds(self) -> None:
        import threading

        registry = JobRegistry()
        n_threads = 20

        def add_one() -> None:
            registry.add(_job())

        threads = [threading.Thread(target=add_one) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(registry) == n_threads
