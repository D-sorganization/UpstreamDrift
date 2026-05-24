"""Tests for :mod:`training.scheduler`."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from training import (
    CompatibilityChecker,
    CompatibilityError,
    JobRegistry,
    RunResult,
    Scheduler,
    SchedulerError,
    StatusChangeEvent,
    TrainingConfig,
    TrainingFramework,
    TrainingStatus,
    new_run_id,
)
from training.contracts import CancelToken, ProgressSink
from training.runtime import (
    InProcessDriver,
    RunnerRegistry,
)
from training.runtime.runner_registry import NoRunnerAvailableError

pytestmark = pytest.mark.unit


def _config() -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="m:train",
        output_dir=Path("/tmp/out"),
    )


class _SuccessRunner:
    framework = TrainingFramework.PYTORCH

    def can_run(self, config: TrainingConfig) -> bool:
        return True

    def prepare(self, config: TrainingConfig) -> None:
        return None

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        return RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=0.01,
        )


class _BlockingRunner:
    framework = TrainingFramework.PYTORCH

    def __init__(self) -> None:
        self.started = threading.Event()

    def can_run(self, config: TrainingConfig) -> bool:
        return True

    def prepare(self, config: TrainingConfig) -> None:
        return None

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        self.started.set()
        while not cancel.is_cancelled:
            time.sleep(0.01)
        return RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.CANCELLED,
            duration_s=0.0,
        )


def _scheduler(
    *,
    runners: RunnerRegistry | None = None,
    driver: InProcessDriver | None = None,
    compat: CompatibilityChecker | None = None,
    clock=None,
) -> Scheduler:
    runners = runners or RunnerRegistry()
    runners.register(_SuccessRunner())
    driver = driver or InProcessDriver(runners)
    kwargs: dict[str, object] = {}
    if clock is not None:
        kwargs["clock"] = clock
    return Scheduler(
        registry=JobRegistry(),
        runners=runners,
        driver=driver,
        compatibility_checker=compat,
        **kwargs,  # type: ignore[arg-type]
    )


class TestSchedulerSubmit:
    def test_submit_queues_job(self) -> None:
        scheduler = _scheduler()
        try:
            job = scheduler.submit(_config())
            assert job.status is TrainingStatus.QUEUED
            assert scheduler.get(job.job_id) == job
        finally:
            scheduler.shutdown()

    def test_submit_fails_without_runner(self) -> None:
        runners = RunnerRegistry()
        driver = InProcessDriver(runners)
        scheduler = Scheduler(
            registry=JobRegistry(),
            runners=runners,
            driver=driver,
        )
        try:
            with pytest.raises(NoRunnerAvailableError):
                scheduler.submit(_config())
        finally:
            scheduler.shutdown()

    def test_submit_runs_compat_check_when_engine_passed(self) -> None:
        scheduler = _scheduler(compat=CompatibilityChecker())
        try:
            # PyTorch + drake is compatible.
            scheduler.submit(_config(), target_engine="drake")
        finally:
            scheduler.shutdown()

    def test_compat_failure_blocks_submit(self) -> None:
        gym_config = TrainingConfig(
            framework=TrainingFramework.GYMNASIUM,
            entry_point="m:train",
            output_dir=Path("/tmp/out"),
        )
        runners = RunnerRegistry()
        runners.register(_GymRunner())
        scheduler = Scheduler(
            registry=JobRegistry(),
            runners=runners,
            driver=InProcessDriver(runners),
            compatibility_checker=CompatibilityChecker(),
        )
        try:
            with pytest.raises(CompatibilityError):
                scheduler.submit(gym_config, target_engine="drake")
        finally:
            scheduler.shutdown()

    def test_unknown_engine_blocks_submit(self) -> None:
        scheduler = _scheduler(compat=CompatibilityChecker())
        try:
            with pytest.raises(CompatibilityError):
                scheduler.submit(_config(), target_engine="bogus")
        finally:
            scheduler.shutdown()


class TestSchedulerLifecycle:
    def test_full_lifecycle(self) -> None:
        scheduler = _scheduler()
        try:
            job = scheduler.submit(_config())
            running = scheduler.start(job.job_id)
            assert running.status is TrainingStatus.RUNNING
            result = scheduler.collect(job.job_id, timeout=2.0)
            assert result.status is TrainingStatus.COMPLETED
            final = scheduler.get(job.job_id)
            assert final.status is TrainingStatus.COMPLETED
        finally:
            scheduler.shutdown()

    def test_start_rejects_non_queued(self) -> None:
        scheduler = _scheduler()
        try:
            job = scheduler.submit(_config())
            scheduler.start(job.job_id)
            with pytest.raises(SchedulerError, match="cannot start"):
                scheduler.start(job.job_id)
            scheduler.collect(job.job_id, timeout=2.0)
        finally:
            scheduler.shutdown()

    def test_cancel_pending_job(self) -> None:
        scheduler = _scheduler()
        try:
            job = scheduler.submit(_config())
            cancelled = scheduler.cancel(job.job_id)
            assert cancelled.status is TrainingStatus.CANCELLED
            assert cancelled.started_at is None
            assert cancelled.completed_at is not None
        finally:
            scheduler.shutdown()

    def test_cancel_running_job(self) -> None:
        runners = RunnerRegistry()
        blocking = _BlockingRunner()
        runners.register(blocking)
        driver = InProcessDriver(runners)
        scheduler = Scheduler(
            registry=JobRegistry(),
            runners=runners,
            driver=driver,
        )
        try:
            job = scheduler.submit(_config())
            scheduler.start(job.job_id)
            assert blocking.started.wait(timeout=2.0)
            scheduler.cancel(job.job_id)
            result = scheduler.collect(job.job_id, timeout=2.0)
            assert result.status is TrainingStatus.CANCELLED
        finally:
            scheduler.shutdown()

    def test_cancel_terminal_raises(self) -> None:
        scheduler = _scheduler()
        try:
            job = scheduler.submit(_config())
            scheduler.cancel(job.job_id)
            with pytest.raises(SchedulerError, match="terminal"):
                scheduler.cancel(job.job_id)
        finally:
            scheduler.shutdown()


class TestSchedulerPauseResume:
    def test_pause_resume_round_trip(self) -> None:
        runners = RunnerRegistry()
        blocking = _BlockingRunner()
        runners.register(blocking)
        driver = InProcessDriver(runners)
        scheduler = Scheduler(
            registry=JobRegistry(),
            runners=runners,
            driver=driver,
        )
        try:
            job = scheduler.submit(_config())
            scheduler.start(job.job_id)
            assert blocking.started.wait(timeout=2.0)
            paused = scheduler.pause(job.job_id)
            assert paused.status is TrainingStatus.PAUSED
            resumed = scheduler.resume(job.job_id)
            assert resumed.status is TrainingStatus.RUNNING
            scheduler.cancel(job.job_id)
            scheduler.collect(job.job_id, timeout=2.0)
        finally:
            scheduler.shutdown()

    def test_pause_rejects_non_running(self) -> None:
        scheduler = _scheduler()
        try:
            job = scheduler.submit(_config())
            with pytest.raises(SchedulerError, match="can only pause"):
                scheduler.pause(job.job_id)
        finally:
            scheduler.shutdown()

    def test_resume_rejects_non_paused(self) -> None:
        scheduler = _scheduler()
        try:
            job = scheduler.submit(_config())
            with pytest.raises(SchedulerError, match="can only resume"):
                scheduler.resume(job.job_id)
        finally:
            scheduler.shutdown()


class TestSchedulerObservers:
    def test_observer_receives_status_changes(self) -> None:
        scheduler = _scheduler()
        events: list[StatusChangeEvent] = []
        unsubscribe = scheduler.on_status_change(events.append)
        try:
            job = scheduler.submit(_config())  # noqa: F841 - id captured via events
            scheduler.start(job.job_id)
            scheduler.collect(job.job_id, timeout=2.0)
            transitions = [(e.previous_status, e.new_status) for e in events]
            assert (TrainingStatus.PENDING, TrainingStatus.QUEUED) in transitions
            assert (TrainingStatus.QUEUED, TrainingStatus.RUNNING) in transitions
            assert (TrainingStatus.RUNNING, TrainingStatus.COMPLETED) in transitions
        finally:
            unsubscribe()
            scheduler.shutdown()

    def test_unsubscribe_stops_delivery(self) -> None:
        scheduler = _scheduler()
        events: list[StatusChangeEvent] = []
        unsubscribe = scheduler.on_status_change(events.append)
        try:
            scheduler.submit(_config())  # one event
            unsubscribe()
            scheduler.submit(_config())  # not received
            assert len(events) == 1
        finally:
            scheduler.shutdown()

    def test_observer_exception_does_not_break_fan_out(self) -> None:
        scheduler = _scheduler()
        received_after_bad: list[StatusChangeEvent] = []

        def bad(event: StatusChangeEvent) -> None:
            raise RuntimeError("observer kaboom")

        scheduler.on_status_change(bad)
        scheduler.on_status_change(received_after_bad.append)
        try:
            scheduler.submit(_config())
            assert len(received_after_bad) == 1
        finally:
            scheduler.shutdown()


class _GymRunner:
    framework = TrainingFramework.GYMNASIUM

    def can_run(self, config: TrainingConfig) -> bool:
        return True

    def prepare(self, config: TrainingConfig) -> None:
        return None

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        return RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=0.0,
        )
