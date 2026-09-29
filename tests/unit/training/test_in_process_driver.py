"""Tests for :mod:`training.runtime.in_process_driver`."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from training import (
    RunResult,
    TrainingConfig,
    TrainingFramework,
    TrainingJob,
    TrainingStatus,
    new_job_id,
    new_run_id,
)
from training.contracts import CancelToken, ProgressSink
from training.runtime import (
    Driver,
    InMemoryProgressSink,
    InProcessDriver,
    JobHandle,
    JobHandleStatus,
    RunnerRegistry,
)
from training.runtime.driver import DriverError

pytestmark = pytest.mark.unit


def _config() -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="m:train",
        output_dir=Path("/tmp/out"),
    )


def _queued_job() -> TrainingJob:
    pending = TrainingJob(
        job_id=new_job_id(),
        config=_config(),
        status=TrainingStatus.PENDING,
        created_at=0.0,
    )
    return pending.with_status(TrainingStatus.QUEUED, now=0.0)


class _SuccessRunner:
    framework = TrainingFramework.PYTORCH

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.run_calls = 0

    def can_run(self, config: TrainingConfig) -> bool:
        return True

    def prepare(self, config: TrainingConfig) -> None:
        self.prepare_calls += 1

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        self.run_calls += 1
        return RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=0.0,
        )


class _RaisingRunner:
    framework = TrainingFramework.PYTORCH

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

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
        raise self._exc


class _CancellableRunner:
    framework = TrainingFramework.PYTORCH

    def __init__(self) -> None:
        self.cancelled_observed = threading.Event()

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
        for _ in range(1000):
            if cancel.is_cancelled:
                self.cancelled_observed.set()
                return RunResult(
                    run_id=new_run_id(),
                    status=TrainingStatus.CANCELLED,
                    duration_s=0.0,
                )
            time.sleep(0.01)
        return RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=0.0,
        )


class TestInProcessDriverConstruction:
    def test_satisfies_driver_protocol(self) -> None:
        driver = InProcessDriver(RunnerRegistry())
        assert isinstance(driver, Driver)
        driver.shutdown()

    def test_rejects_non_runner_registry(self) -> None:
        with pytest.raises(TypeError):
            InProcessDriver({})  # type: ignore[arg-type]

    def test_rejects_bad_max_workers(self) -> None:
        with pytest.raises(ValueError):
            InProcessDriver(RunnerRegistry(), max_workers=0)


class TestInProcessDriverExecution:
    def test_runs_to_completion(self) -> None:
        runner = _SuccessRunner()
        registry = RunnerRegistry()
        registry.register(runner)
        driver = InProcessDriver(registry)
        try:
            handle = driver.start(_queued_job(), progress=InMemoryProgressSink())
            result = driver.result(handle, timeout=2.0)
            assert result.status is TrainingStatus.COMPLETED
            assert runner.prepare_calls == 1
            assert runner.run_calls == 1
        finally:
            driver.shutdown()

    def test_runner_resolution_failure_becomes_failed_result(self) -> None:
        # Registry has no runner for PYTORCH.
        registry = RunnerRegistry()
        driver = InProcessDriver(registry)
        try:
            handle = driver.start(_queued_job(), progress=InMemoryProgressSink())
            result = driver.result(handle, timeout=2.0)
            assert result.status is TrainingStatus.FAILED
            assert "runner resolution failed" in (result.error or "")
        finally:
            driver.shutdown()

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("oom"),
            ValueError("bad shape"),
            ImportError("torch missing"),
            OSError("disk full"),
        ],
    )
    def test_runner_exception_becomes_failed_result(self, exc: BaseException) -> None:
        registry = RunnerRegistry()
        registry.register(_RaisingRunner(exc))
        driver = InProcessDriver(registry)
        try:
            handle = driver.start(_queued_job(), progress=InMemoryProgressSink())
            result = driver.result(handle, timeout=2.0)
            assert result.status is TrainingStatus.FAILED
            assert str(exc) in (result.error or "")
        finally:
            driver.shutdown()

    def test_cancellation_observed_by_runner(self) -> None:
        runner = _CancellableRunner()
        registry = RunnerRegistry()
        registry.register(runner)
        driver = InProcessDriver(registry)
        try:
            handle = driver.start(_queued_job(), progress=InMemoryProgressSink())
            time.sleep(0.05)  # let it start
            driver.cancel(handle)
            assert runner.cancelled_observed.wait(timeout=2.0)
            result = driver.result(handle, timeout=2.0)
            assert result.status is TrainingStatus.CANCELLED
        finally:
            driver.shutdown()

    def test_rejects_non_queued_job(self) -> None:
        registry = RunnerRegistry()
        registry.register(_SuccessRunner())
        driver = InProcessDriver(registry)
        pending = TrainingJob(
            job_id=new_job_id(),
            config=_config(),
            status=TrainingStatus.PENDING,
            created_at=0.0,
        )
        try:
            with pytest.raises(DriverError, match="QUEUED"):
                driver.start(pending, progress=InMemoryProgressSink())
        finally:
            driver.shutdown()

    def test_status_transitions(self) -> None:
        runner = _CancellableRunner()
        registry = RunnerRegistry()
        registry.register(runner)
        driver = InProcessDriver(registry, max_workers=1)
        try:
            handle = driver.start(_queued_job(), progress=InMemoryProgressSink())
            # eventually running
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if driver.status(handle) is JobHandleStatus.RUNNING:
                    break
                time.sleep(0.01)
            driver.cancel(handle)
            driver.result(handle, timeout=2.0)
            assert driver.status(handle) is JobHandleStatus.FINISHED
        finally:
            driver.shutdown()

    def test_unknown_handle_raises(self) -> None:
        driver = InProcessDriver(RunnerRegistry())
        fake = JobHandle(job_id=new_job_id(), cancel_token=_NoOpToken())
        try:
            with pytest.raises(DriverError):
                driver.cancel(fake)
            with pytest.raises(DriverError):
                driver.status(fake)
            with pytest.raises(DriverError):
                driver.result(fake)
        finally:
            driver.shutdown()

    def test_shutdown_cancels_in_flight(self) -> None:
        runner = _CancellableRunner()
        registry = RunnerRegistry()
        registry.register(runner)
        driver = InProcessDriver(registry)
        handle = driver.start(_queued_job(), progress=InMemoryProgressSink())
        time.sleep(0.05)
        driver.shutdown(wait=True)
        assert handle.cancel_token.is_cancelled is True


class _NoOpToken:
    @property
    def is_cancelled(self) -> bool:
        return False

    def request_cancel(self) -> None:
        return None
