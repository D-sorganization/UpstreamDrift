"""Tests for :mod:`training.runtime.subprocess_driver`.

End-to-end coverage: each test spawns a real worker subprocess via
:class:`SubprocessDriver` and exercises a fixture entry point from
``tests/unit/training/runtime/_fixture_runner.py``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from training import (
    TrainingConfig,
    TrainingFramework,
    TrainingJob,
    TrainingStatus,
    new_job_id,
)
from training.runtime import (
    Driver,
    InMemoryProgressSink,
    JobHandle,
    JobHandleStatus,
    SubprocessDriver,
    scan_pidfiles,
)
from training.runtime.driver import DriverError

pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_PY = _REPO_ROOT / "src" / "shared" / "python"
_FIXTURE_DIR = Path(__file__).resolve().parent


def _worker_env() -> dict[str, str]:
    """Env vars that let the worker import training + the fixture module."""

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(_SRC_PY), str(_FIXTURE_DIR)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _make_config(
    entry_point: str,
    output_dir: Path,
    *,
    hyperparameters: dict[str, object] | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point=entry_point,
        output_dir=output_dir,
        hyperparameters=hyperparameters or {},
    )


def _queued_job(config: TrainingConfig) -> TrainingJob:
    pending = TrainingJob(
        job_id=new_job_id(),
        config=config,
        status=TrainingStatus.PENDING,
        created_at=0.0,
    )
    return pending.with_status(TrainingStatus.QUEUED, now=0.0)


@pytest.fixture
def driver() -> Iterator[SubprocessDriver]:
    drv = SubprocessDriver(worker_env=_worker_env())
    try:
        yield drv
    finally:
        drv.shutdown()


class TestSubprocessDriverConstruction:
    def test_satisfies_driver_protocol(self) -> None:
        drv = SubprocessDriver(worker_env=_worker_env())
        try:
            assert isinstance(drv, Driver)
        finally:
            drv.shutdown()

    def test_rejects_empty_worker_command(self) -> None:
        with pytest.raises(ValueError, match="worker_command"):
            SubprocessDriver(worker_command=[])

    def test_rejects_non_positive_cancel_timeout(self) -> None:
        with pytest.raises(ValueError, match="cancel_timeout_s"):
            SubprocessDriver(cancel_timeout_s=0)

    def test_rejects_negative_kill_timeout(self) -> None:
        with pytest.raises(ValueError, match="kill_timeout_s"):
            SubprocessDriver(kill_timeout_s=-1)


class TestSubprocessDriverExecution:
    def test_runs_to_completion(self, driver: SubprocessDriver, tmp_path: Path) -> None:
        config = _make_config(
            "_fixture_runner:emit_metrics",
            tmp_path,
            hyperparameters={"num_metrics": 3},
        )
        sink = InMemoryProgressSink()
        handle = driver.start(_queued_job(config), progress=sink)
        result = driver.result(handle, timeout=15.0)

        assert result.status is TrainingStatus.COMPLETED
        assert result.duration_s >= 0.0
        # Metrics arrived in order.
        metrics = sink.metrics
        assert len(metrics) == 3
        assert [m.step for m in metrics] == [0, 1, 2]
        assert [m.name for m in metrics] == ["loss", "loss", "loss"]
        # Status event recorded.
        assert any(s is TrainingStatus.RUNNING for s, _ in sink.statuses)

    def test_handle_finished_status_after_result(
        self, driver: SubprocessDriver, tmp_path: Path
    ) -> None:
        config = _make_config(
            "_fixture_runner:emit_metrics",
            tmp_path,
            hyperparameters={"num_metrics": 1},
        )
        handle = driver.start(_queued_job(config), progress=InMemoryProgressSink())
        driver.result(handle, timeout=15.0)
        assert driver.status(handle) is JobHandleStatus.FINISHED

    def test_cancel_mid_run_yields_cancelled_result(
        self, driver: SubprocessDriver, tmp_path: Path
    ) -> None:
        config = _make_config(
            "_fixture_runner:slow_until_cancel",
            tmp_path,
            hyperparameters={"poll_interval": 0.02},
        )
        sink = InMemoryProgressSink()
        handle = driver.start(_queued_job(config), progress=sink)

        # Wait for the worker to have started running.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if any(s is TrainingStatus.RUNNING for s, _ in sink.statuses):
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker never reported RUNNING status")

        driver.cancel(handle)
        result = driver.result(handle, timeout=15.0)
        assert result.status is TrainingStatus.CANCELLED

    def test_worker_crash_becomes_failed_result(
        self, driver: SubprocessDriver, tmp_path: Path
    ) -> None:
        config = _make_config(
            "_fixture_runner:raise_immediately",
            tmp_path,
        )
        handle = driver.start(_queued_job(config), progress=InMemoryProgressSink())
        result = driver.result(handle, timeout=15.0)
        assert result.status is TrainingStatus.FAILED
        assert result.error is not None
        assert "simulated training failure" in result.error

    def test_stderr_does_not_pollute_metric_stream(
        self, driver: SubprocessDriver, tmp_path: Path
    ) -> None:
        config = _make_config(
            "_fixture_runner:emit_metrics_with_stderr",
            tmp_path,
            hyperparameters={"num_metrics": 2},
        )
        sink = InMemoryProgressSink()
        handle = driver.start(_queued_job(config), progress=sink)
        result = driver.result(handle, timeout=15.0)
        assert result.status is TrainingStatus.COMPLETED
        # Exactly 2 well-formed metrics; the stderr garbage never appeared.
        assert len(sink.metrics) == 2

    def test_rejects_non_queued_job(
        self, driver: SubprocessDriver, tmp_path: Path
    ) -> None:
        config = _make_config("_fixture_runner:emit_metrics", tmp_path)
        pending = TrainingJob(
            job_id=new_job_id(),
            config=config,
            status=TrainingStatus.PENDING,
            created_at=0.0,
        )
        with pytest.raises(DriverError, match="QUEUED"):
            driver.start(pending, progress=InMemoryProgressSink())

    def test_unknown_handle_raises(self, driver: SubprocessDriver) -> None:
        fake = JobHandle(job_id=new_job_id(), cancel_token=_NoOpToken())
        with pytest.raises(DriverError):
            driver.cancel(fake)
        with pytest.raises(DriverError):
            driver.status(fake)
        with pytest.raises(DriverError):
            driver.result(fake)

    def test_start_after_shutdown_raises(self, tmp_path: Path) -> None:
        drv = SubprocessDriver(worker_env=_worker_env())
        drv.shutdown()
        config = _make_config("_fixture_runner:emit_metrics", tmp_path)
        with pytest.raises(DriverError, match="shut down"):
            drv.start(_queued_job(config), progress=InMemoryProgressSink())


class TestPidfileBookkeeping:
    def test_pidfile_written_during_run_and_cleaned_up(
        self, driver: SubprocessDriver, tmp_path: Path
    ) -> None:
        config = _make_config(
            "_fixture_runner:emit_metrics",
            tmp_path,
            hyperparameters={"num_metrics": 1},
        )
        handle = driver.start(_queued_job(config), progress=InMemoryProgressSink())
        driver.result(handle, timeout=15.0)
        # After completion the pidfile is removed.
        assert not (tmp_path / ".training.pid").exists()

    def test_scan_pidfiles_detects_dead_pid(self, tmp_path: Path) -> None:
        pidfile = tmp_path / ".training.pid"
        # Pid 1 is init / systemd on most systems and not us, but the
        # important case is "definitely-not-a-real-pid". Pick a very
        # high pid unlikely to be in use.
        pidfile.write_text("999999999\n", encoding="utf-8")
        result = scan_pidfiles([tmp_path])
        assert result[pidfile] is None

    def test_scan_pidfiles_detects_live_pid(self, tmp_path: Path) -> None:
        pidfile = tmp_path / ".training.pid"
        pidfile.write_text(f"{os.getpid()}\n", encoding="utf-8")
        result = scan_pidfiles([tmp_path])
        assert result[pidfile] == os.getpid()

    def test_scan_pidfiles_skips_missing(self, tmp_path: Path) -> None:
        result = scan_pidfiles([tmp_path])
        assert result == {}

    def test_scan_pidfiles_handles_malformed(self, tmp_path: Path) -> None:
        pidfile = tmp_path / ".training.pid"
        pidfile.write_text("not a number", encoding="utf-8")
        result = scan_pidfiles([tmp_path])
        assert result[pidfile] is None


class _NoOpToken:
    @property
    def is_cancelled(self) -> bool:
        return False

    def request_cancel(self) -> None:
        return None
