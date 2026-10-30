"""Unit tests for sim_run isolated subprocess worker pool and lifecycle (Epic #14085 / #14088)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.core.error_utils import SimulationBusyError
from src.sim_run.models import RunRequest
from src.sim_run.store import SimulationRunStore
from src.sim_run.worker import (
    LinuxLimits,
    SimulationWorkerPool,
    apply_linux_rlimits,
    reap_subprocess,
    run_simulation_job,
)


@pytest.fixture
def run_store(tmp_path: Path) -> SimulationRunStore:
    return SimulationRunStore(results_root=tmp_path / "sim_runs")


@pytest.fixture
def sample_request() -> RunRequest:
    return RunRequest(
        engine_type="golf_swing_pendulum",
        duration=1.0,
        timestep=0.01,
        initial_state={"q": [0.0, 0.0]},
        timeout_seconds=5.0,
    )


@pytest.mark.unit
def test_falsifying_uncooperative_worker_timeout_reaped_and_failed_timeout(
    run_store: SimulationRunStore, sample_request: RunRequest
) -> None:
    """Falsifying test: an uncooperative fixture worker exceeding timeout is killed, reaped, and failed:timeout."""
    record = run_store.create_run(sample_request)
    pool = SimulationWorkerPool(run_store, max_concurrency=1)
    uncoop_cmd = [
        sys.executable,
        "-c",
        "import time, signal\n"
        "hasattr(signal, 'SIGTERM') and signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(30)",
    ]
    cleaned = False

    def on_cleanup() -> None:
        nonlocal cleaned
        cleaned = True

    finished = pool.execute_run(
        record.run_id, worker_cmd=uncoop_cmd, timeout_seconds=0.3, on_cleanup=on_cleanup
    )

    assert (
        finished.status == "failed"
        and finished.error
        and finished.error.code == "timeout"
    )
    assert "exceeded wall-clock timeout" in finished.error.message and cleaned is True

    reloaded = run_store.get_run(record.run_id)
    assert (
        reloaded.status == "failed"
        and reloaded.error
        and reloaded.error.code == "timeout"
    )


@pytest.mark.unit
def test_worker_reaped_before_cleanup(
    run_store: SimulationRunStore, sample_request: RunRequest
) -> None:
    """Worker process must be strictly reaped before any cleanup occurs."""
    record = run_store.create_run(sample_request)
    pool = SimulationWorkerPool(run_store, max_concurrency=1)
    reaped_at_cleanup: list[bool] = []
    orig_reap, last_proc = reap_subprocess, []

    def mock_reap(proc: subprocess.Popen[Any], kill_timeout: float = 1.0) -> int:
        last_proc.append(proc)
        return orig_reap(proc, kill_timeout=kill_timeout)

    def on_cleanup() -> None:
        reaped_at_cleanup.append(bool(last_proc and last_proc[-1].poll() is not None))

    with patch("src.sim_run.worker.reap_subprocess", side_effect=mock_reap):
        pool.execute_run(
            record.run_id,
            worker_cmd=[sys.executable, "-c", "import sys; sys.exit(0)"],
            on_cleanup=on_cleanup,
        )

    assert reaped_at_cleanup == [True]


@pytest.mark.unit
def test_concurrency_bound(
    run_store: SimulationRunStore, sample_request: RunRequest
) -> None:
    """Pool enforces concurrency bound and rejects or blocks when saturated."""
    pool = SimulationWorkerPool(run_store, max_concurrency=1)
    rec1 = run_store.create_run(sample_request)
    rec2 = run_store.create_run(sample_request)
    sleep_cmd = [sys.executable, "-c", "import time; time.sleep(1.0)"]

    t1 = threading.Thread(
        target=pool.execute_run,
        args=(rec1.run_id,),
        kwargs={"worker_cmd": sleep_cmd, "timeout_seconds": 2.0},
    )
    t1.start()
    time.sleep(0.1)

    with pytest.raises(SimulationBusyError, match="Concurrency limit"):
        pool.execute_run(
            rec2.run_id, worker_cmd=sleep_cmd, timeout_seconds=2.0, block=False
        )
    t1.join()


@pytest.mark.unit
def test_successful_worker_execution(
    run_store: SimulationRunStore, sample_request: RunRequest
) -> None:
    """Valid worker execution produces succeeded status and populated ResultEnvelope."""
    rec = run_store.create_run(sample_request)
    fin = SimulationWorkerPool(run_store, max_concurrency=2).execute_run(rec.run_id)
    assert fin.status == "succeeded" and fin.result and fin.result.success
    assert fin.result.provenance.engine_type == "golf_swing_pendulum"


@pytest.mark.unit
def test_worker_nonzero_exit_failed(
    run_store: SimulationRunStore, sample_request: RunRequest
) -> None:
    """Non-zero worker exit code results in failed status with worker_error code."""
    rec = run_store.create_run(sample_request)
    fin = SimulationWorkerPool(run_store, max_concurrency=2).execute_run(
        rec.run_id,
        worker_cmd=[sys.executable, "-c", "import sys; sys.exit(2)"],
    )
    assert fin.status == "failed" and fin.error and fin.error.code == "worker_error"
    assert fin.error.details and fin.error.details.get("exit_code") == 2


@pytest.mark.unit
def test_linux_rlimits_contract() -> None:
    """apply_linux_rlimits documents behavior: applied on Linux, safe no-op on non-Linux."""
    limits = LinuxLimits(cpu_seconds=10, memory_bytes=1024 * 1024 * 512)
    assert apply_linux_rlimits(None) is False
    if sys.platform.startswith("linux"):
        mock_res = MagicMock()
        with patch.dict("sys.modules", {"resource": mock_res}):
            assert (
                apply_linux_rlimits(limits) is True
                and mock_res.setrlimit.call_count == 2
            )
    else:
        assert apply_linux_rlimits(limits) is False


@pytest.mark.unit
def test_run_simulation_job_direct(sample_request: RunRequest) -> None:
    """Direct execution of run_simulation_job produces valid ResultEnvelope."""
    envelope = run_simulation_job(sample_request, run_id="run_direct_123")
    assert envelope.run_id == "run_direct_123" and envelope.success is True
    assert envelope.provenance.input_sha256 is not None
