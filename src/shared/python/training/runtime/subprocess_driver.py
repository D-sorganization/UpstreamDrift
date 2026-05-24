"""Subprocess-isolated :class:`Driver` for the training scheduler.

Where :class:`InProcessDriver` runs every training job on a thread in
the launcher process, :class:`SubprocessDriver` spawns one worker
process per job (see :mod:`training.runtime.worker_main`) and talks to
it over the JSON wire protocol in :mod:`training.runtime.wire_protocol`.
This gives:

* **Crash isolation** — a segfaulting framework adapter kills its
  worker, not the launcher.
* **GPU isolation** — separate processes can pin to separate CUDA
  devices via env vars without inter-job interference.
* **Restart resilience** — a pidfile in the job's ``output_dir`` lets
  the launcher detect orphaned workers on restart and clean them up.

The Driver Protocol is identical to the in-process driver's so the
scheduler swap is a one-line constructor change.

Subprocess management goes through
:func:`core.process_safety.managed_popen` (mandatory per the
error-handling ratchet — see ``CLAUDE.md`` "Error handling"). We open
the context inside a worker-thread and hold the proc live for the job
duration; cleanup on exit funnels through ``managed_popen``'s
SIGTERM → SIGKILL escalation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from src.shared.python.core.process_safety import managed_popen
from src.shared.python.logging_pkg.logging_config import get_logger

from ..contracts import ProgressSink, ThreadingCancelToken
from ..identifiers import RunId, new_run_id
from ..job import RunResult, TrainingJob
from ..metrics import TrainingMetric
from ..persistence import (
    run_result_from_dict,
    training_config_to_dict,
    training_metric_from_dict,
)
from ..status import TrainingStatus
from .driver import DriverError, JobHandle, JobHandleStatus
from .wire_protocol import (
    COMMAND_CANCEL,
    COMMAND_RUN,
    EVENT_METRIC,
    EVENT_RESULT,
    EVENT_STATUS,
    WireProtocolError,
    decode_event,
    encode_command,
)

__all__ = ["SubprocessDriver", "scan_pidfiles"]


logger = get_logger(__name__)


_PIDFILE_NAME = ".training.pid"
_DEFAULT_CANCEL_TIMEOUT_S = 10.0
_DEFAULT_KILL_TIMEOUT_S = 5.0


# Mirrors `in_process_driver._RUNNER_FAILURE_TYPES`. The driver is the
# trust boundary between worker IPC and the scheduler — anything not
# in this tuple (`KeyboardInterrupt`, `SystemExit`, …) propagates so
# the scheduler can shut down cleanly.
_DRIVER_FAILURE_TYPES: tuple[type[BaseException], ...] = (
    ArithmeticError,
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class _JobState:
    """Mutable per-job bookkeeping owned by the driver thread."""

    __slots__ = (
        "cancel_token",
        "future",
        "handle",
        "job",
        "pidfile",
        "stderr_buffer",
        "thread",
    )

    def __init__(
        self,
        handle: JobHandle,
        job: TrainingJob,
        cancel_token: ThreadingCancelToken,
        pidfile: Path | None,
    ) -> None:
        self.handle = handle
        self.job = job
        self.cancel_token = cancel_token
        self.pidfile = pidfile
        self.future: Future[RunResult] = Future()
        self.thread: threading.Thread | None = None
        self.stderr_buffer: list[str] = []


class SubprocessDriver:
    """One-worker-per-job execution backend.

    Args:
        worker_command: Optional override for the subprocess command;
            defaults to ``[sys.executable, "-m", "training.runtime.worker_main"]``.
            Tests can override to inject a fixture wrapper.
        worker_env: Optional override for the worker's environment;
            defaults to ``os.environ``. Pass a custom mapping to set
            ``CUDA_VISIBLE_DEVICES`` and similar isolation knobs.
        cancel_timeout_s: Seconds to wait after writing ``cancel`` to
            the worker's stdin before falling back to
            ``managed_popen``'s SIGTERM → SIGKILL escalation. Defaults
            to 10 s.
        kill_timeout_s: Seconds between SIGTERM and SIGKILL escalation
            on shutdown. Forwarded to :func:`managed_popen`.

    Conforms to :class:`Driver` Protocol structurally.
    """

    __slots__ = (
        "_cancel_timeout_s",
        "_kill_timeout_s",
        "_lock",
        "_shutdown",
        "_states",
        "_worker_command",
        "_worker_env",
    )

    def __init__(
        self,
        *,
        worker_command: Sequence[str] | None = None,
        worker_env: dict[str, str] | None = None,
        cancel_timeout_s: float = _DEFAULT_CANCEL_TIMEOUT_S,
        kill_timeout_s: float = _DEFAULT_KILL_TIMEOUT_S,
    ) -> None:
        if cancel_timeout_s <= 0:
            raise ValueError(
                f"cancel_timeout_s must be positive (got {cancel_timeout_s!r})"
            )
        if kill_timeout_s < 0:
            raise ValueError(
                f"kill_timeout_s must be non-negative (got {kill_timeout_s!r})"
            )
        if worker_command is None:
            worker_command = (
                sys.executable,
                "-m",
                "training.runtime.worker_main",
            )
        self._worker_command: tuple[str, ...] = tuple(worker_command)
        if not self._worker_command:
            raise ValueError("worker_command must not be empty")
        self._worker_env: dict[str, str] | None = (
            dict(worker_env) if worker_env is not None else None
        )
        self._cancel_timeout_s = float(cancel_timeout_s)
        self._kill_timeout_s = float(kill_timeout_s)
        self._lock = threading.RLock()
        self._states: dict[JobHandle, _JobState] = {}
        self._shutdown = False

    # ------------------------------------------------------------------
    # Driver Protocol
    # ------------------------------------------------------------------

    def start(
        self,
        job: TrainingJob,
        *,
        progress: ProgressSink,
    ) -> JobHandle:
        with self._lock:
            if self._shutdown:
                raise DriverError("driver has been shut down")
            if job.status is not TrainingStatus.QUEUED:
                raise DriverError(
                    f"SubprocessDriver.start expects QUEUED job "
                    f"(got {job.status.value!r})"
                )
            cancel_token = ThreadingCancelToken()
            handle = JobHandle(job_id=job.job_id, cancel_token=cancel_token)
            pidfile = self._pidfile_for(job)
            state = _JobState(handle, job, cancel_token, pidfile)
            self._states[handle] = state

        thread = threading.Thread(
            target=self._execute,
            name=f"subprocess-driver-{job.job_id.value}",
            args=(state, progress),
            daemon=True,
        )
        state.thread = thread
        thread.start()
        return handle

    def cancel(self, handle: JobHandle) -> None:
        with self._lock:
            state = self._states.get(handle)
        if state is None:
            raise DriverError(f"unknown handle for job {handle.job_id.value!r}")
        state.cancel_token.request_cancel()
        # Best-effort cancel command — the worker may have already
        # closed stdin or be unresponsive; managed_popen's escalation
        # handles the latter when the run thread tears the proc down.

    def status(self, handle: JobHandle) -> JobHandleStatus:
        with self._lock:
            state = self._states.get(handle)
        if state is None:
            raise DriverError(f"unknown handle for job {handle.job_id.value!r}")
        if state.future.done():
            return JobHandleStatus.FINISHED
        thread = state.thread
        if thread is not None and thread.is_alive():
            return JobHandleStatus.RUNNING
        return JobHandleStatus.PENDING

    def result(self, handle: JobHandle, *, timeout: float | None = None) -> RunResult:
        with self._lock:
            state = self._states.get(handle)
        if state is None:
            raise DriverError(f"unknown handle for job {handle.job_id.value!r}")
        try:
            return state.future.result(timeout=timeout)
        except TimeoutError as exc:
            raise DriverError(
                f"job {handle.job_id.value!r} not finished within {timeout}s timeout"
            ) from exc

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            states = tuple(self._states.values())
        for state in states:
            state.cancel_token.request_cancel()
        if wait:
            for state in states:
                thread = state.thread
                if thread is not None:
                    thread.join()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _execute(self, state: _JobState, progress: ProgressSink) -> None:
        """Per-job worker thread: spawn child, pump events, store result."""

        run_id = state.job.run_id if state.job.run_id is not None else new_run_id()
        start_wall = time.monotonic()
        result: RunResult
        try:
            result = self._run_worker(state, progress, run_id, start_wall)
        except _DRIVER_FAILURE_TYPES as exc:
            logger.exception(
                "subprocess driver failed for job %s", state.job.job_id.value
            )
            result = RunResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                duration_s=time.monotonic() - start_wall,
                error=f"driver: {exc}",
            )
        # Remove pidfile BEFORE publishing the result so callers that
        # poll on `result()` and then check the filesystem don't race.
        self._remove_pidfile(state)
        state.future.set_result(result)

    def _run_worker(
        self,
        state: _JobState,
        progress: ProgressSink,
        run_id: RunId,
        start_wall: float,
    ) -> RunResult:
        config_dict = training_config_to_dict(state.job.config)
        run_command = encode_command(COMMAND_RUN, {"config": config_dict})
        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
        }
        if self._worker_env is not None:
            popen_kwargs["env"] = dict(self._worker_env)

        result_holder: dict[str, RunResult | None] = {"value": None}
        with managed_popen(
            list(self._worker_command),
            kill_timeout=self._kill_timeout_s,
            **popen_kwargs,
        ) as proc:
            self._write_pidfile(state, proc.pid)
            self._send_run(proc, run_command, state, run_id, start_wall, result_holder)
            if result_holder["value"] is not None:
                # Failed before we got to read stdout.
                return result_holder["value"]

            stderr_thread = self._spawn_stderr_drain(proc, state)
            cancel_watcher = self._spawn_cancel_watcher(proc, state)

            final_result = self._read_event_stream(
                proc, progress, state, run_id, start_wall
            )

            cancel_watcher.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

        return final_result

    def _send_run(
        self,
        proc: Any,
        run_command: str,
        state: _JobState,
        run_id: RunId,
        start_wall: float,
        result_holder: dict[str, RunResult | None],
    ) -> None:
        try:
            assert proc.stdin is not None
            proc.stdin.write(run_command)
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            logger.exception(
                "failed to send run command to worker for job %s",
                state.job.job_id.value,
            )
            result_holder["value"] = RunResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                duration_s=time.monotonic() - start_wall,
                error=f"driver: failed to send run command: {exc}",
            )

    def _spawn_cancel_watcher(
        self,
        proc: Any,
        state: _JobState,
    ) -> threading.Thread:
        """Forward cancel-token flips into a stdin command for the worker."""

        def _watch() -> None:
            while not state.cancel_token.is_cancelled:
                if proc.poll() is not None:
                    return
                time.sleep(0.05)
            if proc.poll() is not None or proc.stdin is None:
                return
            try:
                proc.stdin.write(encode_command(COMMAND_CANCEL))
                proc.stdin.flush()
            except (OSError, ValueError):
                # Worker may have closed stdin already — that's fine,
                # the wait/timeout path will escalate via managed_popen.
                return

        thread = threading.Thread(
            target=_watch,
            name=f"subprocess-cancel-watcher-{state.job.job_id.value}",
            daemon=True,
        )
        thread.start()
        return thread

    def _spawn_stderr_drain(
        self,
        proc: Any,
        state: _JobState,
    ) -> threading.Thread:
        """Drain worker stderr into ``state.stderr_buffer`` for diagnostics."""

        def _drain() -> None:
            if proc.stderr is None:
                return
            try:
                for line in proc.stderr:
                    state.stderr_buffer.append(line)
            except (OSError, ValueError):
                return

        thread = threading.Thread(
            target=_drain,
            name=f"subprocess-stderr-drain-{state.job.job_id.value}",
            daemon=True,
        )
        thread.start()
        return thread

    def _read_event_stream(
        self,
        proc: Any,
        progress: ProgressSink,
        state: _JobState,
        run_id: RunId,
        start_wall: float,
    ) -> RunResult:
        """Pump JSON events from the worker's stdout until result or EOF."""

        final_result: RunResult | None = None
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event_name, payload = decode_event(line)
            except WireProtocolError:
                logger.warning(
                    "subprocess driver dropping malformed worker line: %s", line
                )
                continue
            if event_name == EVENT_STATUS:
                self._dispatch_status(progress, payload)
            elif event_name == EVENT_METRIC:
                self._dispatch_metric(progress, payload)
            elif event_name == EVENT_RESULT:
                final_result = self._decode_result(payload, run_id, start_wall)
                # Result is contractually the final event; keep reading
                # in case the worker writes a trailing newline, but
                # don't expect more events.

        if final_result is not None:
            return final_result

        # Stdout closed before we got a result — synthesize FAILED.
        stderr_text = "".join(state.stderr_buffer).strip()
        message = "worker exited without emitting a result"
        if stderr_text:
            message = f"{message}; stderr: {stderr_text}"
        return RunResult(
            run_id=run_id,
            status=TrainingStatus.FAILED,
            duration_s=time.monotonic() - start_wall,
            error=f"driver: {message}",
        )

    def _dispatch_status(
        self,
        progress: ProgressSink,
        payload: dict[str, Any],
    ) -> None:
        status_raw = payload.get("status")
        try:
            status = TrainingStatus(status_raw)
        except ValueError:
            logger.warning("subprocess driver got unknown status %r", status_raw)
            return
        message = payload.get("message")
        if message is not None and not isinstance(message, str):
            message = str(message)
        progress.emit_status(status, message=message)

    def _dispatch_metric(
        self,
        progress: ProgressSink,
        payload: dict[str, Any],
    ) -> None:
        metric_dict = payload.get("metric")
        if not isinstance(metric_dict, dict):
            logger.warning("subprocess driver got metric event without 'metric' dict")
            return
        try:
            metric: TrainingMetric = training_metric_from_dict(metric_dict)
        except _DRIVER_FAILURE_TYPES:
            logger.exception("subprocess driver could not decode metric")
            return
        progress.emit_metric(metric)

    def _decode_result(
        self,
        payload: dict[str, Any],
        run_id: RunId,
        start_wall: float,
    ) -> RunResult:
        result_dict = payload.get("result")
        if not isinstance(result_dict, dict):
            return RunResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                duration_s=time.monotonic() - start_wall,
                error="driver: result event missing 'result' dict",
            )
        try:
            return run_result_from_dict(result_dict)
        except _DRIVER_FAILURE_TYPES as exc:
            logger.exception("subprocess driver could not decode result")
            return RunResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                duration_s=time.monotonic() - start_wall,
                error=f"driver: malformed result event: {exc}",
            )

    # ------------------------------------------------------------------
    # Pidfile bookkeeping
    # ------------------------------------------------------------------

    def _pidfile_for(self, job: TrainingJob) -> Path | None:
        output_dir = job.config.output_dir
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("could not prepare output_dir %s for pidfile", output_dir)
            return None
        return output_dir / _PIDFILE_NAME

    def _write_pidfile(self, state: _JobState, pid: int) -> None:
        if state.pidfile is None:
            return
        try:
            state.pidfile.write_text(f"{pid}\n", encoding="utf-8")
        except OSError:
            logger.exception(
                "could not write pidfile %s for job %s",
                state.pidfile,
                state.job.job_id.value,
            )

    def _remove_pidfile(self, state: _JobState) -> None:
        if state.pidfile is None:
            return
        try:
            state.pidfile.unlink(missing_ok=True)
        except OSError:
            logger.exception(
                "could not remove pidfile %s for job %s",
                state.pidfile,
                state.job.job_id.value,
            )


def scan_pidfiles(output_dirs: Sequence[Path]) -> dict[Path, int | None]:
    """Inspect ``output_dirs`` for ``.training.pid`` files.

    Args:
        output_dirs: Directories that previously hosted training jobs.

    Returns:
        Mapping of pidfile path → ``pid`` (int) if the recorded process
        appears alive, ``None`` if the recorded pid is dead or the
        pidfile is malformed. Missing pidfiles are omitted from the
        result entirely. Used by the launcher on restart to surface
        reattach candidates and to mark orphans FAILED.
    """

    findings: dict[Path, int | None] = {}
    for output_dir in output_dirs:
        pidfile = Path(output_dir) / _PIDFILE_NAME
        if not pidfile.is_file():
            continue
        try:
            raw = pidfile.read_text(encoding="utf-8").strip()
            pid = int(raw)
        except (OSError, ValueError):
            logger.warning("malformed pidfile at %s", pidfile)
            findings[pidfile] = None
            continue
        findings[pidfile] = pid if _pid_is_alive(pid) else None
    return findings


def _pid_is_alive(pid: int) -> bool:
    """POSIX-safe ``kill(pid, 0)`` liveness probe.

    Returns ``False`` for non-positive pids without calling ``kill``;
    on Windows (no ``os.kill`` with signal 0 semantics) we fall back to
    a permissive ``True`` since the launcher's reattach UI can still
    re-probe via its own mechanism.
    """

    if pid <= 0:
        return False
    if not hasattr(os, "kill"):
        return True  # pragma: no cover - exotic platforms
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack signal rights — treat as alive.
        return True
    except OSError:
        return False
    return True
