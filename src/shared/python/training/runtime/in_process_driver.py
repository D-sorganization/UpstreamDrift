"""Thread-per-job :class:`Driver` running in the launcher process.

Uses a bounded :class:`concurrent.futures.ThreadPoolExecutor` so the
driver naturally limits how many jobs run simultaneously — handy when
several PyTorch jobs would otherwise oversubscribe the GPU.
Cancellation is cooperative: the runner polls
:attr:`CancelToken.is_cancelled` between iterations and exits cleanly.

The driver wraps every runner invocation in error handling so a
runner exception becomes a :class:`RunResult` with status ``FAILED``
rather than killing the worker thread silently.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from ..contracts import ProgressSink, ThreadingCancelToken
from ..identifiers import new_run_id
from ..job import RunResult, TrainingJob
from ..status import TERMINAL_STATUSES, TrainingStatus
from .driver import DriverError, JobHandle, JobHandleStatus
from .runner_registry import RunnerRegistry

__all__ = ["InProcessDriver"]


logger = logging.getLogger(__name__)


_RUNNER_FAILURE_TYPES: tuple[type[BaseException], ...] = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    MemoryError,
    NotImplementedError,
    OSError,
    RecursionError,
    RuntimeError,
    TypeError,
    UnicodeError,
    ValueError,
)
"""Exception types the driver intercepts when invoking an adapter.

Deliberately wider than what ``narrow_catch`` permits, because the
driver is the trust boundary that converts arbitrary runner failures
into a ``FAILED`` :class:`RunResult` (without it, a buggy runner takes
the worker thread down silently). ``KeyboardInterrupt``,
``SystemExit``, and the rest of ``BaseException`` are intentionally
not included — those signal "stop everything" and must propagate.
"""


class InProcessDriver:
    """In-process, thread-per-job execution backend.

    Args:
        runner_registry: Lookup table for framework adapters.
        max_workers: Cap on simultaneous running jobs. Defaults to 4.

    Conforms to :class:`Driver` Protocol structurally; we don't inherit
    so the Protocol stays narrow.
    """

    __slots__ = (
        "_executor",
        "_futures",
        "_handles",
        "_lock",
        "_runner_registry",
        "_shutdown",
    )

    def __init__(
        self,
        runner_registry: RunnerRegistry,
        *,
        max_workers: int = 4,
    ) -> None:
        if not isinstance(runner_registry, RunnerRegistry):
            raise TypeError("runner_registry must be a RunnerRegistry")
        if not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError(
                f"max_workers must be a positive int (got {max_workers!r})"
            )
        self._runner_registry = runner_registry
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="training-driver"
        )
        self._futures: dict[JobHandle, Future[RunResult]] = {}
        self._handles: dict[JobHandle, ThreadingCancelToken] = {}
        self._lock = threading.RLock()
        self._shutdown = False

    @property
    def max_workers(self) -> int:
        return self._executor._max_workers  # type: ignore[attr-defined]

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
                    f"InProcessDriver.start expects QUEUED job "
                    f"(got {job.status.value!r})"
                )
            cancel_token = ThreadingCancelToken()
            handle = JobHandle(job_id=job.job_id, cancel_token=cancel_token)
            future = self._executor.submit(self._execute, job, progress, cancel_token)
            self._futures[handle] = future
            self._handles[handle] = cancel_token
            return handle

    def cancel(self, handle: JobHandle) -> None:
        with self._lock:
            token = self._handles.get(handle)
        if token is None:
            raise DriverError(f"unknown handle for job {handle.job_id.value!r}")
        token.request_cancel()

    def status(self, handle: JobHandle) -> JobHandleStatus:
        with self._lock:
            future = self._futures.get(handle)
        if future is None:
            raise DriverError(f"unknown handle for job {handle.job_id.value!r}")
        if future.done():
            return JobHandleStatus.FINISHED
        if future.running():
            return JobHandleStatus.RUNNING
        return JobHandleStatus.PENDING

    def result(self, handle: JobHandle, *, timeout: float | None = None) -> RunResult:
        with self._lock:
            future = self._futures.get(handle)
        if future is None:
            raise DriverError(f"unknown handle for job {handle.job_id.value!r}")
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            raise DriverError(
                f"job {handle.job_id.value!r} not finished within {timeout}s timeout"
            ) from exc

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            tokens = tuple(self._handles.values())
        for token in tokens:
            token.request_cancel()
        self._executor.shutdown(wait=wait)

    def _execute(
        self,
        job: TrainingJob,
        progress: ProgressSink,
        cancel_token: ThreadingCancelToken,
    ) -> RunResult:
        start_wall = time.monotonic()
        run_id = job.run_id if job.run_id is not None else new_run_id()
        try:
            runner = self._runner_registry.resolve(job.config)
        except _RUNNER_FAILURE_TYPES as exc:
            duration = time.monotonic() - start_wall
            logger.exception("runner resolution failed for job %s", job.job_id.value)
            return RunResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                duration_s=duration,
                error=f"runner resolution failed: {exc}",
            )
        try:
            runner.prepare(job.config)
        except _RUNNER_FAILURE_TYPES as exc:
            duration = time.monotonic() - start_wall
            logger.exception("runner.prepare raised for job %s", job.job_id.value)
            return RunResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                duration_s=duration,
                error=f"runner.prepare failed: {exc}",
            )
        try:
            result = runner.run(job.config, progress=progress, cancel=cancel_token)
        except _RUNNER_FAILURE_TYPES as exc:
            duration = time.monotonic() - start_wall
            logger.exception("runner.run raised for job %s", job.job_id.value)
            return RunResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                duration_s=duration,
                error=f"runner.run raised: {exc}",
            )
        if not isinstance(result, RunResult):
            duration = time.monotonic() - start_wall
            return RunResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                duration_s=duration,
                error=(
                    f"runner.run returned {type(result).__name__}, expected RunResult"
                ),
            )
        if result.status not in TERMINAL_STATUSES:
            duration = time.monotonic() - start_wall
            return RunResult(
                run_id=run_id,
                status=TrainingStatus.FAILED,
                duration_s=duration,
                error=(f"runner returned non-terminal status {result.status.value!r}"),
            )
        return result
