"""Training scheduler — admission, lifecycle, and observer surface.

The scheduler is the single object the GUI (PR3) and CLI / API layers
interact with for everything training-related. It owns:

- The :class:`JobRegistry` (what jobs exist, in what status).
- A :class:`RunnerRegistry` (which framework adapters are available).
- A :class:`Driver` (how jobs actually run — in-process by default).
- An optional :class:`CompatibilityChecker` (idiot-proof engine gate).
- A list of observers (functions called whenever a job's status
  changes — the dashboard subscribes to this).

The scheduler does *no* training itself; it choreographs registry
updates and driver calls. Every state transition goes through
:meth:`TrainingJob.with_status`, so the state machine is the single
source of truth for what's legal.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .compatibility import CompatibilityChecker, CompatibilityReport
from .config import TrainingConfig
from .contracts import ProgressSink
from .errors import CompatibilityError, JobNotFoundError, TrainingError
from .identifiers import JobId, new_job_id, new_run_id
from .job import RunResult, TrainingJob
from .registry import JobRegistry
from .runtime.driver import Driver, JobHandle
from .runtime.progress_sinks import NullProgressSink
from .runtime.runner_registry import RunnerRegistry
from .status import TrainingStatus

__all__ = [
    "Scheduler",
    "SchedulerError",
    "StatusChangeEvent",
]


logger = logging.getLogger(__name__)


class SchedulerError(TrainingError):
    """Raised when the scheduler cannot satisfy a request.

    Sub-cases (compatibility, missing runner, missing job) raise their
    own dedicated exception types from :mod:`errors` /
    :mod:`runtime.runner_registry`; this is the catch-all for cases
    the scheduler itself decides about (e.g. illegal cancel-of-cancel).
    """


@dataclass(frozen=True, slots=True)
class StatusChangeEvent:
    """Observation passed to every registered status-change observer."""

    job: TrainingJob
    previous_status: TrainingStatus
    new_status: TrainingStatus
    timestamp: float


StatusObserver = Callable[[StatusChangeEvent], None]
"""Type of callbacks registered with :meth:`Scheduler.on_status_change`."""

ProgressSinkFactory = Callable[[TrainingJob], ProgressSink]
"""Per-job sink factory; the scheduler invokes it when starting a job."""


_RUNNING_HANDLES_LOCK_NAME = "scheduler"


class Scheduler:
    """Coordinates training-job admission, lifecycle, and observation.

    Args:
        registry: Job registry the scheduler owns. Each scheduler
            should have its own registry; sharing across schedulers is
            not supported.
        runners: Lookup for per-framework adapters.
        driver: Execution backend. Defaults to in-process; subprocess
            and ray backends slot in here.
        compatibility_checker: Optional engine-compat gate. When
            provided, :meth:`submit` calls
            :meth:`CompatibilityChecker.check` against the configured
            target engine before transitioning to ``QUEUED``.
        progress_sink_factory: Builds the :class:`ProgressSink` for
            each job. Defaults to a factory that returns a shared
            :class:`NullProgressSink`. PR3 wires the dashboard's
            realtime channel sink here.
        clock: Function returning current wall-clock time (seconds).
            Defaulted to :func:`time.time` so tests can inject a
            deterministic clock.
    """

    __slots__ = (
        "_clock",
        "_compat",
        "_driver",
        "_handles",
        "_lock",
        "_observers",
        "_progress_factory",
        "_registry",
        "_runners",
    )

    def __init__(
        self,
        *,
        registry: JobRegistry,
        runners: RunnerRegistry,
        driver: Driver,
        compatibility_checker: CompatibilityChecker | None = None,
        progress_sink_factory: ProgressSinkFactory | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(registry, JobRegistry):
            raise TypeError("registry must be a JobRegistry")
        if not isinstance(runners, RunnerRegistry):
            raise TypeError("runners must be a RunnerRegistry")
        if not isinstance(driver, Driver):
            raise TypeError("driver must satisfy the Driver Protocol")
        if compatibility_checker is not None and not isinstance(
            compatibility_checker, CompatibilityChecker
        ):
            raise TypeError(
                "compatibility_checker must be a CompatibilityChecker or None"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._registry = registry
        self._runners = runners
        self._driver = driver
        self._compat = compatibility_checker
        self._progress_factory = progress_sink_factory or (
            lambda _job: NullProgressSink()
        )
        self._clock = clock
        self._observers: list[StatusObserver] = []
        self._handles: dict[JobId, JobHandle] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ submit

    def submit(
        self,
        config: TrainingConfig,
        *,
        target_engine: str | None = None,
        job_id: JobId | None = None,
    ) -> TrainingJob:
        """Validate, register, and queue a training job.

        Args:
            config: The validated configuration.
            target_engine: Optional engine name. When provided AND a
                ``compatibility_checker`` is configured, the
                checker runs first and an incompatible pairing raises
                :class:`CompatibilityError` *before* anything is
                registered.
            job_id: Optional explicit id (for retries / re-submits).
                Defaults to a fresh UUID4-hex id.

        Returns:
            The newly-registered job in :attr:`TrainingStatus.QUEUED`.

        Raises:
            CompatibilityError: When the compatibility check fails.
            NoRunnerAvailableError: When no runner is registered for
                the config's framework.
        """

        if not isinstance(config, TrainingConfig):
            raise TypeError(f"expected TrainingConfig (got {type(config).__name__})")
        if target_engine is not None and self._compat is not None:
            report = self._compat.check(config, target_engine)
            if not report.is_compatible:
                raise self._compat_error(report, target_engine)
        # Fail fast if no runner is registered, so users see the
        # mistake at submit time, not minutes later when the driver
        # tries to start the job.
        self._runners.resolve(config)
        identifier = job_id if job_id is not None else new_job_id()
        now = self._clock()
        pending = TrainingJob(
            job_id=identifier,
            config=config,
            status=TrainingStatus.PENDING,
            created_at=now,
        )
        self._registry.add(pending)
        queued = pending.with_status(TrainingStatus.QUEUED, now=now)
        self._registry.replace(queued)
        self._notify(pending, queued, now)
        return queued

    @staticmethod
    def _compat_error(report: CompatibilityReport, engine: str) -> CompatibilityError:
        msgs = "; ".join(i.message for i in report.errors)
        return CompatibilityError(
            f"engine {engine!r} is incompatible with the submitted config: {msgs}"
        )

    # ------------------------------------------------------------------ start

    def start(self, job_id: JobId) -> TrainingJob:
        """Move a ``QUEUED`` job to ``RUNNING`` and hand it to the driver."""

        job = self._registry.get(job_id)
        if job.status is not TrainingStatus.QUEUED:
            raise SchedulerError(
                f"cannot start job {job_id.value!r} in status {job.status.value!r}"
            )
        now = self._clock()
        run_id = new_run_id()
        job_with_run = TrainingJob(
            job_id=job.job_id,
            config=job.config,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            run_id=run_id,
        )
        progress = self._progress_factory(job_with_run)
        handle = self._driver.start(job_with_run, progress=progress)
        running = job_with_run.with_status(TrainingStatus.RUNNING, now=now)
        with self._lock:
            self._handles[job_id] = handle
        self._registry.replace(running)
        self._notify(job, running, now)
        return running

    # ------------------------------------------------------------------ cancel

    def cancel(self, job_id: JobId) -> TrainingJob:
        """Signal cancellation. The runner observes via the cancel token.

        Transitions the registry entry to ``CANCELLED`` immediately so
        the dashboard updates without waiting for the runner to wind
        down. The driver still observes the cancel token and returns
        a terminal :class:`RunResult` asynchronously; :meth:`collect`
        reconciles the two once the run is finished.
        """

        job = self._registry.get(job_id)
        if job.status.is_terminal:
            raise SchedulerError(
                f"cannot cancel job {job_id.value!r} in terminal status "
                f"{job.status.value!r}"
            )
        with self._lock:
            handle = self._handles.get(job_id)
        if handle is not None:
            self._driver.cancel(handle)
        now = self._clock()
        cancelled = job.with_status(TrainingStatus.CANCELLED, now=now)
        self._registry.replace(cancelled)
        self._notify(job, cancelled, now)
        return cancelled

    # ------------------------------------------------------------------ pause / resume

    def pause(self, job_id: JobId) -> TrainingJob:
        """Mark a ``RUNNING`` job as ``PAUSED``.

        Note: this is a *registry-level* pause — the underlying runner
        does not necessarily halt. PR5 introduces a richer cooperative
        pause via a ``PauseToken``; for now this is the lightweight
        intent marker the dashboard reads.
        """

        job = self._registry.get(job_id)
        if job.status is not TrainingStatus.RUNNING:
            raise SchedulerError(
                f"can only pause RUNNING jobs (got {job.status.value!r})"
            )
        now = self._clock()
        paused = job.with_status(TrainingStatus.PAUSED, now=now)
        self._registry.replace(paused)
        self._notify(job, paused, now)
        return paused

    def resume(self, job_id: JobId) -> TrainingJob:
        """Transition a ``PAUSED`` job back to ``RUNNING``."""

        job = self._registry.get(job_id)
        if job.status is not TrainingStatus.PAUSED:
            raise SchedulerError(
                f"can only resume PAUSED jobs (got {job.status.value!r})"
            )
        now = self._clock()
        running = job.with_status(TrainingStatus.RUNNING, now=now)
        self._registry.replace(running)
        self._notify(job, running, now)
        return running

    # ------------------------------------------------------------------ collect

    def collect(self, job_id: JobId, *, timeout: float | None = None) -> RunResult:
        """Block until a job's driver-side run finishes, then reconcile.

        Returns the :class:`RunResult` the driver produced and updates
        the registry to the corresponding terminal status. Safe to
        call after :meth:`cancel`; the result's status will normally
        be ``CANCELLED`` in that case.
        """

        with self._lock:
            handle = self._handles.get(job_id)
        if handle is None:
            raise SchedulerError(
                f"no driver handle for job {job_id.value!r} (was it started?)"
            )
        result = self._driver.result(handle, timeout=timeout)
        job = self._registry.get(job_id)
        if job.status.is_terminal:
            # cancel() already updated registry; preserve that.
            return result
        now = self._clock()
        if result.status is TrainingStatus.FAILED:
            updated = job.with_status(
                TrainingStatus.FAILED,
                now=now,
                error_message=result.error or "runner reported failure",
            )
        elif result.status in (TrainingStatus.COMPLETED, TrainingStatus.CANCELLED):
            updated = job.with_status(result.status, now=now)
        else:  # pragma: no cover - driver guarantees terminal
            raise SchedulerError(
                f"driver returned non-terminal status {result.status.value!r}"
            )
        self._registry.replace(updated)
        self._notify(job, updated, now)
        return result

    # ------------------------------------------------------------------ observers

    def on_status_change(self, observer: StatusObserver) -> Callable[[], None]:
        """Register a status-change observer. Returns an unsubscribe fn."""

        if not callable(observer):
            raise TypeError("observer must be callable")
        with self._lock:
            self._observers.append(observer)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._observers.remove(observer)
                except ValueError:
                    return

        return _unsubscribe

    def _notify(
        self,
        previous: TrainingJob,
        new: TrainingJob,
        timestamp: float,
    ) -> None:
        event = StatusChangeEvent(
            job=new,
            previous_status=previous.status,
            new_status=new.status,
            timestamp=timestamp,
        )
        with self._lock:
            observers = tuple(self._observers)
        for observer in observers:
            try:
                observer(event)
            except (RuntimeError, ValueError, TypeError, OSError, LookupError):
                logger.exception("scheduler observer raised; continuing fan-out")

    # ------------------------------------------------------------------ misc

    @property
    def registry(self) -> JobRegistry:
        return self._registry

    @property
    def runners(self) -> RunnerRegistry:
        return self._runners

    def get(self, job_id: JobId) -> TrainingJob:
        """Convenience: registry passthrough."""

        return self._registry.get(job_id)

    def shutdown(self, *, wait: bool = True) -> None:
        """Tear the scheduler down. Cancels and joins every in-flight job."""

        with self._lock:
            running = tuple(self._handles.items())
        for job_id, _ in running:
            try:
                self.cancel(job_id)
            except (SchedulerError, JobNotFoundError):
                logger.exception("error cancelling job %s during shutdown", job_id)
        self._driver.shutdown(wait=wait)
