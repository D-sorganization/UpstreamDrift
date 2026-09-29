"""Driver Protocol — the execution backend the scheduler delegates to.

A :class:`Driver` is what actually runs a :class:`TrainingJob`. Three
backends are planned:

- :class:`InProcessDriver` (PR2, this module's neighbour) —
  thread-per-job in the launcher process. Default; testable.
- ``SubprocessDriver`` (PR3) — spawns a worker process per job; the
  worker uses the in-process driver internally. Survives launcher
  restarts via pidfile metadata.
- ``RayDriver`` (future) — distributes across a Ray cluster.

The scheduler talks to whichever driver it's configured with via this
narrow Protocol, so swapping backends is a one-line constructor
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ..contracts import CancelToken, ProgressSink
from ..errors import TrainingError
from ..identifiers import JobId
from ..job import RunResult, TrainingJob

__all__ = [
    "Driver",
    "DriverError",
    "JobHandle",
    "JobHandleStatus",
]


class DriverError(TrainingError):
    """Raised when a driver cannot start, cancel, or report on a job."""


class JobHandleStatus(Enum):
    """Lifecycle of a driver-side job handle.

    Distinct from :class:`TrainingStatus` — the handle tracks the
    *execution* state (is the thread / process alive?) while the
    scheduler tracks the *job* state (queued / running / failed). The
    two move in lock-step but the driver has no opinion about whether
    a finished job succeeded — that's encoded in the
    :class:`RunResult` it returns.
    """

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class JobHandle:
    """Driver-side reference to a started job.

    Returned by :meth:`Driver.start`; passed back to
    :meth:`Driver.cancel` and :meth:`Driver.result`. Opaque to the
    scheduler — only the driver itself looks inside.
    """

    job_id: JobId
    cancel_token: CancelToken

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, JobId):
            raise TypeError("job_id must be a JobId")
        if not isinstance(self.cancel_token, CancelToken):
            raise TypeError("cancel_token must satisfy CancelToken")


@runtime_checkable
class Driver(Protocol):
    """Backend Protocol the scheduler delegates execution to."""

    def start(
        self,
        job: TrainingJob,
        *,
        progress: ProgressSink,
    ) -> JobHandle:
        """Begin executing ``job``. Returns a handle for cancel/result.

        Implementations resolve a :class:`TrainingJobRunner` from
        their own registry, set up a :class:`CancelToken`, and arrange
        for the runner's :meth:`run` to be invoked.

        Preconditions:
            - ``job.status`` is :attr:`TrainingStatus.QUEUED` —
              callers (the scheduler) have already validated and
              registered the job.
        """

    def cancel(self, handle: JobHandle) -> None:
        """Signal cancellation. The runner observes via the cancel token."""

    def status(self, handle: JobHandle) -> JobHandleStatus:
        """Snapshot of the handle's execution state."""

    def result(self, handle: JobHandle, *, timeout: float | None = None) -> RunResult:
        """Block (up to ``timeout`` seconds) for the run to finish.

        Args:
            handle: The handle returned from :meth:`start`.
            timeout: ``None`` blocks indefinitely; ``0`` is a
                non-blocking poll.

        Raises:
            DriverError: When the job is not yet finished and the
                timeout elapses, or when the handle is unknown.
        """

    def shutdown(self, *, wait: bool = True) -> None:
        """Tear down driver resources (thread pool, subprocess, ...).

        If ``wait`` is ``True``, blocks until every in-flight job has
        observed the cancel signal and returned a :class:`RunResult`.
        """
