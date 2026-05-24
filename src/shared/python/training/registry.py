"""In-memory :class:`TrainingJob` registry.

The registry is the scheduler's local source of truth — every job
the scheduler knows about is stored here. It owns lookup, duplicate
detection, and atomic replace-by-id; it does *not* run jobs (that's
the driver's job, in :mod:`runtime.in_process_driver`).

Persistence (writing the registry to JSON on launcher shutdown,
reloading on next launch) lives in :mod:`persistence` and is wired in
PR5 alongside the resource monitor.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from typing import TypeAlias

from .errors import DuplicateJobError, JobNotFoundError
from .identifiers import JobId
from .job import TrainingJob
from .status import TrainingStatus

__all__ = ["JobRegistry", "JobFilter"]


JobFilter: TypeAlias = Callable[[TrainingJob], bool]
"""Predicate used by :meth:`JobRegistry.list` to filter results."""


class JobRegistry:
    """Thread-safe map from :class:`JobId` to :class:`TrainingJob`.

    All mutations take the same lock so concurrent submits / status
    updates from multiple threads don't race. Lookups also take the
    lock (briefly) — the cost is negligible at the rates a single host
    will ever produce training jobs.
    """

    __slots__ = ("_jobs", "_lock")

    def __init__(self, initial: Iterable[TrainingJob] | None = None) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[JobId, TrainingJob] = {}
        for job in initial or ():
            self.add(job)

    def add(self, job: TrainingJob) -> None:
        """Insert a brand-new job.

        Raises:
            DuplicateJobError: When ``job.job_id`` is already known.
        """

        if not isinstance(job, TrainingJob):
            raise TypeError(f"expected TrainingJob (got {type(job).__name__})")
        with self._lock:
            if job.job_id in self._jobs:
                raise DuplicateJobError(
                    f"job_id {job.job_id.value!r} is already registered"
                )
            self._jobs[job.job_id] = job

    def get(self, job_id: JobId) -> TrainingJob:
        """Look up a job by id.

        Raises:
            JobNotFoundError: When ``job_id`` is unknown.
        """

        if not isinstance(job_id, JobId):
            raise TypeError(f"expected JobId (got {type(job_id).__name__})")
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise JobNotFoundError(
                    f"no job registered with id {job_id.value!r}"
                ) from exc

    def has(self, job_id: JobId) -> bool:
        """``True`` when ``job_id`` is registered. Does not raise."""

        with self._lock:
            return job_id in self._jobs

    def replace(self, job: TrainingJob) -> TrainingJob:
        """Atomically replace an existing job with an updated copy.

        The caller is responsible for ensuring the replacement's id
        matches the original (the state-machine transition produces a
        new :class:`TrainingJob` with the same id via
        :meth:`TrainingJob.with_status`).

        Returns:
            The previous registry entry.

        Raises:
            JobNotFoundError: When ``job.job_id`` is unknown.
            TypeError: When ``job`` is not a :class:`TrainingJob`.
        """

        if not isinstance(job, TrainingJob):
            raise TypeError(f"expected TrainingJob (got {type(job).__name__})")
        with self._lock:
            if job.job_id not in self._jobs:
                raise JobNotFoundError(
                    f"no job registered with id {job.job_id.value!r}"
                )
            previous = self._jobs[job.job_id]
            self._jobs[job.job_id] = job
            return previous

    def remove(self, job_id: JobId) -> TrainingJob:
        """Remove and return the job for ``job_id``.

        Raises:
            JobNotFoundError: When ``job_id`` is unknown.
        """

        if not isinstance(job_id, JobId):
            raise TypeError(f"expected JobId (got {type(job_id).__name__})")
        with self._lock:
            try:
                return self._jobs.pop(job_id)
            except KeyError as exc:
                raise JobNotFoundError(
                    f"no job registered with id {job_id.value!r}"
                ) from exc

    def list(
        self,
        *,
        status: TrainingStatus | None = None,
        predicate: JobFilter | None = None,
    ) -> tuple[TrainingJob, ...]:
        """Snapshot list of jobs, optionally filtered.

        Both filters are AND-combined. ``status`` is a shortcut for
        the common "give me everything that is X" query; ``predicate``
        is the escape hatch for tag-based / time-based filtering.

        The result is a snapshot tuple — safe to iterate concurrently
        with further registry mutations.
        """

        with self._lock:
            snapshot = tuple(self._jobs.values())
        if status is not None:
            if not isinstance(status, TrainingStatus):
                raise TypeError(
                    f"status must be a TrainingStatus (got {type(status).__name__})"
                )
            snapshot = tuple(j for j in snapshot if j.status is status)
        if predicate is not None:
            snapshot = tuple(j for j in snapshot if predicate(j))
        return snapshot

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def __iter__(self) -> Iterator[TrainingJob]:
        with self._lock:
            return iter(tuple(self._jobs.values()))
