"""Training-job record and final-run result.

A :class:`TrainingJob` is the persistent, dashboard-visible record:
identifier, config, current status, and lifecycle timestamps. It is
immutable; transitions are made by constructing a new instance via
:meth:`TrainingJob.with_status`, which enforces the state-machine
rules from :mod:`status`.

A :class:`RunResult` is the worker's final report for a single
execution attempt. Status is terminal-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Self

from .config import TrainingConfig
from .errors import TrainingConfigError
from .identifiers import JobId, RunId
from .metrics import TrainingMetric
from .status import TrainingStatus, validate_transition

__all__ = ["RunResult", "TrainingJob"]


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """Immutable record of a scheduled training job.

    Attributes:
        job_id: Stable identifier for the job (survives restarts).
        config: The validated configuration.
        status: Current lifecycle status.
        created_at: Wall-clock time the job was created.
        started_at: When the job first transitioned to ``RUNNING``.
            ``None`` until that happens.
        completed_at: When the job entered a terminal status.
            ``None`` until that happens.
        error_message: Failure reason; required when ``status`` is
            :attr:`TrainingStatus.FAILED`, otherwise ``None``.
        run_id: Identifier of the current execution attempt. ``None``
            until the worker assigns one.

    Invariants:
        - ``started_at`` (when set) is ``>= created_at``.
        - ``completed_at`` (when set) is ``>= started_at`` (which must
          also be set) — a job cannot complete without having started.
        - When ``status`` is terminal, ``completed_at`` is set.
        - When ``status == FAILED``, ``error_message`` is a non-empty
          string. Otherwise ``error_message`` is ``None``.
    """

    job_id: JobId
    config: TrainingConfig
    status: TrainingStatus
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    run_id: RunId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, JobId):
            raise TrainingConfigError("job_id must be a JobId")
        if not isinstance(self.config, TrainingConfig):
            raise TrainingConfigError("config must be a TrainingConfig")
        if not isinstance(self.status, TrainingStatus):
            raise TrainingConfigError("status must be a TrainingStatus")
        if not isinstance(self.created_at, (int, float)) or self.created_at < 0:
            raise TrainingConfigError(
                f"created_at must be a non-negative number (got {self.created_at!r})"
            )
        self._validate_timestamps()
        self._validate_error_message()
        if self.run_id is not None and not isinstance(self.run_id, RunId):
            raise TrainingConfigError("run_id must be a RunId or None")
        object.__setattr__(self, "created_at", float(self.created_at))
        if self.started_at is not None:
            object.__setattr__(self, "started_at", float(self.started_at))
        if self.completed_at is not None:
            object.__setattr__(self, "completed_at", float(self.completed_at))

    def _validate_timestamps(self) -> None:
        if self.started_at is not None and (
            not isinstance(self.started_at, (int, float))
            or self.started_at < self.created_at
        ):
            raise TrainingConfigError(
                "started_at must be a number >= created_at "
                f"(got {self.started_at!r}, created_at={self.created_at!r})"
            )
        if self.completed_at is not None:
            # COMPLETED / FAILED require a run actually happened; CANCELLED
            # may occur from PENDING or QUEUED before the job ever started.
            if self.started_at is None and self.status is not TrainingStatus.CANCELLED:
                raise TrainingConfigError(
                    "completed_at cannot be set without started_at "
                    f"for status {self.status.value!r}"
                )
            floor = self.started_at if self.started_at is not None else self.created_at
            if (
                not isinstance(self.completed_at, (int, float))
                or self.completed_at < floor
            ):
                raise TrainingConfigError(
                    f"completed_at must be a number >= {'started_at' if self.started_at is not None else 'created_at'} "
                    f"(got {self.completed_at!r}, floor={floor!r})"
                )
        if self.status.is_terminal and self.completed_at is None:
            raise TrainingConfigError(
                f"completed_at must be set for terminal status {self.status.value!r}"
            )

    def _validate_error_message(self) -> None:
        if self.status == TrainingStatus.FAILED:
            if (
                not isinstance(self.error_message, str)
                or not self.error_message.strip()
            ):
                raise TrainingConfigError(
                    "error_message must be a non-empty string when status is FAILED"
                )
        elif self.error_message is not None:
            raise TrainingConfigError(
                f"error_message must be None when status is {self.status.value!r}"
            )

    def with_status(
        self,
        new_status: TrainingStatus,
        *,
        now: float,
        error_message: str | None = None,
        run_id: RunId | None = None,
    ) -> Self:
        """Return a copy with ``status`` updated, enforcing transition rules.

        Args:
            new_status: Target status.
            now: Wall-clock time of the transition. Used to populate
                ``started_at`` / ``completed_at`` automatically.
            error_message: Required when ``new_status == FAILED``;
                forbidden otherwise.
            run_id: Required when transitioning to ``RUNNING`` for the
                first time; ignored otherwise.

        Raises:
            InvalidStatusTransitionError: When the transition is not
                permitted by the state machine.
            TrainingConfigError: When ``error_message`` / ``run_id``
                preconditions are violated.
        """

        validate_transition(self.status, new_status)
        if not isinstance(now, (int, float)) or now < self.created_at:
            raise TrainingConfigError(
                f"now must be a number >= created_at (got {now!r})"
            )
        kwargs: dict[str, object] = {"status": new_status}
        if new_status == TrainingStatus.RUNNING and self.started_at is None:
            kwargs["started_at"] = float(now)
            if run_id is None and self.run_id is None:
                raise TrainingConfigError(
                    "run_id must be provided on first transition to RUNNING"
                )
            if run_id is not None:
                kwargs["run_id"] = run_id
        if new_status.is_terminal:
            kwargs["completed_at"] = float(now)
        if new_status == TrainingStatus.FAILED:
            if not isinstance(error_message, str) or not error_message.strip():
                raise TrainingConfigError(
                    "error_message required for transition to FAILED"
                )
            kwargs["error_message"] = error_message
        elif error_message is not None:
            raise TrainingConfigError(
                "error_message must be None for non-FAILED transitions"
            )
        return replace(self, **kwargs)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Final report from one execution attempt of a job.

    Attributes:
        run_id: The execution-attempt identifier.
        status: Terminal status — one of ``COMPLETED``, ``FAILED``,
            ``CANCELLED``.
        final_metrics: Last observation per distinct metric name. The
            scheduler writes these to disk and surfaces them in the
            dashboard's "summary" panel.
        artifacts: Filesystem paths produced by the run (checkpoints,
            metrics.json, logs). Paths are not validated for existence
            here — the caller writes them.
        duration_s: Wall-clock seconds the run took. ``>= 0``.
        error: Required when ``status == FAILED``; ``None`` otherwise.

    Invariants:
        - ``status`` is terminal.
        - ``duration_s >= 0``.
        - ``final_metrics`` is a tuple of :class:`TrainingMetric`.
        - ``artifacts`` is a tuple of :class:`Path`.
        - ``error`` is non-empty string iff ``status == FAILED``.
    """

    run_id: RunId
    status: TrainingStatus
    duration_s: float
    final_metrics: tuple[TrainingMetric, ...] = field(default_factory=tuple)
    artifacts: tuple[Path, ...] = field(default_factory=tuple)
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, RunId):
            raise TrainingConfigError("run_id must be a RunId")
        if not isinstance(self.status, TrainingStatus):
            raise TrainingConfigError("status must be a TrainingStatus")
        if not self.status.is_terminal:
            raise TrainingConfigError(
                f"RunResult.status must be terminal (got {self.status.value!r})"
            )
        if not isinstance(self.duration_s, (int, float)) or self.duration_s < 0:
            raise TrainingConfigError(
                f"duration_s must be a non-negative number (got {self.duration_s!r})"
            )
        if not isinstance(self.final_metrics, tuple) or not all(
            isinstance(m, TrainingMetric) for m in self.final_metrics
        ):
            raise TrainingConfigError("final_metrics must be a tuple of TrainingMetric")
        if not isinstance(self.artifacts, tuple) or not all(
            isinstance(p, Path) for p in self.artifacts
        ):
            raise TrainingConfigError("artifacts must be a tuple of pathlib.Path")
        if self.status == TrainingStatus.FAILED:
            if not isinstance(self.error, str) or not self.error.strip():
                raise TrainingConfigError(
                    "error must be a non-empty string when status is FAILED"
                )
        elif self.error is not None:
            raise TrainingConfigError(
                f"error must be None when status is {self.status.value!r}"
            )
        object.__setattr__(self, "duration_s", float(self.duration_s))
