"""Training-job lifecycle status and transition rules.

The state machine is intentionally narrow: a job moves through
``PENDING → QUEUED → RUNNING`` to one of three terminal states
(``COMPLETED``, ``FAILED``, ``CANCELLED``), with an optional
``PAUSED`` detour from ``RUNNING``. Any transition outside the
declared edges raises :class:`InvalidStatusTransitionError`.

This module owns the *rules*; concrete jobs (see :mod:`job`) consult
them but do not embed the table.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping

from .errors import InvalidStatusTransitionError

__all__ = [
    "TERMINAL_STATUSES",
    "TrainingStatus",
    "can_transition",
    "validate_transition",
]


class TrainingStatus(Enum):
    """Lifecycle status of a training job.

    Values are stable wire-format strings so the enum can serialize to
    JSON without a translation table.
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """``True`` when no further transitions are permitted."""

        return self in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        """``True`` when the job is in-flight (running or paused)."""

        return self in {TrainingStatus.RUNNING, TrainingStatus.PAUSED}


TERMINAL_STATUSES: frozenset[TrainingStatus] = frozenset(
    {
        TrainingStatus.COMPLETED,
        TrainingStatus.FAILED,
        TrainingStatus.CANCELLED,
    }
)


_TRANSITIONS: Mapping[TrainingStatus, frozenset[TrainingStatus]] = MappingProxyType(
    {
        TrainingStatus.PENDING: frozenset(
            {TrainingStatus.QUEUED, TrainingStatus.CANCELLED, TrainingStatus.FAILED}
        ),
        TrainingStatus.QUEUED: frozenset(
            {TrainingStatus.RUNNING, TrainingStatus.CANCELLED, TrainingStatus.FAILED}
        ),
        TrainingStatus.RUNNING: frozenset(
            {
                TrainingStatus.PAUSED,
                TrainingStatus.COMPLETED,
                TrainingStatus.FAILED,
                TrainingStatus.CANCELLED,
            }
        ),
        TrainingStatus.PAUSED: frozenset(
            {
                TrainingStatus.RUNNING,
                TrainingStatus.CANCELLED,
                TrainingStatus.FAILED,
            }
        ),
        TrainingStatus.COMPLETED: frozenset(),
        TrainingStatus.FAILED: frozenset(),
        TrainingStatus.CANCELLED: frozenset(),
    }
)


def can_transition(source: TrainingStatus, destination: TrainingStatus) -> bool:
    """Pure predicate: is ``source → destination`` a permitted edge?

    Args:
        source: The current job status.
        destination: The status the caller wants to move to.

    Returns:
        ``True`` if the transition is allowed, ``False`` otherwise.
        A no-op transition (``source == destination``) is **not**
        allowed — callers should not re-apply the same status.
    """

    if not isinstance(source, TrainingStatus):
        raise TypeError(f"source must be TrainingStatus, got {type(source).__name__}")
    if not isinstance(destination, TrainingStatus):
        raise TypeError(
            f"destination must be TrainingStatus, got {type(destination).__name__}"
        )
    return destination in _TRANSITIONS[source]


def validate_transition(source: TrainingStatus, destination: TrainingStatus) -> None:
    """Raise :class:`InvalidStatusTransitionError` if the edge is illegal.

    Args:
        source: Current status.
        destination: Proposed new status.

    Raises:
        InvalidStatusTransitionError: When :func:`can_transition` is False.
    """

    if not can_transition(source, destination):
        raise InvalidStatusTransitionError(source, destination)
