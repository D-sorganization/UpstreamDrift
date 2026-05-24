"""Opaque identifier types for the training controller.

Using nominal types (:class:`JobId`, :class:`RunId`) instead of bare
``str`` prevents accidental cross-wiring at call sites — passing a
``run_id`` where a ``job_id`` is required is a type error, not a silent
runtime bug. Both wrap a single canonical string value.

A "job" is the user-visible scheduled unit (one entry in the dashboard).
A "run" is a single execution attempt of a job; one job may produce
multiple runs (retries, restarts) over its lifetime.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from .errors import TrainingConfigError

__all__ = [
    "MAX_ID_LENGTH",
    "JobId",
    "RunId",
    "new_job_id",
    "new_run_id",
]


MAX_ID_LENGTH = 64
"""Maximum length for any identifier (matches typical filesystem cap)."""

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_id(value: str, *, label: str) -> None:
    """Shared invariant check for identifier strings.

    Args:
        value: Candidate identifier.
        label: Human-readable name used in the error message.

    Raises:
        TrainingConfigError: If ``value`` is empty, too long, or contains
            characters outside ``[A-Za-z0-9_-]``.
    """

    if not isinstance(value, str):
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value:
        raise TrainingConfigError(f"{label} must be non-empty")
    if len(value) > MAX_ID_LENGTH:
        raise TrainingConfigError(
            f"{label} must be <= {MAX_ID_LENGTH} characters (got {len(value)})"
        )
    if not _ID_PATTERN.fullmatch(value):
        raise TrainingConfigError(f"{label} must match [A-Za-z0-9_-]+ (got {value!r})")


@dataclass(frozen=True, slots=True, order=True)
class JobId:
    """Opaque, immutable identifier for a scheduled training job.

    Invariants:
        ``value`` is non-empty, <= :data:`MAX_ID_LENGTH` characters, and
        matches ``[A-Za-z0-9_-]+``.
    """

    value: str

    def __post_init__(self) -> None:
        _validate_id(self.value, label="JobId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class RunId:
    """Opaque, immutable identifier for a single execution of a job.

    Invariants: same as :class:`JobId`.
    """

    value: str

    def __post_init__(self) -> None:
        _validate_id(self.value, label="RunId")

    def __str__(self) -> str:
        return self.value


def new_job_id() -> JobId:
    """Mint a fresh :class:`JobId` backed by a UUID4 hex string."""

    return JobId(uuid.uuid4().hex)


def new_run_id() -> RunId:
    """Mint a fresh :class:`RunId` backed by a UUID4 hex string."""

    return RunId(uuid.uuid4().hex)
