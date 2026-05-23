"""Domain exceptions for the training controller.

All training-controller errors derive from :class:`TrainingError` so callers
can broadly catch the package's failure modes without resorting to bare
``except Exception`` (forbidden by the BLE001 ratchet — see CLAUDE.md).
"""

from __future__ import annotations

__all__ = [
    "CompatibilityError",
    "DuplicateJobError",
    "InvalidStatusTransitionError",
    "JobNotFoundError",
    "TrainingConfigError",
    "TrainingError",
]


class TrainingError(Exception):
    """Base class for training-controller domain errors."""


class TrainingConfigError(TrainingError, ValueError):
    """Raised when a :class:`TrainingConfig` precondition is violated.

    Subclasses :class:`ValueError` so callers using stdlib idioms still
    catch it.
    """


class InvalidStatusTransitionError(TrainingError):
    """Raised when a training-status transition is not permitted.

    Attributes:
        source: The status the job is currently in.
        destination: The status the caller attempted to transition to.
    """

    def __init__(self, source: object, destination: object) -> None:
        self.source = source
        self.destination = destination
        super().__init__(
            f"Cannot transition training status from {source!r} to {destination!r}"
        )


class CompatibilityError(TrainingError):
    """Raised when a training config is incompatible with the target engine."""


class JobNotFoundError(TrainingError, LookupError):
    """Raised when a job-id lookup fails in a registry."""


class DuplicateJobError(TrainingError):
    """Raised when a job-id collision is detected on registration."""
