"""Protocols and cooperative primitives for the training controller.

This module defines the *contract* surface that downstream layers
implement:

- :class:`CancelToken` — cooperative cancellation signal.
- :class:`ProgressSink` — what a runner emits metrics / status to.
- :class:`TrainingJobRunner` — what a framework adapter must satisfy.

A small concrete :class:`ThreadingCancelToken` ships alongside the
Protocol because every consumer needs one and re-implementing it would
violate DRY. Heavier implementations (scheduler, subprocess workers,
GUI sinks) live in PR2 / PR3.

This module deliberately imports no GUI / framework code so headless
unit tests can exercise the contract surface.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from .config import TrainingConfig, TrainingFramework
from .job import RunResult
from .metrics import TrainingMetric
from .status import TrainingStatus

__all__ = [
    "CancelToken",
    "ProgressSink",
    "ThreadingCancelToken",
    "TrainingJobRunner",
]


@runtime_checkable
class CancelToken(Protocol):
    """Cooperative cancellation signal.

    Runners poll :attr:`is_cancelled` between iterations and should exit
    cleanly when it becomes ``True``. Setting the token is one-way:
    once cancelled, it stays cancelled. Implementations must be safe to
    share across threads (and, where applicable, across the process
    boundary via shared memory / IPC).
    """

    @property
    def is_cancelled(self) -> bool:
        """``True`` once :meth:`request_cancel` has been called."""

    def request_cancel(self) -> None:
        """Signal cancellation. Idempotent — calling twice is a no-op."""


class ThreadingCancelToken:
    """Thread-safe :class:`CancelToken` implementation.

    Uses a :class:`threading.Event` so multiple threads (e.g. a worker
    loop polling and a UI handler signalling) can interact safely.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def request_cancel(self) -> None:
        self._event.set()


@runtime_checkable
class ProgressSink(Protocol):
    """Drain for training-loop observations.

    Runners hand metrics and status updates to a sink; the sink decides
    whether to write to disk, publish over WebSocket, push to a GUI
    widget, or all three. Sinks must be safe to call from the runner's
    thread; they are not required to be thread-safe across multiple
    runners (each runner gets its own sink).
    """

    def emit_metric(self, metric: TrainingMetric) -> None:
        """Record a single metric observation."""

    def emit_status(
        self, status: TrainingStatus, *, message: str | None = None
    ) -> None:
        """Announce a status change.

        Args:
            status: The new status.
            message: Optional human-readable context (e.g. failure
                reason). Ignored by sinks that only care about the enum.
        """


@runtime_checkable
class TrainingJobRunner(Protocol):
    """Framework-specific adapter that executes a :class:`TrainingConfig`.

    One implementation per framework. The scheduler picks a runner by
    matching :attr:`framework` against ``config.framework``; if no
    runner matches the job is failed with a clear message.

    Implementations should:

    - Treat ``config`` as immutable input.
    - Poll ``cancel`` between iterations and return promptly with a
      ``CANCELLED`` status when set.
    - Emit metrics to ``progress`` at a frequency the dashboard can
      keep up with (typically once per epoch / per N steps).
    - Never raise on cancellation — return a ``CANCELLED`` RunResult.
      Raise only for unrecoverable framework / environment failures.
    """

    framework: TrainingFramework

    def can_run(self, config: TrainingConfig) -> bool:
        """``True`` if this runner is willing to execute ``config``.

        Allows finer-grained matching than ``framework`` alone — e.g. a
        PyTorch runner may decline configs that demand a GPU when none
        is available.
        """

    def prepare(self, config: TrainingConfig) -> None:
        """One-shot setup before :meth:`run` (create dirs, validate deps).

        Called by the scheduler before the job moves to ``RUNNING``.
        May raise — failures here surface as ``FAILED`` runs with the
        exception message attached.
        """

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        """Execute the training loop. Returns a terminal RunResult."""
