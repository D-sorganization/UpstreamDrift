# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Progress reporting and state tracking for the trajectory optimiser."""

from __future__ import annotations

import threading
import time
from contextlib import AbstractContextManager

from .result import ProgressReport
from .tuning import STALL_THRESHOLD, STALL_WINDOW


class ProgressTracker:
    """Mutable progress state used by the primary optimisation start.

    Preconditions:
        progress_cb is callable or None.

    This class owns the iteration counter, cost history, and best-cost
    tracking that are only updated on the single-threaded (primary) start.
    The parallel worker threads use ``_compute_cost`` directly without
    touching this state.
    """

    def __init__(
        self,
        progress_cb=None,
    ) -> None:
        self.progress_cb = progress_cb
        self._iter: int = 0
        self._cost_history: list[float] = []
        self._best_cost: float = float("inf")
        self._start_time: float = 0.0
        self._progress_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API (stable) — prefer these over touching private attributes.
    # ------------------------------------------------------------------

    @property
    def cost_history(self) -> list[float]:
        """Full cost history recorded so far (list, most recent last)."""
        return self._cost_history

    @cost_history.setter
    def cost_history(self, value: list[float]) -> None:
        """Replace the cost history (used by tests and diagnostics)."""
        self._cost_history = value

    @property
    def iteration_count(self) -> int:
        """Number of cost evaluations recorded."""
        return self._iter

    def elapsed(self) -> float:
        """Seconds since the last :meth:`reset` call."""
        return time.monotonic() - self._start_time

    def lock(self) -> AbstractContextManager[bool]:
        """Return the internal progress lock as a context manager.

        Use this to synchronise external reads/writes with
        :meth:`record_parallel`.
        """
        return self._progress_lock

    def reset(self) -> None:
        """Reset all mutable counters before a new optimisation run."""
        self._iter = 0
        self._cost_history = []
        self._best_cost = float("inf")
        self._start_time = time.monotonic()

    def record(self, cost: float) -> None:
        """Record a new cost evaluation (single-thread path)."""
        self._iter += 1
        self._cost_history.append(cost)
        self._best_cost = min(self._best_cost, cost)

        if self.progress_cb and self._iter % 20 == 0:
            self.emit(cost)

    def record_parallel(self, cost: float, total_evals: int) -> None:
        """Update state from a completed parallel worker result (thread-safe)."""
        with self._progress_lock:
            self._iter = total_evals
            self._cost_history.append(cost)
            self._best_cost = min(self._best_cost, cost)
            if self.progress_cb:
                self.emit(cost)

    def emit(self, current_cost: float) -> None:
        """Build and dispatch a ProgressReport to the callback."""
        elapsed = time.monotonic() - self._start_time
        if len(self._cost_history) >= 40:
            prev = self._cost_history[-40]
            improvement = (prev - current_cost) / abs(prev) * 100 if prev != 0 else 0.0
        else:
            improvement = 0.0

        is_stalled, stall_reason = detect_stall(self._cost_history)
        recent_history = self._cost_history[-200:]

        report = ProgressReport(
            iteration=self._iter,
            cost=current_cost,
            best_cost=self._best_cost,
            improvement_pct=improvement,
            elapsed_s=elapsed,
            cost_history=recent_history,
            is_stalled=is_stalled,
            stall_reason=stall_reason,
        )
        if self.progress_cb:
            self.progress_cb(report)


def detect_stall(history: list[float]) -> tuple[bool, str]:
    """Detect whether optimisation has stalled based on cost history.

    Returns (is_stalled, reason_string).  ``reason_string`` is empty when
    the optimisation is not stalled.
    """
    if len(history) < STALL_WINDOW:
        return False, ""
    recent = history[-STALL_WINDOW:]
    old_cost = recent[0]
    new_cost = recent[-1]
    if old_cost == 0:
        return False, ""
    rel_improvement = abs(old_cost - new_cost) / abs(old_cost)
    if rel_improvement < STALL_THRESHOLD:
        return True, (
            f"Cost changed < {STALL_THRESHOLD * 100:.2f}% "
            f"over last {STALL_WINDOW} evals "
            f"({old_cost:.1f} -> {new_cost:.1f})"
        )
    return False, ""
