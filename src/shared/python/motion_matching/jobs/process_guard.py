"""Owned process-tree termination for cancelled matching jobs.

Wraps the same terminate→kill escalation used by
:func:`core.process_safety.managed_popen` without introducing a scheduler.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Protocol

logger = logging.getLogger(__name__)

__all__ = ["ProcessGuard", "TerminableProcess"]


class TerminableProcess(Protocol):
    """Minimal process surface used by the guard."""

    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class ProcessGuard:
    """Tracks owned worker processes and terminates them on cancel."""

    def __init__(self) -> None:
        self._owned: list[TerminableProcess] = []

    def register(self, proc: TerminableProcess) -> None:
        if proc is None:
            raise TypeError("proc must not be None")
        self._owned.append(proc)

    def terminate_all(self, *, reason: str, kill_timeout: float = 5.0) -> None:
        """Terminate every owned live process; escalate to kill if needed."""
        logger.info("terminating %d owned process(es): %s", len(self._owned), reason)
        still: list[TerminableProcess] = []
        for proc in list(self._owned):
            if proc.poll() is not None:
                continue
            try:
                proc.terminate()
            except OSError:
                logger.exception(
                    "terminate failed for pid=%s", getattr(proc, "pid", "?")
                )
                continue
            still.append(proc)
        for proc in still:
            try:
                proc.wait(timeout=kill_timeout)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                    proc.wait(timeout=kill_timeout)
                except OSError:
                    logger.exception(
                        "kill failed for pid=%s", getattr(proc, "pid", "?")
                    )
        self._owned.clear()

    @property
    def owned_count(self) -> int:
        return len(self._owned)

    def snapshot(self) -> list[dict[str, Any]]:
        return [{"pid": p.pid, "alive": p.poll() is None} for p in self._owned]
