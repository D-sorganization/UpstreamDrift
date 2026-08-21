"""Off-GUI-thread execution of the Sidekick API readiness probe (issue #8939).

``check_sidekick_api_readiness`` performs synchronous HTTP I/O.  Running it
directly from a ``QTimer`` callback blocks the GUI thread for up to the
socket timeout on every attempt (~90 attempts in the first minute when the
API is slow to start).  This worker runs the probe on a ``QThread`` and
marshals the result back to the GUI thread via a signal.

Contract (DbC): the probe callable is only ever invoked from ``run()``,
i.e. never on the thread that owns the launcher widgets.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from PyQt6.QtCore import QThread, pyqtSignal


class SidekickReadinessProbeThread(QThread):
    """Run one Sidekick API readiness probe away from the GUI thread."""

    readiness_ready = pyqtSignal(object)

    def __init__(
        self,
        *,
        probe: Callable[..., Any],
        expected_instance_id: str | None,
        parent: Any | None = None,
    ) -> None:
        """Create a single-shot probe worker.

        Args:
            probe: Callable accepting ``expected_instance_id`` keyword and
                returning a ``SidekickApiReadiness``-like result.
            expected_instance_id: Instance identity the probe must match,
                or None to accept any ready API.
            parent: Optional Qt parent.

        Raises:
            TypeError: If ``probe`` is not callable.
        """
        super().__init__(parent)
        if not callable(probe):
            raise TypeError("probe must be callable")
        self._probe = probe
        self._expected_instance_id = expected_instance_id

    def run(self) -> None:  # pragma: no cover - exercised via Qt thread tests
        """Execute the blocking probe on this worker thread."""
        result = self._probe(expected_instance_id=self._expected_instance_id)
        self.readiness_ready.emit(result)


__all__ = ["SidekickReadinessProbeThread"]
