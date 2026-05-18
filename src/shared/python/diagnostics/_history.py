"""DiagnosticsHistory — ring-buffer of recent DiagnosticsSnapshot instances.

The ring-buffer has a fixed capacity (default 100).  When full, the oldest
snapshot is evicted automatically.  All operations are O(1) amortised thanks
to ``collections.deque`` with ``maxlen``.

Design-by-Contract invariants
------------------------------
- ``capacity`` must be a positive integer.
- ``record(snapshot)`` precondition: ``snapshot`` must be a ``DiagnosticsSnapshot``.
- ``get_recent(n)`` postcondition: returns a ``list`` of at most ``min(n, len)`` items.
- ``len(history) <= capacity`` is always maintained.

Implements part of Epic #5698.
"""

from __future__ import annotations

import logging
from collections import deque

from src.shared.python.contracts import ensure, require
from src.shared.python.diagnostics._snapshot import DiagnosticsSnapshot

logger = logging.getLogger(__name__)

_DEFAULT_CAPACITY = 100


class DiagnosticsHistory:
    """Fixed-capacity ring-buffer of ``DiagnosticsSnapshot`` instances.

    Args:
        capacity: Maximum number of snapshots to retain.  Defaults to 100.
            Older snapshots are silently evicted when capacity is exceeded.

    Examples::

        history = DiagnosticsHistory(capacity=10)
        for snapshot in stream:
            history.record(snapshot)
        recent = history.get_recent(5)
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        require(
            isinstance(capacity, int) and capacity > 0,
            "capacity must be a positive integer",
            capacity,
        )
        self._capacity = capacity
        self._buffer: deque[DiagnosticsSnapshot] = deque(maxlen=capacity)
        logger.debug("diagnostics_history_created capacity=%d", capacity)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of snapshots retained."""
        return self._capacity

    def __len__(self) -> int:
        """Return the number of snapshots currently stored."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, snapshot: DiagnosticsSnapshot) -> None:
        """Append *snapshot* to the history, evicting the oldest if full.

        Preconditions:
            - ``snapshot`` must be a ``DiagnosticsSnapshot``.

        Postcondition:
            ``len(self) <= self.capacity``.

        Args:
            snapshot: The snapshot to record.
        """
        require(
            isinstance(snapshot, DiagnosticsSnapshot),
            "snapshot must be a DiagnosticsSnapshot",
            snapshot,
        )
        self._buffer.append(snapshot)
        ensure(
            len(self._buffer) <= self._capacity,
            "record postcondition: len must not exceed capacity",
        )
        logger.debug(
            "diagnostics_recorded ts=%s size=%d",
            snapshot.timestamp.isoformat(),
            len(self._buffer),
        )

    def clear(self) -> None:
        """Remove all snapshots from the history."""
        self._buffer.clear()
        logger.debug("diagnostics_history_cleared")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_recent(self, n: int) -> list[DiagnosticsSnapshot]:
        """Return up to the *n* most-recent snapshots, newest last.

        Preconditions:
            - ``n`` must be a non-negative integer.

        Postcondition:
            - Returns a ``list``.
            - ``len(result) <= min(n, len(self))``.

        Args:
            n: Maximum number of snapshots to return.

        Returns:
            List of at most *n* snapshots in chronological order (oldest first).
        """
        require(
            isinstance(n, int) and n >= 0,
            "n must be a non-negative integer",
            n,
        )
        items = list(self._buffer)
        result = items[-n:] if n > 0 else []
        ensure(
            isinstance(result, list),
            "get_recent postcondition: must return a list",
        )
        ensure(
            len(result) <= min(n, len(self._buffer)),
            "get_recent postcondition: len(result) must be <= min(n, len(history))",
        )
        return result

    def get_all(self) -> list[DiagnosticsSnapshot]:
        """Return all stored snapshots in chronological order (oldest first).

        Returns:
            List of all snapshots.
        """
        return list(self._buffer)
