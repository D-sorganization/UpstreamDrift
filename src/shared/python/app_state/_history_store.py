"""Thread-safe ring-buffer store for AppEvent objects."""

from __future__ import annotations

import json
import threading
from collections import deque
from typing import Any

from src.shared.python.app_state._events import AppEvent
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_MAXLEN = 1000


class HistoryStore:
    """Thread-safe, fixed-capacity store of :class:`AppEvent` objects.

    Uses a ``collections.deque`` with *maxlen* as the backing ring-buffer,
    meaning older events are silently dropped once capacity is reached.

    Attributes:
        maxlen: Maximum number of events retained.
    """

    def __init__(self, maxlen: int = _DEFAULT_MAXLEN) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self.maxlen: int = maxlen
        self._deque: deque[AppEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append a new event.

        Args:
            event_type: Non-empty string identifying the event category.
            payload: Arbitrary key-value context.

        Raises:
            ValueError: If *event_type* is empty.
        """
        if not event_type:
            raise ValueError("event_type must be non-empty")
        event = AppEvent(type=event_type, payload=payload)
        with self._lock:
            self._deque.append(event)

    def clear(self) -> None:
        """Remove all stored events."""
        with self._lock:
            self._deque.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def latest(self) -> AppEvent | None:
        """Return the most recent event, or ``None`` if the store is empty."""
        with self._lock:
            return self._deque[-1] if self._deque else None

    def snapshot(self, max_events: int | None = None) -> list[AppEvent]:
        """Return a point-in-time copy of stored events.

        Args:
            max_events: If given, return only the *max_events* most recent.

        Returns:
            List of :class:`AppEvent`, oldest first.
        """
        with self._lock:
            items = list(self._deque)
        if max_events is not None and max_events < len(items):
            items = items[-max_events:]
        return items

    def as_json(self) -> str:
        """Serialise the store to a JSON string.

        Serialisation errors are caught internally; the result is always
        valid JSON (falls back to ``"[]"`` if necessary).

        Returns:
            JSON array string, each element following ``AppEvent.as_dict()``.
        """
        try:
            return json.dumps([e.as_dict() for e in self.snapshot()])
        except (TypeError, ValueError) as exc:
            logger.warning("HistoryStore.as_json serialisation failed: %s", exc)
            return "[]"

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)

    def __iter__(self):  # type: ignore[override]
        return iter(self.snapshot())
