"""Singleton StateLogger: centralised event-recording facade."""

from __future__ import annotations

import threading
from typing import Any

from src.shared.python.app_state._history_store import HistoryStore
from src.shared.python.logging_pkg.logging_config import get_logger

_logger = get_logger(__name__)

_SINGLETON_LOCK = threading.Lock()
_singleton: StateLogger | None = None


class StateLogger:
    """Centralised recorder for user actions, simulation runs, and exceptions.

    Call :func:`get_state_logger` to obtain the process-level singleton.

    Attributes:
        store: The :class:`HistoryStore` backing this logger.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self.store = HistoryStore(maxlen=maxlen)

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Record a named event with optional context payload.

        Args:
            event_type: Non-empty string identifying the event category
                (e.g. ``"simulation_run"``, ``"param_change"``).
            payload: Optional key-value context for the event.

        Raises:
            ValueError: If *event_type* is empty.

        Postcondition:
            ``len(self.store)`` increases by one (capped at *maxlen*).
        """
        if not event_type:
            raise ValueError("event_type must be non-empty")
        self.store.append_event(event_type, payload or {})
        _logger.debug("AppState event recorded: %s", event_type)

    def log_exception(self, exc: Exception, context: str = "") -> None:
        """Convenience wrapper that records an exception as an event.

        Args:
            exc: The exception to record.
            context: Human-readable description of where the error occurred.
        """
        self.log_event(
            "exception",
            {"type": type(exc).__name__, "message": str(exc), "context": context},
        )

    def log_fallback(self, component: str, reason: str) -> None:
        """Record a fallback/degraded-mode activation.

        Args:
            component: The component that fell back (e.g. ``"MuJoCoEngine"``).
            reason: Why the fallback was triggered.
        """
        self.log_event("fallback", {"component": component, "reason": reason})


def get_state_logger() -> StateLogger:
    """Return the process-level :class:`StateLogger` singleton.

    Thread-safe; creates the singleton on first call.

    Returns:
        The shared :class:`StateLogger` instance.
    """
    global _singleton
    if _singleton is None:
        with _SINGLETON_LOCK:
            if _singleton is None:
                _singleton = StateLogger()
    return _singleton
