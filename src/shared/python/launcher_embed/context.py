"""Shared in-process context for embedded launcher tools (#7210).

``LauncherContext`` is deliberately small and headless. It gives tools
mounted in one launcher process a shared event bus and value registry
without introducing cross-process IPC. Use the realtime package when a
feature must cross process boundaries.

Standard event names:

- ``engine.switched``
- ``model.loaded``
- ``simulation.started``
- ``simulation.stopped``
- ``tab.opened``
- ``tab.closed``
- ``value_changed:<key>`` for state-registry changes
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

EventPayload = dict[str, Any]
EventCallback = Callable[[EventPayload], None]
Unsubscribe = Callable[[], None]

__all__ = [
    "EventCallback",
    "EventPayload",
    "InMemoryLauncherContext",
    "LauncherContext",
    "Unsubscribe",
]


@runtime_checkable
class LauncherContext(Protocol):
    """Shared event/value surface injected into opt-in embedded tools."""

    def emit(self, event_type: str, payload: EventPayload) -> None:
        """Publish an event to subscribers of ``event_type``."""
        ...

    def subscribe(self, event_type: str, callback: EventCallback) -> Unsubscribe:
        """Subscribe to an event type and return an idempotent unsubscribe."""
        ...

    def get_value(self, key: str, default: Any = None) -> Any:
        """Return a shared value or ``default`` when absent."""
        ...

    def set_value(self, key: str, value: Any) -> Any:
        """Store a shared value and return the previous value, if any."""
        ...


class _Subscription:
    """Idempotent unsubscribe handle for one callback registration."""

    def __init__(
        self,
        subscribers: list[EventCallback],
        callback: EventCallback,
    ) -> None:
        self._subscribers = subscribers
        self._callback: EventCallback | None = callback

    def __call__(self) -> None:
        """Remove the callback from future dispatch; safe to call repeatedly."""
        if self._callback is None:
            return
        with contextlib.suppress(ValueError):
            self._subscribers.remove(self._callback)
        self._callback = None


class InMemoryLauncherContext:
    """In-process ``LauncherContext`` implementation.

    Dispatch uses a snapshot of the subscriber list. A callback may
    unsubscribe itself or emit/set values re-entrantly without mutating
    the current dispatch iteration.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._values: dict[str, Any] = dict(initial or {})
        self._subscribers: dict[str, list[EventCallback]] = {}

    def emit(self, event_type: str, payload: EventPayload) -> None:
        """Publish ``payload`` to subscribers of ``event_type``."""
        self._validate_name(event_type, "event_type")
        if not isinstance(payload, dict):
            raise TypeError(f"payload must be a dict, got {type(payload).__name__!r}")
        callbacks = list(self._subscribers.get(event_type, ()))
        for callback in callbacks:
            try:
                callback(dict(payload))
            except Exception:  # noqa: BLE001 - subscribers must not break host
                logger.exception(
                    "LauncherContext subscriber %r raised for %s",
                    callback,
                    event_type,
                )

    def subscribe(self, event_type: str, callback: EventCallback) -> Unsubscribe:
        """Register ``callback`` for ``event_type``."""
        self._validate_name(event_type, "event_type")
        if not callable(callback):
            raise TypeError(
                f"callback must be callable, got {type(callback).__name__!r}"
            )
        subscribers = self._subscribers.setdefault(event_type, [])
        subscribers.append(callback)
        return _Subscription(subscribers, callback)

    def get_value(self, key: str, default: Any = None) -> Any:
        """Return a value from the shared registry."""
        self._validate_name(key, "key")
        return self._values.get(key, default)

    def set_value(self, key: str, value: Any) -> Any:
        """Store ``value`` and emit ``value_changed:<key>``."""
        self._validate_name(key, "key")
        existed = key in self._values
        previous = self._values.get(key)
        self._values[key] = value
        self.emit(
            f"value_changed:{key}",
            {
                "key": key,
                "value": value,
                "previous": previous,
                "existed": existed,
            },
        )
        return previous

    def list(self) -> list[str]:
        """Return stored value keys in stable order.

        This intentionally mirrors the Sidekick workspace registry's
        small ``list/get/set`` surface so the launcher context can back
        agent-visible workspace variables without an adapter layer.
        """
        return sorted(self._values)

    def get(self, key: str, default: Any = None) -> Any:
        """Workspace-compatible alias for :meth:`get_value`."""
        return self.get_value(key, default)

    def set(self, key: str, value: Any) -> None:
        """Workspace-compatible alias for :meth:`set_value`."""
        self.set_value(key, value)

    @staticmethod
    def _validate_name(value: str, label: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
