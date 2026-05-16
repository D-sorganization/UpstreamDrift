"""agent_context() — serialises recent AppState history for AI agents."""

from __future__ import annotations

from typing import Any

from src.shared.python.app_state._history_store import HistoryStore

_DEFAULT_MAX_EVENTS: int = 50


def agent_context(
    store: HistoryStore,
    max_events: int = _DEFAULT_MAX_EVENTS,
) -> list[dict[str, Any]]:
    """Return a JSON-safe list of recent events for consumption by AI agents.

    Args:
        store: The :class:`HistoryStore` to sample.
        max_events: Maximum number of recent events to include.

    Returns:
        List of dicts, each following ``AppEvent.as_dict()`` format.
    """
    return [e.as_dict() for e in store.snapshot(max_events=max_events)]
