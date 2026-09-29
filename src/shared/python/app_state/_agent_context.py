"""agent_context: serialise app state for Sidekick/AI chat agents."""

from __future__ import annotations

from typing import Any

from src.shared.python.app_state._history_store import HistoryStore

_DEFAULT_MAX_EVENTS = 50


def agent_context(
    store: HistoryStore,
    max_events: int = _DEFAULT_MAX_EVENTS,
    last_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable context dict for AI agents (e.g. Sidekick).

    Args:
        store: The :class:`HistoryStore` to sample events from.
        max_events: Maximum number of recent events to include.
        last_diagnostics: Optional list of diagnostic result dicts from
            :meth:`DiagnosticEngine.run_checks` (serialised via
            ``dataclasses.asdict``).

    Returns:
        Dict with keys:

        - ``"events"`` — list of recent event dicts.
        - ``"last_diagnostics"`` — list of diagnostic result dicts (may be empty).
        - ``"summary"`` — short human-readable summary string.

    Postcondition:
        ``len(result["events"]) <= max_events``
    """
    if max_events <= 0:
        raise ValueError(f"max_events must be positive, got {max_events}")

    recent = store.snapshot(max_events=max_events)
    events_dicts = [e.as_dict() for e in recent]

    diag_list = last_diagnostics if last_diagnostics is not None else []

    total = len(store)
    pass_count = sum(1 for d in diag_list if d.get("status") == "PASS")
    fail_count = sum(1 for d in diag_list if d.get("status") == "FAIL")

    summary = (
        f"Total events recorded: {total}. "
        f"Last diagnostic run: {pass_count} PASS, {fail_count} FAIL."
    )

    return {
        "events": events_dicts,
        "last_diagnostics": diag_list,
        "summary": summary,
    }
