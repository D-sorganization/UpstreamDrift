# ruff: noqa: E501
"""Conversation-management helpers (Tools issue #2872).

Free helpers for ``/use-session`` resolution and breadcrumb mutation.
The chat dock retains thin public methods that delegate to these so the
parent module fits the repo's 1500-line budget.
"""

from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


def resolve_use_session_target(dock: Any, target: str) -> str | None:
    """Resolve a ``/use-session`` argument to a session id.

    Accepts either an exact session id or a case-insensitive title match.
    Returns ``None`` when no session matches.
    """
    if not target:
        return None
    manager = dock._session_manager
    if manager is None:
        return None
    sessions = list(manager.list_sessions())
    for info in sessions:
        if info.get("id") == target:
            return target
    needle = target.casefold()
    for info in sessions:
        title = str(info.get("title", "")).casefold()
        if title == needle:
            return cast(str | None, info.get("id"))
    return None


def add_context_session(dock: Any, session_id: str) -> None:
    """Append ``session_id`` to the breadcrumb context list."""
    if not session_id:
        raise ValueError("session_id must be provided")
    loaded = dock._loaded_context_sessions
    if session_id in loaded:
        return
    loaded.append(session_id)
    dock._refresh_breadcrumb()


def remove_context_session(dock: Any, session_id: str) -> None:
    """Remove ``session_id`` from the breadcrumb context list."""
    loaded = dock._loaded_context_sessions
    if session_id in loaded:
        loaded.remove(session_id)
        dock._refresh_breadcrumb()


def breadcrumb_labels(dock: Any) -> list[str]:
    """Return the human-readable titles for the loaded context sessions."""
    manager = dock._session_manager
    if manager is None:
        return []
    info_by_id = {info.get("id"): info for info in manager.list_sessions()}
    labels: list[str] = []
    for sid in dock._loaded_context_sessions:
        info = info_by_id.get(sid)
        if info is None:
            labels.append(sid)
        else:
            labels.append(str(info.get("title") or sid))
    return labels


def refresh_breadcrumb(dock: Any) -> None:
    """Re-render the breadcrumb strip after a context-list mutation."""
    try:
        widget = dock._breadcrumb_widget
    except (AttributeError, RuntimeError):
        widget = None
    if widget is None:
        return
    try:
        widget.set_labels(breadcrumb_labels(dock))
    except Exception:  # noqa: BLE001 - host UI failures must not break logic
        logger.exception("breadcrumb refresh failed")
