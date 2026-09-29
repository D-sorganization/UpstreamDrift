"""Shared formatting helpers for the chat history UI.

These helpers are kept module-private and DRY: a single
:func:`format_conversation_preview` is used by both the recent and the
archived list rendering paths, and a single :func:`group_by_date` powers
the date headers in both panels.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Any


def format_conversation_preview(conversation: dict[str, Any]) -> str:
    """Return a single-line preview string for a conversation dict.

    Used by both the recent and archived sections so they stay visually
    consistent (DRY).
    """
    title = conversation.get("title") or "Untitled"
    snippet = conversation.get("snippet") or ""
    ts = _format_timestamp(conversation.get("timestamp"))
    if snippet:
        return f"{title}  ·  {ts}\n  {snippet}"
    return f"{title}  ·  {ts}"


def group_by_date(
    conversations: list[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    """Return conversations grouped by their ``YYYY-MM-DD`` date.

    The grouping preserves the input order so callers can sort the list
    before calling and have the result honour that ordering.
    """
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for conv in conversations:
        key = _date_key(conv.get("timestamp"))
        groups.setdefault(key, []).append(conv)
    return groups


def _date_key(timestamp: Any) -> str:
    """Return a ``YYYY-MM-DD`` date string for grouping."""
    if not timestamp:
        return "Unknown"
    if isinstance(timestamp, datetime):
        return timestamp.date().isoformat()
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp).date().isoformat()
        except ValueError:
            return "Unknown"
    return "Unknown"


def _format_timestamp(timestamp: Any) -> str:
    """Return a human-readable timestamp string."""
    if not timestamp:
        return ""
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M")
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return timestamp
    return str(timestamp)
