"""Transcript formatting utilities for the peer-review subsystem (Tools #2738).

Provides :func:`format_transcript` which extracts a conversation history into
a ``<transcript>`` XML-like block suitable for injection into a reviewer's
system prompt.

Design-by-Contract:
    - ``messages`` must be a ``list`` (raises ``TypeError`` otherwise).
    - Each element must expose ``role`` and ``content`` either as dict keys
      or object attributes.
"""

from __future__ import annotations

from typing import Any


def format_transcript(messages: list[Any]) -> str:
    """Format a list of chat messages into a ``<transcript>`` block.

    Accepts two message shapes:
    - ``dict`` with ``"role"`` and ``"content"`` keys.
    - Any object with ``.role`` and ``.content`` attributes.

    Args:
        messages: List of message dicts or message-like objects.

    Returns:
        A string ``<transcript>\\n...\\n</transcript>`` containing one line
        per message in ``[ROLE]: content`` format.

    Raises:
        TypeError: If ``messages`` is not a ``list``.
        ValueError: If ``messages`` is ``None``.
    """
    if messages is None:
        raise ValueError("messages must be a list, got None")
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__!r}")

    lines: list[str] = []
    for msg in messages:
        role, content = _extract_role_content(msg)
        lines.append(f"[{role.upper()}]: {content}")

    body = "\n".join(lines)
    if body:
        return f"<transcript>\n{body}\n</transcript>"
    return "<transcript>\n</transcript>"


def _extract_role_content(msg: Any) -> tuple[str, str]:
    """Extract (role, content) from a dict or an object with attributes.

    Args:
        msg: A message dict or object.

    Returns:
        ``(role, content)`` as strings.

    Raises:
        ValueError: If neither interface is present.
    """
    if isinstance(msg, dict):
        return str(msg.get("role", "unknown")), str(msg.get("content", ""))
    role = getattr(msg, "role", None)
    content = getattr(msg, "content", None)
    if role is None and content is None:
        raise ValueError(
            f"Message object must have 'role'/'content' keys or attributes, "
            f"got {type(msg).__name__!r}"
        )
    return str(role or "unknown"), str(content or "")


__all__ = ["format_transcript"]
