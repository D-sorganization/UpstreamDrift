"""Chat export and copy utilities for the AI assistant.

Provides:
- ``export_thread_to_markdown`` — convert a list of Messages to a structured
  markdown string (``**User:** ...\\n\\n**Agent:** ...``).
- ``copy_message_to_clipboard`` — copy a single text string to the system clipboard.
- ``copy_thread_to_clipboard`` — aggregate a full conversation and copy it.
- ``save_thread_as_markdown`` — prompt the user with QFileDialog and write to .md file.

Design by contract
------------------
Every public function validates its inputs and raises ``TypeError`` or
``ValueError`` on invalid arguments.  No ``print()`` calls; use ``logging``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QApplication, QFileDialog

logger = logging.getLogger(__name__)

# Role display labels used in the exported markdown
_ROLE_LABELS: dict[str, str] = {
    "user": "User",
    "assistant": "Agent",
}

# Roles that are silently skipped when exporting
_SKIP_ROLES: frozenset[str] = frozenset({"system", "tool"})


def export_thread_to_markdown(messages: list[Any]) -> str:
    """Convert a conversation thread to structured markdown.

    The output format is::

        **User:** <message text>

        **Agent:** <response text>

    System and tool messages are omitted.

    Args:
        messages: List of :class:`~src.shared.python.ai.types.Message` objects.

    Returns:
        Markdown string representing the conversation.  Returns an empty
        string when *messages* is empty.

    Raises:
        TypeError: If *messages* is not a list.
        ValueError: If *messages* is ``None``.
    """
    if messages is None:
        raise ValueError("messages must not be None")
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__!r}")

    parts: list[str] = []
    for msg in messages:
        role: str = getattr(msg, "role", "")
        if role in _SKIP_ROLES:
            continue
        content: str = getattr(msg, "content", "")
        label = _ROLE_LABELS.get(role, role.title())
        parts.append(f"**{label}:** {content}")

    return "\n\n".join(parts)


def copy_message_to_clipboard(text: str) -> None:
    """Copy a single text string to the system clipboard.

    Args:
        text: The text to copy.  Must not be ``None``.

    Raises:
        ValueError: If *text* is ``None``.
        TypeError: If *text* is not a string.
    """
    if text is None:
        raise ValueError("text must not be None")
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__!r}")

    clipboard = QApplication.clipboard()
    if clipboard is not None:
        clipboard.setText(text)
    logger.debug("Copied %d characters to clipboard", len(text))


def copy_thread_to_clipboard(messages: list[Any]) -> None:
    """Aggregate the full conversation and copy it to the clipboard.

    Args:
        messages: List of :class:`~src.shared.python.ai.types.Message` objects.

    Raises:
        TypeError: If *messages* is not a list.
        ValueError: If *messages* is ``None``.
    """
    markdown = export_thread_to_markdown(messages)
    clipboard = QApplication.clipboard()
    if clipboard is not None:
        clipboard.setText(markdown)
    logger.debug("Copied thread (%d messages) to clipboard", len(messages))


def save_thread_as_markdown(
    messages: list[Any],
    parent: Any = None,
) -> Path | None:
    """Prompt the user for a file path and write the conversation as markdown.

    Opens a :class:`QFileDialog` save dialog.  If the user cancels the dialog,
    this function returns ``None`` without raising.

    Args:
        messages: List of :class:`~src.shared.python.ai.types.Message` objects.
        parent: Optional parent widget for the dialog.

    Returns:
        The :class:`pathlib.Path` written, or ``None`` if the dialog was
        cancelled.

    Raises:
        TypeError: If *messages* is not a list.
        ValueError: If *messages* is ``None``.
        OSError: If the file cannot be written.
    """
    markdown = export_thread_to_markdown(messages)

    path_str, _ = QFileDialog.getSaveFileName(
        parent,
        "Save Thread as Markdown",
        "chat_export.md",
        "Markdown (*.md);;All Files (*)",
    )
    if not path_str:
        logger.debug("Save-as-markdown dialog cancelled by user")
        return None

    dest = Path(path_str)
    dest.write_text(markdown, encoding="utf-8")
    logger.info("Thread exported to %s", dest)
    return dest
