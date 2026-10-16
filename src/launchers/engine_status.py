"""Engine status-label text and reset helpers (issue #12962, refs #12678).

The launcher status label reads ``* MuJoCo Running`` while an engine tab is
open. Closing the tab used to leave that text behind. This module owns the
text format (one place, DRY) and the "reset when that engine's tab closes"
rule so the launcher only wires a signal to it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

__all__ = [
    "engine_running_text",
    "is_engine_running_status",
    "reset_status_if_engine_closed",
]

_RUNNING_PREFIX_CHARS = "*> \t"


class _TextLabel(Protocol):
    def text(self) -> str: ...


def engine_running_text(name: str, suffix: str = "") -> str:
    """Return the status-label text for a running engine (``* Name Running``)."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not name.strip():
        raise ValueError("name must not be blank")
    if not isinstance(suffix, str):
        raise TypeError("suffix must be a string")
    tail = f" ({suffix})" if suffix else ""
    return f"* {name} Running{tail}"


def is_engine_running_status(text: str, name: str) -> bool:
    """Return True when ``text`` is any ``Running`` status for engine ``name``."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not name.strip():
        return False
    body = text.lstrip(_RUNNING_PREFIX_CHARS)
    if not body.startswith(name):
        return False
    rest = body[len(name) :]
    return rest == " Running" or rest.startswith(" Running ")


def reset_status_if_engine_closed(
    label: _TextLabel,
    closed_title: str,
    restore_idle: Callable[[], None],
) -> bool:
    """Call ``restore_idle`` if ``label`` still says ``closed_title`` is running.

    Returns True when the status was reset. A label showing a different
    engine's status is left untouched.
    """
    if label is None:
        raise ValueError("label must be provided")
    if not callable(restore_idle):
        raise TypeError("restore_idle must be callable")
    if not is_engine_running_status(label.text(), closed_title):
        return False
    restore_idle()
    return True
