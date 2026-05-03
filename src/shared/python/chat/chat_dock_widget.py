"""Lazy exports for the Qt-backed shared chat dock widget.

Importing this module is intentionally safe during headless test collection.
The PyQt6 implementation is loaded only when widget classes are requested.
"""

# ruff: noqa: F822

from __future__ import annotations

from pathlib import Path
from typing import Any

_DEFAULT_SERVER = "ws://127.0.0.1:8000"
_QT_EXPORTS = {"ChatDockWidget", "ChatMessageBubble"}


def _session_file_path(app_name: str) -> Path:
    """Return the path to the shared session ID file for an application."""
    return Path.home() / f".{app_name}" / "active_chat_session.txt"


def _read_shared_session_id(path: Path) -> str | None:
    """Read the active session ID from a shared file."""
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except (PermissionError, OSError):
        pass
    return None


def _write_shared_session_id(session_id: str, path: Path) -> None:
    """Write the active session ID to the shared file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(session_id, encoding="utf-8")
    except (PermissionError, OSError):
        pass


def _load_qt_module() -> Any:
    from src.shared.python.chat import _chat_dock_widget_qt

    return _chat_dock_widget_qt


def __getattr__(name: str) -> Any:
    if name in _QT_EXPORTS:
        return getattr(_load_qt_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatDockWidget",
    "ChatMessageBubble",
    "_DEFAULT_SERVER",
    "_session_file_path",
    "_read_shared_session_id",
    "_write_shared_session_id",
]
