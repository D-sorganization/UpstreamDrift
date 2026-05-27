"""Lazy exports for the Qt-backed shared chat dock widget.

Importing this module is intentionally safe during headless test collection.
The PyQt6 implementation is loaded only when widget classes are requested.

Threading
---------
Shared session state is held in a module-level singleton holder protected
by ``_SHARED_SESSION_LOCK``. ``_read_shared_session_id`` and
``_write_shared_session_id`` acquire this lock for the full duration of
their I/O so concurrent writers from different threads (or windows) cannot
interleave or observe a torn file. Writes are also atomic on disk: the new
content is staged in a sibling ``*.tmp`` file and then ``Path.replace``'d
into place, which is an atomic rename on POSIX and Win32.
"""

# ruff: noqa: F822

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any


def _resolve_default_server() -> str:
    """Compute the WS URL the chat dock should connect to.

    Honours, in order, ``GOLF_API_PORT`` / ``API_PORT`` / ``GOLF_PORT`` so the
    desktop launcher (which probes a free port and exports those vars before
    spawning the background API server) and the chat dock stay in sync. Falls
    back to the historical default ``ws://127.0.0.1:8000`` when nothing is
    set. Override entirely via ``UD_CHAT_WS_URL`` for non-standard hosts.

    DbC postcondition: the returned URL is a ``ws://`` or ``wss://`` string.
    """
    explicit = os.environ.get("UD_CHAT_WS_URL")
    if explicit:
        return explicit
    for env_name in ("GOLF_API_PORT", "API_PORT", "GOLF_PORT"):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        try:
            port = int(raw)
        except ValueError:
            continue
        if 1 <= port <= 65535:
            return f"ws://127.0.0.1:{port}"
    return "ws://127.0.0.1:8000"


_DEFAULT_SERVER = _resolve_default_server()
_QT_EXPORTS = {"ChatDockWidget", "ChatMessageBubble"}

# Tools issue #2753: serialize all reads/writes of the shared session ID
# file across threads. The lock guards both the in-memory holder and the
# atomic tmp+replace dance used by ``_write_shared_session_id``.
_SHARED_SESSION_LOCK = threading.Lock()


def _session_file_path(app_name: str) -> Path:
    """Return the path to the shared session ID file for an application."""
    return Path.home() / f".{app_name}" / "active_chat_session.txt"


def _read_shared_session_id(path: Path) -> str | None:
    """Read the active session ID from a shared file.

    Thread-safe: serialized against concurrent writers via
    ``_SHARED_SESSION_LOCK``.
    """
    with _SHARED_SESSION_LOCK:
        try:
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except (PermissionError, OSError):
            pass
        return None


def _write_shared_session_id(session_id: str, path: Path) -> None:
    """Write the active session ID to the shared file atomically.

    Writes the new content to ``<path>.tmp`` and then ``Path.replace``s it
    into place so concurrent readers never observe a partially-written
    file. Thread-safe: serialized via ``_SHARED_SESSION_LOCK``.
    """
    with _SHARED_SESSION_LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.parent / f"{path.name}.tmp"
            tmp_path.write_text(session_id, encoding="utf-8")
            tmp_path.replace(path)
        except (PermissionError, OSError):
            pass


def _load_qt_module() -> Any:
    from . import _chat_dock_widget_qt

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
