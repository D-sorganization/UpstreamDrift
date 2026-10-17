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
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit

from .qt_diagnostics import ChatQtDiagnostic, diagnose_chat_qt_runtime

_LAUNCHER_TOKEN_ENV = "UD_LAUNCHER_CSRF_TOKEN"
_LAUNCHER_TOKEN_QUERY_KEY = "launcher_token"


def _resolve_default_server() -> str:
    """Compute the WS URL the chat dock should connect to.

    Honours, in order, ``GOLF_API_PORT`` / ``API_PORT`` / ``GOLF_PORT`` so a
    host app (the UpstreamDrift desktop launcher, the Gasification_Model
    process simulator, etc.) that probes a free port and exports those vars
    before spawning its background API server stays in lock-step with this
    client default. Falls back to the historical ``ws://127.0.0.1:8000``
    when nothing is set. Override entirely via ``UD_CHAT_WS_URL`` for
    non-standard hosts.

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


def _validated_websocket_server(server_url: str) -> SplitResult:
    """Return parsed WebSocket URL parts after enforcing the client contract."""
    if not isinstance(server_url, str):
        raise TypeError("server_url must be a string")
    parsed = urlsplit(server_url)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError("server_url must use ws:// or wss://")
    if not parsed.hostname:
        raise ValueError("server_url must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("server_url must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("server_url must contain only a scheme and authority")
    return parsed


def _native_websocket_origin(server_url: str) -> str:
    """Return the HTTP(S) Origin corresponding to a native WS endpoint.

    Preconditions:
        ``server_url`` is a credential-free ``ws://`` or ``wss://`` URL.

    Postcondition:
        The result contains only scheme and authority, never path, query,
        fragment, or launcher capability material.
    """
    parsed = _validated_websocket_server(server_url)
    origin_scheme = "https" if parsed.scheme == "wss" else "http"
    return urlunsplit((origin_scheme, parsed.netloc, "", "", ""))


def _is_loopback_websocket_server(parsed: SplitResult) -> bool:
    """Return whether validated server parts identify a loopback peer."""
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _resolve_launcher_token() -> str | None:
    """Return the ephemeral host capability without persisting or logging it."""
    token = os.environ.get(_LAUNCHER_TOKEN_ENV)
    return token if token else None


def _build_native_websocket_url(
    server_url: str,
    websocket_path: str,
    launcher_token: str | None,
) -> str:
    """Build the authenticated native chat URL.

    Preconditions:
        ``server_url`` satisfies :func:`_validated_websocket_server`.
        ``websocket_path`` is an absolute path.

    Postcondition:
        The launcher capability is URL encoded exactly once for verified
        loopback peers and is never forwarded to a remote endpoint.
    """
    server = _validated_websocket_server(server_url)
    if not isinstance(websocket_path, str):
        raise TypeError("websocket_path must be a string")
    if not websocket_path.startswith("/"):
        raise ValueError("websocket_path must be absolute")
    if launcher_token is not None and not isinstance(launcher_token, str):
        raise TypeError("launcher_token must be a string or None")

    parsed = urlsplit(f"{server_url.rstrip('/')}{websocket_path}")
    query = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if name != _LAUNCHER_TOKEN_QUERY_KEY
    ]
    if launcher_token and _is_loopback_websocket_server(server):
        query.append((_LAUNCHER_TOKEN_QUERY_KEY, launcher_token))
    return urlunsplit(parsed._replace(query=urlencode(query)))


def _native_websocket_connection(
    server_url: str,
    websocket_path: str,
) -> tuple[str, str]:
    """Return native socket origin and authenticated URL as one contract."""
    return (
        _native_websocket_origin(server_url),
        _build_native_websocket_url(
            server_url,
            websocket_path,
            _resolve_launcher_token(),
        ),
    )


_QT_EXPORTS = {
    "ChatDockWidget",
    "ChatMessageBubble",
    "ChatConnectionConfig",
    "ChatPresentationConfig",
    "ChatIntegrationHooks",
}

# Tools issue #2753: serialize all reads/writes of the shared session ID
# file across threads. The lock guards both the in-memory holder and the
# atomic tmp+replace dance used by ``_write_shared_session_id``.
_SHARED_SESSION_LOCK = threading.Lock()


class ChatQtUnavailableError(ImportError):
    """Raised when the optional PyQt6 chat dock runtime is unavailable."""

    def __init__(self, diagnostic: ChatQtDiagnostic) -> None:
        detail = f": {diagnostic.detail}" if diagnostic.detail else ""
        super().__init__(f"PyQt6 chat dock unavailable ({diagnostic.reason}){detail}")
        self.diagnostic = diagnostic


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
    diagnostic = diagnose_chat_qt_runtime()
    if not diagnostic.available:
        raise ChatQtUnavailableError(diagnostic)

    try:
        from . import _chat_dock_widget_qt
    except ImportError as exc:
        raise ChatQtUnavailableError(
            ChatQtDiagnostic(
                available=False,
                reason="import_failed",
                detail=str(exc),
            )
        ) from exc

    return _chat_dock_widget_qt


def chat_qt_runtime_diagnostic() -> dict[str, str | bool]:
    """Return an import-safe diagnostic for the optional Qt chat dock."""
    diagnostic = diagnose_chat_qt_runtime()
    return {
        "available": diagnostic.available,
        "reason": diagnostic.reason,
        "detail": diagnostic.detail,
    }


def __getattr__(name: str) -> Any:
    if name in _QT_EXPORTS:
        return getattr(_load_qt_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatDockWidget",
    "ChatMessageBubble",
    "ChatConnectionConfig",
    "ChatPresentationConfig",
    "ChatIntegrationHooks",
    "ChatQtUnavailableError",
    "chat_qt_runtime_diagnostic",
    "_session_file_path",
    "_read_shared_session_id",
    "_write_shared_session_id",
    "_resolve_default_server",
]
