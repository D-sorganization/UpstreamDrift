"""Real-time IPC layer with file and WebSocket pub-sub backends.

Provides a unified facade for in-process and cross-process publish/subscribe
messaging, used by the embedded launcher to bridge the GUI tile and the
underlying engine processes.

Subtask 4 of EPIC #4993 (issue #4997).

Public surface:
    - ``publish(channel, payload, *, transport="auto")``
    - ``subscribe(channel, callback, *, transport="auto") -> Subscription``
    - ``Subscription`` (frozen dataclass with ``unsubscribe()``)

Transports:
    - ``"file"`` — atomic JSON write under ``~/.upstream_drift/realtime/``,
      using ``QFileSystemWatcher`` / ``watchdog`` / 100 ms polling fallback.
      Latency budget: < 200 ms.
    - ``"ws"`` — WebSocket pub-sub via the FastAPI server in :mod:`src.api`.
      Latency budget: < 50 ms.
    - ``"auto"`` — chosen via :func:`channels.get_channel_transport`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .channels import get_channel_transport
from .file_pubsub import FilePubSub
from .protocol import Subscription, validate_channel
from .ws_pubsub import WSPubSub

__all__ = [
    "FilePubSub",
    "Subscription",
    "WSPubSub",
    "publish",
    "subscribe",
    "validate_channel",
]


_Transport = Literal["auto", "file", "ws"]


# Module-level singletons so subscribers and publishers share state across
# calls within the same process.
_file_backend: FilePubSub | None = None
_ws_backend: WSPubSub | None = None


def _get_file_backend() -> FilePubSub:
    global _file_backend
    if _file_backend is None:
        _file_backend = FilePubSub()
    return _file_backend


def _get_ws_backend() -> WSPubSub:
    global _ws_backend
    if _ws_backend is None:
        _ws_backend = WSPubSub()
    return _ws_backend


def _resolve_transport(channel: str, transport: _Transport) -> Literal["file", "ws"]:
    if transport == "auto":
        return get_channel_transport(channel)
    if transport in ("file", "ws"):
        return transport
    raise ValueError(
        f"Invalid transport {transport!r}; expected 'auto', 'file', or 'ws'"
    )


def publish(
    channel: str,
    payload: dict,
    *,
    transport: _Transport = "auto",
) -> None:
    """Publish ``payload`` to ``channel`` via the chosen transport."""
    validate_channel(channel)
    chosen = _resolve_transport(channel, transport)
    if chosen == "file":
        _get_file_backend().publish(channel, payload)
    else:
        _get_ws_backend().publish(channel, payload)


def subscribe(
    channel: str,
    callback: Callable[[dict], None],
    *,
    transport: _Transport = "auto",
) -> Subscription:
    """Subscribe to ``channel`` with ``callback``; returns a Subscription."""
    validate_channel(channel)
    chosen = _resolve_transport(channel, transport)
    if chosen == "file":
        return _get_file_backend().subscribe(channel, callback)
    return _get_ws_backend().subscribe(channel, callback)
