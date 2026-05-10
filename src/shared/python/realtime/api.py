"""Public API for the realtime IPC facade.

This module is intentionally tiny: it owns the channel registry and a
small dispatch glue layer that delegates to a transport (file by
default; websocket as a follow-up). The Pose Studio cross-tool demo
(Subtask 6 of EPIC #4993) only needs the file transport, so that is
what is wired up here.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

from .transport_file import FileTransport, default_channel_path

logger = get_logger(__name__)

__all__ = [
    "CHANNEL_REGISTRY",
    "Subscription",
    "publish",
    "register_channel",
    "subscribe",
]


# Channel name -> small descriptor. Tools register here at import time so
# that documentation tooling can enumerate the realtime surface without
# having to import every tool.
CHANNEL_REGISTRY: dict[str, ChannelInfo] = {}


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    """Static description of a realtime channel.

    Attributes:
        name: Channel name (e.g. ``"pose/canonical"``).
        description: Short human-readable description.
        owner_tool_id: Tool id of the canonical publisher (``None`` if any).
    """

    name: str
    description: str
    owner_tool_id: str | None = None


def register_channel(
    name: str, description: str, owner_tool_id: str | None = None
) -> None:
    """Register a channel descriptor.

    Idempotent: re-registering the same name with the same description is
    a no-op. Re-registering with a different description raises
    :class:`ValueError`.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("channel name must be a non-empty string")
    existing = CHANNEL_REGISTRY.get(name)
    if existing is not None:
        if (
            existing.description != description
            or existing.owner_tool_id != owner_tool_id
        ):
            raise ValueError(
                f"channel {name!r} already registered with a different descriptor"
            )
        return
    CHANNEL_REGISTRY[name] = ChannelInfo(
        name=name, description=description, owner_tool_id=owner_tool_id
    )


# Built-in channel: registered here so consumers don't have to import
# Pose Studio just to subscribe.
register_channel(
    "pose/canonical",
    "Canonical pose payloads broadcast by Pose Studio (and compatible "
    "tools) for live cross-tool mirroring.",
    owner_tool_id="pose_studio",
)


@dataclass(slots=True)
class Subscription:
    """Handle returned by :func:`subscribe`.

    The :meth:`unsubscribe` method tears the underlying transport
    watcher down. It is idempotent.
    """

    channel: str
    _transport: FileTransport | None = None
    _token: int = -1
    _closed: bool = field(default=False)

    def unsubscribe(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._transport is not None and self._token >= 0:
            try:
                self._transport.unsubscribe(self._token)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Subscription.unsubscribe failed on channel %s", self.channel
                )


# Module-global file transport, lazily initialised. The transport is
# stateless across processes (the on-disk log is the source of truth)
# and cheap to construct, so a single instance per process is fine.
_TRANSPORT: FileTransport | None = None


def _get_transport() -> FileTransport:
    global _TRANSPORT
    if _TRANSPORT is None:
        _TRANSPORT = FileTransport(default_channel_path)
    return _TRANSPORT


def publish(channel: str, payload: Any) -> None:
    """Publish *payload* on *channel*.

    *payload* must be JSON-serialisable. Errors are logged and swallowed
    — callers should treat realtime as a hint layer, never a critical
    path. The default transport is the file transport; the websocket
    transport can be opted into via ``REALTIME_TRANSPORT=ws`` in the
    future (not implemented here).
    """
    if not isinstance(channel, str) or not channel.strip():
        logger.warning("realtime.publish: invalid channel %r", channel)
        return
    transport = os.environ.get("REALTIME_TRANSPORT", "file")
    if transport != "file":
        logger.debug(
            "realtime.publish: transport %r not wired in this build, "
            "falling back to file",
            transport,
        )
    try:
        _get_transport().publish(channel, payload)
    except Exception:
        logger.exception("realtime.publish failed on channel %s", channel)


def subscribe(channel: str, callback: Callable[[Any], None]) -> Subscription:
    """Register *callback* to fire for every payload on *channel*.

    The callback runs on a transport-owned daemon thread; consumers that
    need to touch Qt widgets should marshal back to the GUI thread (e.g.
    via ``QMetaObject.invokeMethod`` or a ``QtCore.pyqtSignal``).

    Returns a :class:`Subscription` whose
    :meth:`Subscription.unsubscribe` tears the watcher down.
    """
    if not isinstance(channel, str) or not channel.strip():
        raise ValueError("channel must be a non-empty string")
    if not callable(callback):
        raise TypeError("callback must be callable")
    try:
        transport = _get_transport()
        token = transport.subscribe(channel, callback)
        return Subscription(channel=channel, _transport=transport, _token=token)
    except Exception:
        logger.exception("realtime.subscribe failed on channel %s", channel)
        return Subscription(channel=channel, _transport=None, _token=-1, _closed=True)
