"""Public API for the realtime IPC facade.

This module is intentionally tiny: it owns the channel registry and a
small dispatch glue layer that delegates to a transport. Two transports
are supported (see :data:`SUPPORTED_TRANSPORTS`):

- ``"file"`` (default) — :class:`~.transport_file.FileTransport`. Works
  across processes started in either order and across crashes; the
  right choice for the Pose Studio cross-tool demo (Subtask 6 of
  EPIC #4993).
- ``"ws"`` — :class:`~.ws_pubsub.WSPubSub`. Lower latency, backed by the
  Rust ``upstream-realtime`` Tokio server when its wheel is importable
  (soak-tested nightly, see ``.github/workflows/realtime-soak.yml``),
  otherwise an autostarted FastAPI/uvicorn server. Opt in explicitly via
  the ``transport`` argument or the ``REALTIME_TRANSPORT`` env var —
  constructing it starts a background server, so it is never selected
  implicitly.

Any other value for ``transport`` / ``REALTIME_TRANSPORT`` is a
configuration error and raises :class:`ValueError` immediately (issue
#8869) rather than silently falling back to the file transport.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

from .transport_file import FileTransport, default_channel_path
from .ws_pubsub import WSPubSub

logger = get_logger(__name__)

__all__ = [
    "CHANNEL_REGISTRY",
    "SUPPORTED_TRANSPORTS",
    "Subscription",
    "publish",
    "register_channel",
    "subscribe",
]

# The only values ``transport`` / ``REALTIME_TRANSPORT`` may resolve to.
# Keep in contract lockstep with the docstring above and with
# ``_resolve_transport``'s error message.
SUPPORTED_TRANSPORTS: tuple[str, ...] = ("file", "ws")


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
    """Register a channel descriptor in :data:`CHANNEL_REGISTRY`.

    Idempotent: re-registering the same name with identical fields is a
    no-op.  Re-registering with a different description or owner raises
    :class:`ValueError` so that naming collisions surface early.

    Args:
        name: Channel name string (e.g. ``"pose/canonical"``).  Must be a
            non-empty string; whitespace-only values are rejected.
        description: Short human-readable description of the channel's
            payload semantics.
        owner_tool_id: Tool id of the canonical publisher, or ``None`` if
            any tool may publish on this channel.

    Raises:
        ValueError: If ``name`` is empty/whitespace, or if a channel with
            the same ``name`` but different fields is already registered.
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
    watcher down. It is idempotent. Backed uniformly by a single
    no-argument callable regardless of which transport produced it, so
    callers never reach into transport internals (Law of Demeter).
    """

    channel: str
    _unsubscribe_fn: Callable[[], None] | None = None
    _closed: bool = field(default=False)

    def unsubscribe(self) -> None:
        """Tear down the transport watcher and release resources.

        Safe to call multiple times; subsequent calls are no-ops.
        Any exception raised by the underlying transport is caught and
        logged so that cleanup code is never interrupted by a transport
        error.
        """
        if self._closed:
            return
        self._closed = True
        if self._unsubscribe_fn is not None:
            try:
                self._unsubscribe_fn()
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Subscription.unsubscribe failed on channel %s", self.channel
                )


# Module-global file transport, lazily initialised. The transport is
# stateless across processes (the on-disk log is the source of truth)
# and cheap to construct, so a single instance per process is fine.
_TRANSPORT: FileTransport | None = None

# Module-global websocket transport, lazily initialised on first use.
# Unlike the file transport this one is *not* free to construct: it
# autostarts a background server (Rust in-process, or a spawned
# FastAPI/uvicorn process). Only build it when a caller actually asks
# for "ws", never speculatively.
_WS_TRANSPORT: WSPubSub | None = None


def _get_transport() -> FileTransport:
    global _TRANSPORT
    if _TRANSPORT is None:
        _TRANSPORT = FileTransport(default_channel_path)
    return _TRANSPORT


def _get_ws_transport() -> WSPubSub:
    global _WS_TRANSPORT
    if _WS_TRANSPORT is None:
        _WS_TRANSPORT = WSPubSub()
    return _WS_TRANSPORT


def _resolve_transport(explicit: str | None) -> str:
    """Resolve and validate the transport to use for one call.

    Precedence: the explicit *transport* argument, then the
    ``REALTIME_TRANSPORT`` env var, else ``"file"``.

    Contract (issue #8869): an unsupported value is a configuration
    error and fails loudly here — it must never silently fall back to
    another transport.

    Args:
        explicit: Caller-supplied transport override, or ``None`` to
            fall back to the environment/default.

    Returns:
        One of :data:`SUPPORTED_TRANSPORTS`.

    Raises:
        ValueError: If the resolved value is not in
            :data:`SUPPORTED_TRANSPORTS`.
    """
    transport = (
        explicit
        if explicit is not None
        else os.environ.get("REALTIME_TRANSPORT", "file")
    )
    if transport not in SUPPORTED_TRANSPORTS:
        raise ValueError(
            f"unsupported realtime transport {transport!r}; supported "
            f"values are {SUPPORTED_TRANSPORTS!r}"
        )
    return transport


def publish(channel: str, payload: Any, transport: str | None = None) -> None:
    """Publish *payload* on *channel*.

    *payload* must be JSON-serialisable. Once the transport is resolved,
    delivery errors (I/O, network) are logged and swallowed — callers
    should treat realtime as a hint layer, never a critical path. An
    unsupported *transport* is different: that is a caller/config
    mistake, not a delivery failure, so it raises immediately instead of
    being swallowed or silently downgraded to the file transport.

    Args:
        channel: Channel to publish on (e.g., "scope/topic/sub")
        payload: JSON-serialisable dict to publish
        transport: Optional transport override ("file" or "ws"). If not
            provided, uses REALTIME_TRANSPORT env var or defaults to "file".

    Raises:
        ValueError: If the resolved transport is not one of
            :data:`SUPPORTED_TRANSPORTS`.
    """
    if not isinstance(channel, str) or not channel.strip():
        logger.warning("realtime.publish: invalid channel %r", channel)
        return
    resolved = _resolve_transport(transport)
    try:
        if resolved == "ws":
            _get_ws_transport().publish(channel, payload)
        else:
            _get_transport().publish(channel, payload)
    except Exception:
        logger.exception("realtime.publish failed on channel %s", channel)


def subscribe(
    channel: str,
    callback: Callable[[Any], None],
    transport: str | None = None,
) -> Subscription:
    """Register *callback* to fire for every payload published on *channel*.

    The callback runs on a transport-owned daemon thread.  Consumers that
    need to touch Qt widgets must marshal back to the GUI thread (e.g. via
    ``QMetaObject.invokeMethod`` or a ``QtCore.pyqtSignal``).

    If the underlying transport raises during setup, the error is logged and
    a closed :class:`Subscription` is returned rather than propagating the
    exception — callers can check ``sub._closed`` if they need to detect the
    failure. An unsupported *transport* is not a setup failure and is not
    swallowed this way; see Raises below.

    Args:
        channel: Channel name to subscribe to (e.g. ``"pose/canonical"``).
            Must be a non-empty string.
        callback: Callable invoked with the decoded payload dict each time a
            message arrives.  Must accept a single positional argument.
        transport: Optional transport override ("file" or "ws"). If not
            provided, uses REALTIME_TRANSPORT env var or defaults to "file".

    Returns:
        A :class:`Subscription` handle.  Call
        :meth:`Subscription.unsubscribe` to stop receiving messages.

    Raises:
        ValueError: If ``channel`` is empty or whitespace, or if the
            resolved transport is not one of :data:`SUPPORTED_TRANSPORTS`.
        TypeError: If ``callback`` is not callable.
    """
    if not isinstance(channel, str) or not channel.strip():
        raise ValueError("channel must be a non-empty string")
    if not callable(callback):
        raise TypeError("callback must be callable")
    resolved = _resolve_transport(transport)
    try:
        if resolved == "ws":
            ws_sub = _get_ws_transport().subscribe(channel, callback)
            return Subscription(channel=channel, _unsubscribe_fn=ws_sub.unsubscribe)
        file_transport = _get_transport()
        token = file_transport.subscribe(channel, callback)
        return Subscription(
            channel=channel,
            _unsubscribe_fn=lambda: file_transport.unsubscribe(token),
        )
    except Exception:
        logger.exception("realtime.subscribe failed on channel %s", channel)
        return Subscription(channel=channel, _unsubscribe_fn=None, _closed=True)
