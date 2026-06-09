"""Public API for the realtime IPC facade.

This module is intentionally tiny: it owns the channel registry and a
small dispatch glue layer that delegates to a transport (file by
default; websocket as a follow-up). The Pose Studio cross-tool demo
(Subtask 6 of EPIC #4993) only needs the file transport, so that is
what is wired up here.
"""

from __future__ import annotations

import atexit
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

from .transport_file import FileTransport, default_channel_path

logger = get_logger(__name__)

__all__ = [
    "CHANNEL_REGISTRY",
    "Subscription",
    "is_healthy",
    "publish",
    "register_channel",
    "shutdown_realtime",
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
    watcher down. It is idempotent.
    """

    channel: str
    _transport: FileTransport | None = None
    _token: int = -1
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
# Guards check-then-act on _TRANSPORT so concurrent first-callers cannot each
# construct a transport and orphan the loser's polling thread (issue #7148 D1).
_TRANSPORT_LOCK = threading.Lock()

# First-failure latch for publish health (issue #7149 D1): a persistent failure
# (e.g. unwritable channel root) is reported once via WARNING + is_healthy(),
# instead of silently spamming the log on every publish.
_HEALTH_LOCK = threading.Lock()
_PUBLISH_FAILURE_REASON: str | None = None


def _get_transport() -> FileTransport:
    global _TRANSPORT
    # Double-checked locking: the fast path avoids the lock once initialised.
    if _TRANSPORT is None:
        with _TRANSPORT_LOCK:
            if _TRANSPORT is None:
                _TRANSPORT = FileTransport(default_channel_path)
    return _TRANSPORT


def is_healthy() -> bool:
    """Return ``True`` unless a prior publish failed persistently (#7149).

    Realtime is a hint layer, so publish never raises; callers (e.g. UIs that
    depend on a channel) can probe this to surface a degraded state once.
    """
    with _HEALTH_LOCK:
        return _PUBLISH_FAILURE_REASON is None


def validate_realtime() -> str | None:
    """Return the latched publish-failure reason, or ``None`` if healthy."""
    with _HEALTH_LOCK:
        return _PUBLISH_FAILURE_REASON


def shutdown_realtime() -> None:
    """Stop the transport polling watcher and clear global state (idempotent).

    Registered via :func:`atexit` and intended for test teardown so channels do
    not leak between tests and buffered writes are flushed (issue #7148 D2).
    """
    global _TRANSPORT, _PUBLISH_FAILURE_REASON
    with _TRANSPORT_LOCK:
        transport = _TRANSPORT
        _TRANSPORT = None
    if transport is not None:
        try:
            transport.shutdown()
        except Exception:  # pragma: no cover - defensive teardown
            logger.exception("realtime.shutdown_realtime failed")
    with _HEALTH_LOCK:
        _PUBLISH_FAILURE_REASON = None


atexit.register(shutdown_realtime)


def publish(channel: str, payload: Any, transport: str | None = None) -> None:
    """Publish *payload* on *channel*.

    *payload* must be JSON-serialisable. Errors are logged and swallowed
    — callers should treat realtime as a hint layer, never a critical
    path. The default transport is the file transport; the websocket
    transport can be opted into via ``REALTIME_TRANSPORT=ws`` in the
    future (not implemented here).

    Args:
        channel: Channel to publish on (e.g., "scope/topic/sub")
        payload: JSON-serialisable dict to publish
        transport: Optional transport override ("file" or "ws"). If not
            provided, uses REALTIME_TRANSPORT env var or defaults to "file".
    """
    if not isinstance(channel, str) or not channel.strip():
        logger.warning("realtime.publish: invalid channel %r", channel)
        return
    if transport is None:
        transport = os.environ.get("REALTIME_TRANSPORT", "file")
    if transport != "file":
        logger.debug(
            "realtime.publish: transport %r not wired in this build, "
            "falling back to file",
            transport,
        )
    global _PUBLISH_FAILURE_REASON
    try:
        _get_transport().publish(channel, payload)
    except Exception as exc:  # noqa: BLE001 - hint layer: never propagate, latch health
        # Latch the first persistent failure and warn once; subsequent failures
        # update is_healthy() silently rather than spamming the log (#7149 D1).
        with _HEALTH_LOCK:
            first_failure = _PUBLISH_FAILURE_REASON is None
            _PUBLISH_FAILURE_REASON = f"{type(exc).__name__}: {exc}"
        if first_failure:
            logger.warning(
                "realtime.publish failed on channel %s; marking realtime "
                "unhealthy (further failures suppressed): %s",
                channel,
                exc,
            )
        else:
            logger.debug("realtime.publish failed again on channel %s", channel)
        return
    # A successful publish clears a previously latched failure.
    with _HEALTH_LOCK:
        if _PUBLISH_FAILURE_REASON is not None:
            _PUBLISH_FAILURE_REASON = None
            logger.info("realtime.publish recovered on channel %s", channel)


def subscribe(channel: str, callback: Callable[[Any], None]) -> Subscription:
    """Register *callback* to fire for every payload published on *channel*.

    The callback runs on a transport-owned daemon thread.  Consumers that
    need to touch Qt widgets must marshal back to the GUI thread (e.g. via
    ``QMetaObject.invokeMethod`` or a ``QtCore.pyqtSignal``).

    If the underlying transport raises during setup, the error is logged and
    a closed :class:`Subscription` is returned rather than propagating the
    exception — callers can check ``sub._closed`` if they need to detect the
    failure.

    Args:
        channel: Channel name to subscribe to (e.g. ``"pose/canonical"``).
            Must be a non-empty string.
        callback: Callable invoked with the decoded payload dict each time a
            message arrives.  Must accept a single positional argument.

    Returns:
        A :class:`Subscription` handle.  Call
        :meth:`Subscription.unsubscribe` to stop receiving messages.

    Raises:
        ValueError: If ``channel`` is empty or whitespace.
        TypeError: If ``callback`` is not callable.
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
