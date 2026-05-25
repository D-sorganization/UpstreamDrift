"""Realtime IPC facade for cross-tool live data streaming.

This package exposes a tiny ``publish`` / ``subscribe`` pair that lets one
launcher tool stream events to another without the two having to share a
process or a Qt widget tree. The default transport is a JSON-line append
log under the user cache dir; a websocket transport can be opted into via
the ``REALTIME_TRANSPORT`` env var (left as a follow-up — see Subtask 4
of EPIC #4993).

The contract is intentionally narrow:

- :func:`publish` writes a single JSON-serialisable payload onto a named
  channel (e.g. ``"pose/canonical"``).
- :func:`subscribe` registers a callback that fires for every published
  payload on a channel and returns a :class:`Subscription` whose
  :meth:`Subscription.unsubscribe` tears the watcher down.

Both sides are best-effort: if the transport is unavailable, ``publish``
swallows the error after logging it and ``subscribe`` returns an inert
:class:`Subscription`. Tools should treat realtime as a *hint* layer, not
a transactional one.

The file transport polls at ~30 Hz on a background daemon thread and only
delivers payloads written *after* :func:`subscribe` is called, so two
processes started in either order both observe live updates.
"""

from __future__ import annotations

from .api import (
    CHANNEL_REGISTRY,
    Subscription,
    publish,
    register_channel,
    subscribe,
)

# Public API version (SemVer MAJOR.MINOR.PATCH).
#
# Bump rules (per issue #5917, ADR-0007):
# - MAJOR: breaking change to ``publish`` / ``subscribe`` / ``Subscription``
#   signatures, transport wire format, or channel-name contract.
# - MINOR: backwards-compatible additions (new public functions, new
#   optional kwargs, new transports).
# - PATCH: bug fixes that do not change the public surface.
__version__ = "1.0.0"

# Wire-format version for the JSON-line file transport and websocket
# payloads. Consumers that persist or replay realtime traffic should
# pin this and refuse mismatched majors.
SCHEMA_VERSION = "1.0.0"

__all__ = [
     "CHANNEL_REGISTRY",
     "Subscription",
     "publish",
     "register_channel",
     "SCHEMA_VERSION",
     "subscribe",
     "__version__",
]
