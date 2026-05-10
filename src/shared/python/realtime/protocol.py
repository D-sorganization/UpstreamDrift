"""Protocol primitives for the realtime IPC layer.

Defines the channel-name validator and the :class:`Subscription` frozen
dataclass returned by both backends.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["Subscription", "validate_channel"]


# Channel naming: scope/topic, with optional further nesting.
# - First segment must start with a lowercase letter, then [a-z0-9_]*.
# - Each subsequent segment is [a-z0-9_]+.
# - At least two segments separated by '/'.
_CHANNEL_RE = re.compile(r"^[a-z][a-z0-9_]*(/[a-z0-9_]+)+$")


def validate_channel(name: str) -> None:
    """Validate a channel name against the ``scope/topic`` pattern.

    Args:
        name: Channel name to validate.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` does not match
            ``^[a-z][a-z0-9_]*(/[a-z0-9_]+)+$``.
    """
    if not isinstance(name, str):
        raise TypeError(f"channel name must be a string, got {type(name).__name__}")
    if not _CHANNEL_RE.match(name):
        raise ValueError(
            f"invalid channel name {name!r}; must match "
            r"'^[a-z][a-z0-9_]*(/[a-z0-9_]+)+$' (scope/topic pattern)"
        )


@dataclass(frozen=True)
class Subscription:
    """A live subscription handle returned by ``subscribe()``.

    The ``unsubscribe`` callable is stored in a frozen field; calling
    :meth:`unsubscribe` invokes it. Backends are responsible for ensuring
    idempotency.
    """

    channel: str
    callback: Callable[[dict], None]
    _unsubscribe: Callable[[], None] = field(repr=False)

    def unsubscribe(self) -> None:
        """Tear down the underlying watcher / WebSocket / poller."""
        self._unsubscribe()
