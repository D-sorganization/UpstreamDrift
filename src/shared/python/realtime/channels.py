"""Central registry of well-known realtime channels.

Each channel declares a frequency hint that selects the default transport:

- ``"high"`` → WebSocket (low-latency, high-rate).
- ``"low"``  → file pub-sub (durable, easier to inspect on disk).

Wildcards: a channel name may contain ``<...>`` placeholders (e.g.
``engine/<name>/state``); concrete channel names are matched against the
literal prefix that precedes the wildcard segment.
"""

from __future__ import annotations

from typing import Literal

from .protocol import validate_channel

__all__ = [
    "FrequencyHint",
    "get_channel_transport",
    "register_channel",
]


FrequencyHint = Literal["low", "high"]
_Transport = Literal["file", "ws"]


# Exact-match channels: maps full channel name to frequency hint.
_EXACT: dict[str, FrequencyHint] = {}

# Prefix-match channels for wildcard registrations like
# ``engine/<name>/state``: maps the literal prefix (everything before the
# first ``<...>`` segment, including the trailing ``/``) to a list of
# (suffix, hint) tuples. The suffix is everything after the wildcard.
_PREFIX: dict[str, list[tuple[str, FrequencyHint]]] = {}


def _split_wildcard(pattern: str) -> tuple[str, str] | None:
    """Return (prefix, suffix) if ``pattern`` contains a single ``<...>``
    wildcard segment, otherwise ``None``.
    """
    parts = pattern.split("/")
    wildcard_indices = [
        i for i, p in enumerate(parts) if p.startswith("<") and p.endswith(">")
    ]
    if not wildcard_indices:
        return None
    if len(wildcard_indices) > 1:
        raise ValueError(
            f"channel pattern {pattern!r} has multiple wildcards; only one supported"
        )
    idx = wildcard_indices[0]
    prefix = "/".join(parts[:idx]) + "/"
    suffix = "/" + "/".join(parts[idx + 1 :]) if idx + 1 < len(parts) else ""
    return prefix, suffix


def register_channel(name: str, frequency_hint: FrequencyHint) -> None:
    """Register a channel (or wildcard pattern) with a frequency hint.

    Args:
        name: Channel name, or a wildcard pattern like ``engine/<name>/state``.
        frequency_hint: ``"low"`` or ``"high"``.

    Raises:
        ValueError: If ``frequency_hint`` is not one of ``"low"`` / ``"high"``,
            or if a literal name fails channel-name validation.
    """
    if frequency_hint not in ("low", "high"):
        raise ValueError(
            f"frequency_hint must be 'low' or 'high', got {frequency_hint!r}"
        )

    split = _split_wildcard(name)
    if split is None:
        # Literal channel — must pass validation.
        validate_channel(name)
        _EXACT[name] = frequency_hint
        return

    prefix, suffix = split
    _PREFIX.setdefault(prefix, []).append((suffix, frequency_hint))


def _lookup_hint(name: str) -> FrequencyHint | None:
    """Find the frequency hint for ``name``, or ``None`` if unknown."""
    if name in _EXACT:
        return _EXACT[name]
    for prefix, entries in _PREFIX.items():
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        # Must consume at least one segment for the wildcard.
        for suffix, hint in entries:
            if suffix == "":
                if "/" not in remainder and remainder:
                    return hint
            elif remainder.endswith(suffix):
                middle = remainder[: -len(suffix)]
                if middle and "/" not in middle:
                    return hint
    return None


def get_channel_transport(name: str) -> _Transport:
    """Resolve the preferred transport for ``name``.

    Returns ``"ws"`` for high-frequency channels, ``"file"`` for low-frequency
    or unknown channels.
    """
    hint = _lookup_hint(name)
    if hint == "high":
        return "ws"
    return "file"


# --- well-known channels ----------------------------------------------------

register_channel("pose/canonical", "high")
register_channel("engine/<name>/state", "high")
register_channel("target/active", "low")
register_channel("session/marker", "low")
