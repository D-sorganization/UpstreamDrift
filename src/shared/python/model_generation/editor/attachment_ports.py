"""Typed attachment ports and their compatibility rules.

A *port* is a named mount point on a model link. Every typed port has a
:class:`PortType` (what physically mates: a hand grip, a hip, a shoulder, an
ankle, ...) and a :class:`PortPolarity`. A host model exposes *sockets*; a part
that can be dropped onto the host exposes a *plug*. Two ports are compatible
only when the host side is a socket, the part side is a plug, the types match,
the left/right sides do not contradict each other, and the part is not heavier
than the socket's declared payload limit.

This module is pure data (no Qt, no XML) so the rules can be unit tested and
shared between the desktop Model Explorer and any other host.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PortType(str, Enum):
    """What kind of joint interface a port represents."""

    GRIP = "grip"
    HIP = "hip"
    SHOULDER = "shoulder"
    ANKLE = "ankle"
    NECK = "neck"
    MOUNT = "mount"


class PortPolarity(str, Enum):
    """Which half of a mating pair a port is."""

    SOCKET = "socket"
    PLUG = "plug"


SIDES: tuple[str, ...] = ("left", "right")


@dataclass(frozen=True)
class TypedPort:
    """The mating-relevant description of one port."""

    port_type: PortType
    polarity: PortPolarity
    side: str | None = None
    max_payload_kg: float | None = None

    def __post_init__(self) -> None:
        if self.side is not None and self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES} or None; got {self.side!r}")
        if self.max_payload_kg is not None and self.max_payload_kg <= 0:
            raise ValueError("max_payload_kg must be greater than zero")


@dataclass(frozen=True)
class PortCompatibility:
    """Outcome of a compatibility check, with a human-readable reason."""

    ok: bool
    reason: str = ""


def parse_port_type(value: str) -> PortType:
    """Parse a manifest string into a :class:`PortType`."""
    if not isinstance(value, str):
        raise TypeError("port type must be a string")
    try:
        return PortType(value.strip().lower())
    except ValueError:
        valid = ", ".join(item.value for item in PortType)
        raise ValueError(
            f"unknown port type {value!r}; expected one of {valid}"
        ) from None


def parse_port_polarity(value: str) -> PortPolarity:
    """Parse a manifest string into a :class:`PortPolarity`."""
    if not isinstance(value, str):
        raise TypeError("port polarity must be a string")
    try:
        return PortPolarity(value.strip().lower())
    except ValueError:
        valid = ", ".join(item.value for item in PortPolarity)
        raise ValueError(
            f"unknown port polarity {value!r}; expected one of {valid}"
        ) from None


def side_from_tags(tags: tuple[str, ...]) -> str | None:
    """Return ``left``/``right`` when exactly one side appears in ``tags``."""
    if tags is None:
        raise ValueError("tags must be provided")
    found = {tag.lower() for tag in tags} & set(SIDES)
    return next(iter(found)) if len(found) == 1 else None


def check_port_compatibility(
    host: TypedPort,
    part: TypedPort,
    *,
    part_mass_kg: float | None = None,
) -> PortCompatibility:
    """Decide whether ``part`` can mate with the host ``host`` port.

    Postcondition: ``ok`` is False exactly when ``reason`` is non-empty.
    """
    if host is None:
        raise ValueError("host must be provided")
    if part is None:
        raise ValueError("part must be provided")
    if host.polarity is not PortPolarity.SOCKET:
        return PortCompatibility(False, "host port is a plug, not a socket")
    if part.polarity is not PortPolarity.PLUG:
        return PortCompatibility(False, "part port is a socket, not a plug")
    if host.port_type is not part.port_type:
        return PortCompatibility(
            False,
            f"port type mismatch: {host.port_type.value} socket "
            f"cannot take a {part.port_type.value} plug",
        )
    if host.side and part.side and host.side != part.side:
        return PortCompatibility(
            False, f"side mismatch: {host.side} socket cannot take a {part.side} part"
        )
    limit = host.max_payload_kg
    if limit is not None and part_mass_kg is not None and part_mass_kg > limit:
        return PortCompatibility(
            False,
            f"payload {part_mass_kg:.2f} kg exceeds the socket limit of {limit:.2f} kg",
        )
    return PortCompatibility(True)


__all__ = [
    "SIDES",
    "PortCompatibility",
    "PortPolarity",
    "PortType",
    "TypedPort",
    "check_port_compatibility",
    "parse_port_polarity",
    "parse_port_type",
    "side_from_tags",
]
