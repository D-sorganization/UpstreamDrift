"""Canonical club-head body naming shared by every engine (UD#12913, UD#13015).

The same club body carries different names per engine:

* MuJoCo full-body golfer: the Simscape solid name
  ``solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector``
* OpenSim / Pinocchio: ``Clubhead`` (see ``full_body_osim`` name mapping)
* MuJoCo swing models: ``clubhead``; some generated models: ``club_head``
* Drake: a generic ``body_N`` that is not recoverable by name.

This module is the single alias table.  It is pure Python (no engine import)
so every engine adapter can share it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)

#: Canonical engine-neutral name of the club-head body (OpenSim / Pinocchio).
CANONICAL_CLUB_HEAD_NAME = "Clubhead"

#: Spec / MuJoCo full-body name of the same body (Simscape solid reference).
SPEC_CLUB_HEAD_BODY_NAME = "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector"

#: Accepted exact names, compared case-insensitively, in priority order.
CLUB_HEAD_BODY_ALIASES: tuple[str, ...] = (
    "clubhead",
    "club_head",
    SPEC_CLUB_HEAD_BODY_NAME,
)


def normalise_body_name(name: str) -> str:
    """Lower-case a body name and drop ``_`` / ``-`` separators."""
    return name.lower().replace("_", "").replace("-", "")


def find_club_head_index(
    names: Iterable[str | None], *, owner: str = "", strict: bool = False
) -> int | None:
    """Return the index in ``names`` of the club-head body.

    Exact alias matches win; otherwise the first name whose normalised form
    contains ``clubhead`` is used (e.g. ``club_head_link``).  When nothing
    matches, ``strict=True`` raises ``ValueError`` listing the candidate names
    tried; otherwise a WARNING is logged and ``None`` returned.
    """
    indexed = [(i, n) for i, n in enumerate(names) if n]
    lowered = {n.lower(): i for i, n in reversed(indexed)}
    for alias in CLUB_HEAD_BODY_ALIASES:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    for i, name in indexed:
        if "clubhead" in normalise_body_name(name):
            return i
    msg = (
        f"{owner or 'club-head lookup'}: no club-head body found "
        f"(tried aliases {', '.join(CLUB_HEAD_BODY_ALIASES)}; "
        f"model bodies: {', '.join(n for _, n in indexed) or '<none>'}); "
        "club-head speed and club metrics will be unavailable for this model."
    )
    if strict:
        raise ValueError(msg)
    logger.warning(msg)
    return None
