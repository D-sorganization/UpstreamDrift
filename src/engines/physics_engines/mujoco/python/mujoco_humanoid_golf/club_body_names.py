"""Single canonical resolver for the golf club-head body (UD#12913).

Swing models name the body ``clubhead`` while a few generated models use
``club_head``. Analyzers previously did a substring match for ``club_head``
only, so ``clubhead`` models silently reported a 0.0 mph club speed.
"""

from __future__ import annotations

import logging

import mujoco

logger = logging.getLogger(__name__)

#: Canonical name first, then accepted aliases (exact, case-insensitive).
CLUB_HEAD_BODY_ALIASES: tuple[str, ...] = ("clubhead", "club_head")


def _normalise(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def find_club_head_body_id(model: mujoco.MjModel, *, owner: str = "") -> int | None:
    """Return the club-head body id, or ``None`` (with a WARNING) if absent.

    Exact alias matches win; otherwise the first body whose normalised name
    contains ``clubhead`` is used (e.g. ``club_head_link``).
    """
    if model is None:
        raise ValueError("model must be provided")
    names: list[tuple[int, str]] = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            names.append((i, body_name))
    for alias in CLUB_HEAD_BODY_ALIASES:
        for i, body_name in names:
            if body_name.lower() == alias:
                return i
    for i, body_name in names:
        if "clubhead" in _normalise(body_name):
            return i
    logger.warning(
        "%s: no club-head body found (tried aliases %s); club-head speed and "
        "club metrics will be unavailable for this model.",
        owner or "club-head lookup",
        ", ".join(CLUB_HEAD_BODY_ALIASES),
    )
    return None
