"""Single canonical resolver for the golf club-head body (UD#12913, UD#13015).

The alias table lives in :mod:`src.shared.python.club_body_names` so every
engine shares it; this module adapts it to ``mujoco.MjModel``.
"""

from __future__ import annotations

import mujoco

from src.shared.python.club_body_names import (
    CLUB_HEAD_BODY_ALIASES,
    find_club_head_index,
)

__all__ = ["CLUB_HEAD_BODY_ALIASES", "find_club_head_body_id"]


def find_club_head_body_id(
    model: mujoco.MjModel, *, owner: str = "", strict: bool = False
) -> int | None:
    """Return the club-head body id.

    With ``strict=False`` a missing body logs a WARNING and returns ``None``.
    With ``strict=True`` it raises ``ValueError`` listing the candidate names,
    so callers computing club-head metrics never silently drop them.
    """
    if model is None:
        raise ValueError("model must be provided")
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    return find_club_head_index(names, owner=owner, strict=strict)
