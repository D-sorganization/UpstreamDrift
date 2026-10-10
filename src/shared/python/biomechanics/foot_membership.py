"""Foot membership of contact bodies (GCV-2, #11708; epic #11706).

One place decides which rigid body belongs to which foot, so every engine
groups its ground contacts identically.  The full-body spec contact groups
(``contact.spheres[*].body``, e.g. ``calcn_l`` / ``calcn_r``) are the source of
truth for the bodies that carry ground contact; :func:`foot_of_body` is the
name rule that classifies them (and the names used by the other humanoid
models: ``left_foot``, ``LeftFoot``, ``toes_r``, ``ud_contact_heel_r`` ...).

Pure Python; no engine imports.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Literal

__all__ = ["FOOT_SIDES", "Side", "foot_bodies_from_spec", "foot_of_body"]

Side = Literal["left", "right"]
FOOT_SIDES: tuple[Side, Side] = ("left", "right")

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")
_FOOT_TOKENS = frozenset(
    {"foot", "feet", "calcn", "calcaneus", "toe", "toes", "heel", "forefoot", "talus"}
)
_LEFT_TOKENS = frozenset({"l", "left"})
_RIGHT_TOKENS = frozenset({"r", "right"})


def foot_of_body(body_name: str) -> Side | None:
    """Foot (``"left"`` or ``"right"``) a body belongs to, else ``None``.

    A body is a foot body when one of its name tokens is a foot-segment word
    and exactly one side token is present.  Names that are not feet, or whose
    side is missing or contradictory, return ``None`` (never a guessed side).

    Raises:
        TypeError: ``body_name`` is not a string.
    """
    if not isinstance(body_name, str):
        raise TypeError(f"body_name must be a string, got {type(body_name).__name__}")
    tokens = {t.lower() for t in _TOKEN_SPLIT.split(body_name) if t}
    if not tokens & _FOOT_TOKENS:
        return None
    left = bool(tokens & _LEFT_TOKENS)
    right = bool(tokens & _RIGHT_TOKENS)
    if left == right:
        return None
    return "left" if left else "right"


def foot_bodies_from_spec(spec: Mapping[str, Any]) -> dict[str, Side]:
    """Bodies carrying foot contact spheres in a full-body ``spec``, by side.

    Postcondition: every value is ``"left"`` or ``"right"``; bodies of other
    contact spheres (e.g. a ball) and spheres on unclassifiable bodies are
    omitted.  A spec without contact groups gives an empty mapping.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a mapping")
    out: dict[str, Side] = {}
    for sphere in spec.get("contact", {}).get("spheres", ()):
        body = str(sphere["body"])
        side = foot_of_body(body)
        if side is not None:
            out[body] = side
    return out
