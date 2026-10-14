"""Helpers shared by the engine adapters."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..model import Anthropometry
from ..names import CANONICAL_COORDINATES, engine_coordinate_name
from ..packs import PackLocation


def build_model_text(pack: PackLocation, lift: str, anthro: Anthropometry) -> str:
    """Call the pack's own ``build_<lift>_model`` generator."""
    pack.activate()
    module = importlib.import_module(f"{pack.package}.exercises.{lift}.{lift}_model")
    builder = getattr(module, f"build_{lift}_model")
    return str(
        builder(
            body_mass=anthro.body_mass_kg,
            height=anthro.height_m,
            plate_mass_per_side=anthro.plate_mass_per_side_kg,
        )
    )


def native_angles(engine: str, q: Mapping[str, float]) -> dict[str, float]:
    """Translate canonical angles to the engine's coordinate names.

    Raises:
        ValueError: If *q* names a coordinate outside the canonical set.
    """
    unknown = sorted(set(q) - set(CANONICAL_COORDINATES))
    if unknown:
        raise ValueError(f"unknown canonical coordinate(s): {unknown}")
    return {engine_coordinate_name(engine, name): float(v) for name, v in q.items()}


PELVIS_ORIGIN = (0.0, 0.0, 1.0)


HandJoint = tuple[str, str, str, str]
"""``(joint name, joint type, parent body, child body)`` of one model joint."""

_SIDES = ("l", "r")


def hand_bar_weld_side(body_a: str, body_b: str) -> str | None:
    """Side (``"l"``/``"r"``) when one body is a hand and the other the bar.

    Orientation does not matter: a pack may weld the bar to the hand (bar is
    child) or hang the hand off the bar as a leaf (hand is child), e.g.
    Drake's ``hand_r`` welded to ``barbell_shaft``. A dummy grip frame such as
    ``barbell_grip_r`` is not a hand, so it never matches.

    Raises:
        TypeError: If either body name is not a ``str``.
    """
    for name in (body_a, body_b):
        if not isinstance(name, str):
            raise TypeError(f"body names must be str, got {type(name).__name__}")
    for side in _SIDES:
        hand = f"hand_{side}"
        if (body_a == hand and body_b.startswith("barbell")) or (
            body_b == hand and body_a.startswith("barbell")
        ):
            return side
    return None


def hand_attachment(
    joints: Iterable[HandJoint],
    welds: Iterable[tuple[str, str]] = (),
) -> dict[str, str]:
    """Per-hand description of how the bar is tied to it (shared by adapters).

    *joints* are the model's tree joints; *welds* are loop-closure
    ``(parent, child)`` body pairs (e.g. Drake's ``biomech:weld``). A hand is
    attached only through a real hand/bar joint or weld; a joint that merely
    hangs a dummy frame off the bar named after the hand is reported as
    ``"none (only dummy frame ...)"``.

    Postcondition: returns ``{"l": ..., "r": ...}``; unattached values start
    with ``"none"``.
    """
    out = dict.fromkeys(_SIDES, "none")
    joint_list = list(joints)
    for _name, jtype, parent, child in joint_list:
        side = hand_bar_weld_side(parent, child)
        if side is not None:
            role = "bar is child" if parent == f"hand_{side}" else "hand is child"
            out[side] = f"{jtype} joint ({role})"
    for parent, child in welds:
        side = hand_bar_weld_side(parent, child)
        if side is not None and out[side].startswith("none"):
            out[side] = "weld constraint (loop closure)"
    for name, _jtype, parent, child in joint_list:
        for side in _SIDES:
            if (
                out[side] == "none"
                and f"hand_{side}" in name
                and parent.startswith("barbell")
            ):
                out[side] = f"none (only dummy frame {child})"
    return out


def pack_api(package: str, module: str, attr: str) -> Callable[..., Any] | None:
    """The pack's own callable ``<package>.<module>.<attr>``, or ``None``.

    ``None`` means the pack does not ship that API (module or attribute
    absent); an import error raised *inside* an existing module propagates.
    """
    qualified = f"{package}.{module}"
    try:
        mod = importlib.import_module(qualified)
    except ModuleNotFoundError as exc:
        if exc.name not in (qualified, package):
            raise
        return None
    fn = getattr(mod, attr, None)
    return fn if callable(fn) else None


def contact_unavailable(reason: str) -> dict[str, Any]:
    """``start_contact_force`` shape when the value cannot be supplied."""
    if not reason:
        raise ValueError("an unavailable contact value needs a reason")
    return {"value_n": None, "reason": reason}


def contact_measured(
    vertical_n: float, *, source: str, n_feet_in_contact: int
) -> dict[str, Any]:
    """``start_contact_force`` shape for a value read through a pack API."""
    return {
        "value_n": float(vertical_n),
        "reason": None,
        "source": source,
        "n_feet_in_contact": int(n_feet_in_contact),
    }
