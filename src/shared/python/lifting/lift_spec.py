"""Lift definitions read from the vendored biomech parity standard (#11743).

The standard (Repository_Management ``shared_scripts/model_parity/
biomech_parity_standard.json``, v2.x, RM#2060/#2025/#2077) is the single
source of truth for every lift's ordered named phases, root-joint policy and
start/catch/lockout via-poses.  It is vendored byte-for-byte under
``_canonical/`` with a ``MANIFEST.json`` digest; never edit the copy, re-sync
it from Repository_Management instead.

Callers use the accessors below rather than walking the raw JSON, so a
standard bump changes phase names and poses everywhere at once.

Postconditions (checked, ``AssertionError`` on violation):

* phases are non-empty, uniquely named, number ``phase_count``, and their
  ``t_frac`` start at 0, strictly increase and stay below 1;
* every via-pose binds to one of the lift's phases and uses only canonical
  coordinate names (``{side}`` patterns expanded to both sides).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .pack_audit.names import CANONICAL_COORDINATES, LIFTS

STANDARD_PATH = (
    Path(__file__).resolve().parent / "_canonical" / ("biomech_parity_standard.json")
)


@dataclass(frozen=True)
class LiftPhase:
    """One named phase: ``t_frac`` is the cycle fraction at which it starts."""

    name: str
    definition: str
    t_frac: float


@dataclass(frozen=True)
class ViaPose:
    """A whole-body pose the lift must pass through (standard 1.75 m lifter).

    ``coordinates`` maps canonical coordinate names to radians; positions are
    metres in the standard's world frame (floor origin at mid-foot, Z up).
    """

    name: str
    phase: str
    root_position_m: tuple[float, float, float]
    root_orientation_rpy_rad: tuple[float, float, float]
    bar_position_m: tuple[float, float, float]
    coordinates: dict[str, float]


@dataclass(frozen=True)
class LiftGrip:
    """A lift's grip: palm-centre width on the bar and the standard's range."""

    attach: str
    kind: str
    width_m: float
    width_range_m: tuple[float, float]


@lru_cache(maxsize=1)
def _standard() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(STANDARD_PATH.read_text(encoding="utf-8"))
    return data


def standard_version() -> str:
    """Version string of the vendored parity standard (``"2.1.0"``...)."""
    return str(_standard()["version"])


def _lift_block(lift: str) -> dict[str, Any]:
    if lift not in LIFTS:
        raise ValueError(f"unknown lift {lift!r}; use {list(LIFTS)}")
    block: dict[str, Any] = _standard()["exercises"][lift]
    return block


def lift_phases(lift: str) -> tuple[LiftPhase, ...]:
    """Ordered named phases of *lift* from the standard.

    Raises:
        ValueError: If *lift* is not one of :data:`LIFTS` or :data:`ADDITIONAL_LIFTS`.
    """
    from .additional_movements import additional_movement_spec, is_additional_movement

    if is_additional_movement(lift):
        spec = additional_movement_spec(lift)
        return spec.phases
    block = _lift_block(lift)
    phases = tuple(
        LiftPhase(str(p["name"]), str(p["definition"]), float(p["t_frac"]))
        for p in block.get("phases", [])
    )
    fractions = [p.t_frac for p in phases]
    assert phases, f"standard gives {lift!r} no named phases"
    assert len(phases) == int(block["phase_count"]), f"{lift}: phase_count mismatch"
    assert len({p.name for p in phases}) == len(phases), f"{lift}: duplicate phase"
    assert fractions[0] == 0.0 and fractions[-1] < 1.0, f"{lift}: t_frac range"
    assert all(b > a for a, b in zip(fractions, fractions[1:], strict=False)), (
        f"{lift}: t_frac must strictly increase"
    )
    return phases


def lift_phase_names(lift: str) -> tuple[str, ...]:
    """Ordered phase names of *lift* (see :func:`lift_phases`)."""
    return tuple(p.name for p in lift_phases(lift))


def lift_root_joint(lift: str) -> str:
    """Root-joint policy of *lift*: ``"free"`` or ``"weld"`` (RM#2024)."""
    from .additional_movements import additional_movement_spec, is_additional_movement

    if is_additional_movement(lift):
        spec = additional_movement_spec(lift)
        return spec.root_joint
    joint = str(_lift_block(lift).get("root_joint", _standard()["root"]["joint"]))
    assert joint in ("free", "weld"), f"{lift}: unknown root joint {joint!r}"
    return joint


def lift_via_pose_names(lift: str) -> tuple[str, ...]:
    """Names of the via-poses *lift* defines (``start``, ``catch``, ``lockout``)."""
    from .additional_movements import additional_movement_spec, is_additional_movement

    if is_additional_movement(lift):
        spec = additional_movement_spec(lift)
        return tuple(spec.via_poses.keys())
    return tuple(_lift_block(lift).get("via_poses", {}))


def lift_grip(lift: str) -> LiftGrip:
    """Grip of *lift* from the standard (palm centres ``width_m`` apart).

    Postcondition: ``width_m`` lies inside ``width_range_m``.

    Raises:
        ValueError: If *lift* is unknown.
    """
    from .additional_movements import additional_movement_spec, is_additional_movement

    if is_additional_movement(lift):
        spec = additional_movement_spec(lift)
        return spec.grip
    raw = _lift_block(lift)["grip"]
    low, high = (float(v) for v in raw["width_range_m"])
    grip = LiftGrip(
        attach=str(raw["attach"]),
        kind=str(raw["type"]),
        width_m=float(raw["width_m"]),
        width_range_m=(low, high),
    )
    assert low <= grip.width_m <= high, f"{lift}: grip width outside its range"
    return grip


def bar_axis() -> tuple[float, float, float]:
    """Bar long axis in the standard's canonical frame (``+Y``: medio-lateral)."""
    return _triple(_standard()["grip_point"]["bar_axis"])


def standard_height_m() -> float:
    """Height the standard's positions are given for (they scale linearly)."""
    return float(_standard()["anthropometrics"]["height_m"])


def _triple(values: Any) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in values)
    return (x, y, z)


def _expand_sides(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in raw.items():
        sides = _standard()["sides"] if "{side}" in name else [None]
        for side in sides:
            out[name.format(side=side) if side else name] = float(value)
    return out


def lift_via_pose(lift: str, pose: str) -> ViaPose:
    """The standard's via-pose *pose* of *lift*, sides expanded.

    Raises:
        ValueError: If *lift* is unknown or has no via-pose *pose*.
    """
    from .additional_movements import additional_movement_spec, is_additional_movement

    if is_additional_movement(lift):
        spec = additional_movement_spec(lift)
        return spec.via_pose(pose)
    poses = _lift_block(lift).get("via_poses", {})

    if pose not in poses:
        raise ValueError(f"{lift!r} has no via-pose {pose!r}; has {sorted(poses)}")
    raw = poses[pose]
    via = ViaPose(
        name=pose,
        phase=str(raw["phase"]),
        root_position_m=_triple(raw["root"]["position_m"]),
        root_orientation_rpy_rad=_triple(raw["root"]["orientation_rpy_rad"]),
        bar_position_m=_triple(raw["bar_position_m"]),
        coordinates=_expand_sides(raw["coordinates"]),
    )
    assert via.phase in lift_phase_names(lift), f"{lift}/{pose}: unbound phase"
    unknown = set(via.coordinates) - set(CANONICAL_COORDINATES)
    assert not unknown, f"{lift}/{pose}: non-canonical coordinates {sorted(unknown)}"
    return via


_REEXPORT_NAMES: frozenset[str] = frozenset(
    {
        "ADDITIONAL_LIFTS",
        "AdditionalMovementSpec",
        "BarbellLoad",
        "StanceGeometry",
        "additional_movement_spec",
        "is_additional_movement",
    }
)


def __getattr__(name: str) -> Any:
    if name in _REEXPORT_NAMES:
        from . import additional_movements

        return getattr(additional_movements, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
