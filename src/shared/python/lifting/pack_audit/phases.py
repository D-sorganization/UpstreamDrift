"""Read each pack's own named phase poses (its de-facto reference trajectory).

The packs do not share a phase schema (RM#2025): attribute names, registry
keys and joint spellings all differ.  This module normalises them to
``PackPhase`` records and keeps what could not be mapped, so the audit can
report it instead of silently dropping targets.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from .names import expand_phase_key
from .packs import PackLocation

_LIFT_KEYS = {"squat": ("squat", "back_squat")}


@dataclass(frozen=True)
class PackPhase:
    """One named phase with canonical-angle targets and mapping diagnostics."""

    name: str
    fraction: float
    angles: dict[str, float]
    unmapped_keys: tuple[str, ...] = ()
    interpreted_keys: tuple[str, ...] = ()
    raw: dict[str, float] = field(default_factory=dict)


def _registry(pack: PackLocation) -> tuple[dict, str, str, str]:
    """Return ``(registry, fraction attr, targets attr, phases attr)``."""
    pack.activate()
    pkg = pack.package
    if pack.engine == "mujoco":
        mod = importlib.import_module(f"{pkg}.optimization.exercise_objectives")
        return mod.OBJECTIVE_REGISTRY, "fraction", "target_joints", "phases"
    if pack.engine == "opensim":
        mod = importlib.import_module(f"{pkg}.optimization.exercise_objectives")
        return mod.EXERCISE_OBJECTIVES, "time_fraction", "joint_targets", "phases"
    if pack.engine == "pinocchio":
        mod = importlib.import_module(f"{pkg}.optimization.objectives.registry")
        return mod.EXERCISE_OBJECTIVES, "fraction", "target_joints", "phases"
    mod = importlib.import_module(f"{pkg}.optimization.exercise_objectives")
    return mod._OBJECTIVES, "time_fraction", "joint_angles", "phases"


def pack_phases(pack: PackLocation, lift: str) -> list[PackPhase]:
    """The pack's phase list for *lift*, mapped to canonical coordinate names.

    Raises:
        KeyError: If the pack has no objective for *lift*.
    """
    registry, frac_attr, target_attr, phase_attr = _registry(pack)
    keys = _LIFT_KEYS.get(lift, (lift,))
    key = next((k for k in keys if k in registry), None)
    if key is None:
        raise KeyError(f"{pack.engine} pack has no objective for {lift!r}")
    out = []
    for phase in getattr(registry[key], phase_attr):
        raw = dict(getattr(phase, target_attr))
        angles: dict[str, float] = {}
        unmapped, interpreted = [], []
        for native, value in raw.items():
            names, exact = expand_phase_key(pack.engine, native)
            if not names:
                unmapped.append(native)
                continue
            if not exact:
                interpreted.append(native)
            for name in names:
                angles[name] = float(value)
        out.append(
            PackPhase(
                name=str(phase.name),
                fraction=float(getattr(phase, frac_attr)),
                angles=angles,
                unmapped_keys=tuple(unmapped),
                interpreted_keys=tuple(interpreted),
                raw={k: float(v) for k, v in raw.items()},
            )
        )
    return out
