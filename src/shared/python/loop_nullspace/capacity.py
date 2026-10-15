"""Per-joint torque capacities for closed-loop null-space exploration (NSX-04).

Capacities are population priors read from
``src/config/loop_nullspace_registry.json`` (``torque_capacity``), not subject
measurements.  Every slot carries a verification status; slots whose value is
not read from a source are reported in ``CapacityVector.assumed_slots`` and
``warnings`` so a caller can surface them.  See
``docs/research/loop_nullspace/contracts.md`` (NSX-04).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.constants import GRAVITY_FLOAT

logger = logging.getLogger(__name__)

REGISTRY_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "config"
    / "loop_nullspace_registry.json"
)
#: Root translations carry forces (N), not joint torques (N m).
NON_TORQUE_PREFIXES: tuple[str, ...] = ("Translation",)
DIRECTIONS: tuple[str, str] = ("negative", "positive")
SUPPORTED_SEX = "male"


@dataclass(frozen=True)
class CapacitySubject:
    """Subject inputs for capacity scaling.

    Preconditions: ``mass_kg`` and ``stature_m`` positive and finite.
    """

    mass_kg: float
    stature_m: float
    sex: str = SUPPORTED_SEX

    def __post_init__(self) -> None:
        for name in ("mass_kg", "stature_m"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not (
                math.isfinite(value) and value > 0.0
            ):
                raise ValueError(f"{name} must be positive and finite, got {value!r}")

    @classmethod
    def from_spec_subject(cls, subject: Mapping[str, Any]) -> CapacitySubject:
        """Build from the ``subject`` block of a full-body spec JSON.

        The specs are built from the de Leva male table, so ``sex`` is male.
        """
        for key in ("mass_kg", "stature_m"):
            if key not in subject:
                raise ValueError(f"spec subject lacks {key!r}")
        return cls(float(subject["mass_kg"]), float(subject["stature_m"]))


@dataclass(frozen=True)
class CapacityVector:
    """Torque capacities for the actuated coordinates, in coordinate order.

    Postconditions: arrays are read-only, positive and finite;
    ``assumed_slots`` and ``verified_slots`` partition ``actuated_names``.
    ``tau_max_negative_nm[i]`` bounds torque along the negative coordinate
    direction as a magnitude.
    """

    coordinate_order: tuple[str, ...]
    actuated_indices: tuple[int, ...]
    actuated_names: tuple[str, ...]
    tau_max_negative_nm: np.ndarray
    tau_max_positive_nm: np.ndarray
    assumed_slots: tuple[str, ...]
    verified_slots: tuple[str, ...]
    warnings: tuple[str, ...]
    force_velocity: Mapping[str, Mapping[str, Mapping[str, float]]]

    @property
    def has_assumed(self) -> bool:
        """True when any slot rests on an ASSUMED (unverified) value."""
        return bool(self.assumed_slots)


@lru_cache(maxsize=1)
def _entries() -> dict[tuple[str, str], dict[str, Any]]:
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in registry["torque_capacity"]:
        table[(entry["joint_slot"], entry["direction"])] = entry
    return table


def _scaled_tau(entry: Mapping[str, Any], subject: CapacitySubject) -> float:
    rule = entry["scaling_rule"]
    if rule["rule"] == "bw_height_normalized":
        return (
            float(rule["coefficient"])
            * subject.mass_kg
            * GRAVITY_FLOAT
            * subject.stature_m
        )
    return float(entry["tau_max_nm"])


def _validate_inputs(
    coordinate_order: Sequence[str], subject: CapacitySubject
) -> tuple[str, ...]:
    if not isinstance(subject, CapacitySubject):
        raise TypeError(f"subject must be CapacitySubject, got {type(subject)!r}")
    order = tuple(coordinate_order)
    if not order:
        raise ValueError("coordinate_order is empty")
    if len(set(order)) != len(order):
        dup = sorted({n for n in order if order.count(n) > 1})
        raise ValueError(f"coordinate_order has duplicate names: {dup}")
    if subject.sex != SUPPORTED_SEX:
        raise ValueError(
            f"capacity table only covers sex={SUPPORTED_SEX!r}, got {subject.sex!r}"
        )
    return order


def capacity_for(
    coordinate_order: Sequence[str],
    subject: CapacitySubject,
    actuated: Sequence[str] | None = None,
) -> CapacityVector:
    """Return torque capacities for the actuated coordinates.

    ``actuated`` defaults to every coordinate except root translations.

    Preconditions: non-empty, duplicate-free ``coordinate_order``; ``actuated``
    a subset of it; every actuated coordinate has an entry in both directions,
    otherwise ``ValueError`` lists the missing ``slot:direction`` pairs.
    Postconditions: positive finite capacities; ASSUMED slots are listed in
    ``assumed_slots`` and ``warnings`` and logged, never silent.
    """
    order = _validate_inputs(coordinate_order, subject)
    if actuated is None:
        names = tuple(n for n in order if not n.startswith(NON_TORQUE_PREFIXES))
    else:
        names = tuple(n for n in order if n in set(actuated))
        stray = sorted(set(actuated) - set(order))
        if stray:
            raise ValueError(f"actuated coordinates not in coordinate_order: {stray}")
    table = _entries()
    missing = [f"{n}:{d}" for n in names for d in DIRECTIONS if (n, d) not in table]
    if missing:
        raise ValueError(f"no torque capacity for slots: {missing}")
    neg = np.array([_scaled_tau(table[(n, "negative")], subject) for n in names])
    pos = np.array([_scaled_tau(table[(n, "positive")], subject) for n in names])
    for arr in (neg, pos):
        if not np.all(np.isfinite(arr) & (arr > 0.0)):
            raise ValueError("registry produced a non-positive or non-finite capacity")
        arr.setflags(write=False)
    assumed = tuple(
        n for n in names if any(table[(n, d)]["assumed"] for d in DIRECTIONS)
    )
    verified = tuple(n for n in names if n not in assumed)
    warnings = ()
    if assumed:
        warnings = (
            f"{len(assumed)} of {len(names)} torque capacities are ASSUMED "
            f"(UNVERIFIED population priors): {', '.join(assumed)}",
        )
        logger.warning(warnings[0])
    fv = {
        n: {
            d: dict(table[(n, d)]["force_velocity"])
            for d in DIRECTIONS
            if "force_velocity" in table[(n, d)]
        }
        for n in names
        if any("force_velocity" in table[(n, d)] for d in DIRECTIONS)
    }
    return CapacityVector(
        coordinate_order=order,
        actuated_indices=tuple(order.index(n) for n in names),
        actuated_names=names,
        tau_max_negative_nm=neg,
        tau_max_positive_nm=pos,
        assumed_slots=assumed,
        verified_slots=verified,
        warnings=warnings,
        force_velocity=fv,
    )
