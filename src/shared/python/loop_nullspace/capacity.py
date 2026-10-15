"""NSX-04: Canonical Joint Torque-Capacity Table Loader and Scaling Contract.

Loads ``torque_capacity[]`` from ``src/config/loop_nullspace_registry.json``,
enforces Design-by-Contract coverage over every actuated coordinate in a
snapshot's ``coordinate_order``, applies anthropometric scaling, and surfaces
explicit warnings for any ``ASSUMED`` / ``UNVERIFIED`` slot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from src.shared.python.loop_nullspace.registry import (
    DEFAULT_REGISTRY_PATH,
    SourceRecord,
    VerificationStatus,
    load_registry,
)

CapacityStatus = Literal["VERIFIED", "ASSUMED", "UNVERIFIED"]
REFERENCE_MASS_KG = 75.0
REFERENCE_HEIGHT_M = 1.78


@dataclass(frozen=True)
class CapacitySlotEntry:
    """Single canonical coordinate torque-capacity entry."""

    coordinate: str
    tau_max_nm: float
    tau_min_nm: float
    population_descriptor: str
    scaling_rule: Mapping[str, Any]
    source: SourceRecord
    status: CapacityStatus
    verification_status: VerificationStatus
    force_velocity_params: Mapping[str, Any] | None = None
    warning: str | None = None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> CapacitySlotEntry:
        """Construct and validate a ``CapacitySlotEntry`` from a registry dict."""
        coord = raw.get("coordinate")
        if not isinstance(coord, str) or not coord.strip():
            raise ValueError("Capacity slot missing non-empty 'coordinate'")

        tau_max = float(raw["tau_max_nm"])
        tau_min = float(raw.get("tau_min_nm", -abs(tau_max)))
        if tau_max <= 0.0:
            raise ValueError(f"Slot {coord!r}: tau_max_nm must be > 0, got {tau_max}")
        if tau_min >= 0.0:
            raise ValueError(f"Slot {coord!r}: tau_min_nm must be < 0, got {tau_min}")

        scaling = raw.get("scaling_rule")
        if not isinstance(scaling, Mapping):
            raise ValueError(f"Slot {coord!r} missing 'scaling_rule' mapping")

        src = SourceRecord.from_raw(raw["source"], {}, f"Slot {coord!r}")
        ver_status = raw.get("verification_status", src.verification_status)
        if ver_status not in {"VERIFIED", "UNVERIFIED", "REMOVED"}:
            raise ValueError(
                f"Slot {coord!r} invalid verification_status {ver_status!r}"
            )

        status_raw = str(raw.get("status", ver_status))
        if status_raw not in {"VERIFIED", "ASSUMED", "UNVERIFIED"}:
            raise ValueError(f"Slot {coord!r} invalid status {status_raw!r}")

        warning = raw.get("warning")
        if status_raw == "ASSUMED" or ver_status == "UNVERIFIED":
            if ver_status != "UNVERIFIED":
                raise ValueError(
                    f"Slot {coord!r} marked ASSUMED must have verification_status='UNVERIFIED'"
                )
            if not isinstance(warning, str) or not warning.strip():
                raise ValueError(
                    f"Slot {coord!r} marked ASSUMED/UNVERIFIED requires a non-empty 'warning'"
                )

        fv_raw = raw.get("force_velocity_params")
        fv_map = MappingProxyType(dict(fv_raw)) if isinstance(fv_raw, Mapping) else None

        return cls(
            coordinate=coord.strip(),
            tau_max_nm=tau_max,
            tau_min_nm=tau_min,
            population_descriptor=str(raw.get("population_descriptor", "")),
            scaling_rule=MappingProxyType(dict(scaling)),
            source=src,
            status=status_raw,  # type: ignore[arg-type]
            verification_status=ver_status,  # type: ignore[arg-type]
            force_velocity_params=fv_map,
            warning=warning.strip() if isinstance(warning, str) else None,
        )


@dataclass(frozen=True)
class CapacityVector:
    """Scaled joint torque-capacity bounds and provenance over a coordinate order."""

    coordinate_order: tuple[str, ...]
    tau_min_nm: np.ndarray
    tau_max_nm: np.ndarray
    entries: tuple[CapacitySlotEntry, ...]
    verification_status: tuple[CapacityStatus, ...]
    sources: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        self.tau_min_nm.setflags(write=False)
        self.tau_max_nm.setflags(write=False)
        if self.tau_max_nm.shape != (len(self.coordinate_order),):
            raise ValueError("tau_max_nm shape mismatch with coordinate_order")
        if self.tau_min_nm.shape != (len(self.coordinate_order),):
            raise ValueError("tau_min_nm shape mismatch with coordinate_order")
        if not np.all(self.tau_max_nm > 0.0):
            raise ValueError("Postcondition violated: all tau_max_nm must be > 0")
        if not np.all(self.tau_min_nm < 0.0):
            raise ValueError("Postcondition violated: all tau_min_nm must be < 0")


def _extract_subject_dimensions(
    subject: Mapping[str, Any] | float | None,
    subject_height_m: float | None,
) -> tuple[float, float]:
    """Extract ``(mass_kg, height_m)`` from subject mapping or scalar arguments."""
    if subject is None:
        mass_kg = REFERENCE_MASS_KG
        height_m = (
            float(subject_height_m)
            if subject_height_m is not None
            else REFERENCE_HEIGHT_M
        )
    elif isinstance(subject, int | float) and not isinstance(subject, bool):
        mass_kg = float(subject)
        height_m = (
            float(subject_height_m)
            if subject_height_m is not None
            else REFERENCE_HEIGHT_M
        )
    elif isinstance(subject, Mapping):
        mass_kg = float(subject.get("mass_kg", REFERENCE_MASS_KG))
        raw_height = subject.get("height_m", subject.get("stature_m", subject_height_m))
        height_m = float(raw_height) if raw_height is not None else REFERENCE_HEIGHT_M
    else:
        raise TypeError(
            f"subject must be a mapping, float mass_kg, or None; got {type(subject).__name__}"
        )

    if mass_kg <= 0.0 or not np.isfinite(mass_kg):
        raise ValueError(f"subject mass_kg must be positive and finite, got {mass_kg}")
    if height_m <= 0.0 or not np.isfinite(height_m):
        raise ValueError(
            f"subject height_m must be positive and finite, got {height_m}"
        )
    return mass_kg, height_m


def _compute_scale_factor(
    scaling_rule: Mapping[str, Any], mass_kg: float, height_m: float
) -> float:
    """Compute anthropometric torque scale factor from ``scaling_rule``."""
    ref_mass = float(scaling_rule.get("reference_mass_kg", REFERENCE_MASS_KG))
    ref_height = float(scaling_rule.get("reference_height_m", REFERENCE_HEIGHT_M))
    rule_name = str(scaling_rule.get("rule", "mass_proportional"))
    if rule_name == "mass_height_proportional":
        return (mass_kg * height_m) / (ref_mass * ref_height)
    return mass_kg / ref_mass


def capacity_for(
    snapshot_coordinate_order: Sequence[str],
    subject: Mapping[str, Any] | float | None = None,
    subject_height_m: float | None = None,
    *,
    unactuated_coordinates: Sequence[str] = (),
    registry_path: Path | str | None = None,
) -> CapacityVector:
    """Return a ``CapacityVector`` covering all actuated coordinates in ``snapshot_coordinate_order``.

    Preconditions:
        - ``snapshot_coordinate_order`` is a non-empty sequence of coordinate names.
        - Every actuated coordinate (not in ``unactuated_coordinates``) exists in
          the registry's ``torque_capacity[]`` table; otherwise raises ``ValueError``
          listing all missing slots.
        - ``subject`` mass and height are positive finite numbers.
    Postconditions:
        - ``tau_max_nm > 0`` and ``tau_min_nm < 0`` for every slot.
        - Any slot with ``status == "ASSUMED"`` or ``verification_status == "UNVERIFIED"``
          produces an explicit warning in ``CapacityVector.warnings``.
    """
    if not isinstance(snapshot_coordinate_order, Sequence) or isinstance(
        snapshot_coordinate_order, str | bytes
    ):
        raise TypeError("snapshot_coordinate_order must be a sequence of strings")

    unactuated_set = set(unactuated_coordinates)
    actuated_order = tuple(
        str(c) for c in snapshot_coordinate_order if c not in unactuated_set
    )
    if not actuated_order:
        raise ValueError("At least one actuated coordinate is required")

    mass_kg, height_m = _extract_subject_dimensions(subject, subject_height_m)

    reg = load_registry(registry_path or DEFAULT_REGISTRY_PATH)
    table_by_coord: dict[str, CapacitySlotEntry] = {}
    for raw_slot in reg.raw_torque_capacity:
        entry = CapacitySlotEntry.from_raw(raw_slot)
        table_by_coord[entry.coordinate] = entry

    missing = [c for c in actuated_order if c not in table_by_coord]
    if missing:
        raise ValueError(
            f"Missing torque_capacity entries for actuated coordinate slots: {missing}"
        )

    scaled_entries: list[CapacitySlotEntry] = []
    tau_min_list: list[float] = []
    tau_max_list: list[float] = []
    statuses: list[CapacityStatus] = []
    sources: list[str] = []
    warnings: list[str] = []

    for coord in actuated_order:
        base = table_by_coord[coord]
        scale = _compute_scale_factor(base.scaling_rule, mass_kg, height_m)
        s_max = base.tau_max_nm * scale
        s_min = base.tau_min_nm * scale
        scaled_entry = CapacitySlotEntry(
            coordinate=base.coordinate,
            tau_max_nm=s_max,
            tau_min_nm=s_min,
            population_descriptor=base.population_descriptor,
            scaling_rule=base.scaling_rule,
            source=base.source,
            status=base.status,
            verification_status=base.verification_status,
            force_velocity_params=base.force_velocity_params,
            warning=base.warning,
        )
        scaled_entries.append(scaled_entry)
        tau_max_list.append(s_max)
        tau_min_list.append(s_min)
        statuses.append(base.verification_status)  # type: ignore[arg-type]
        sources.append(base.source.citation)
        if base.status == "ASSUMED" or base.verification_status == "UNVERIFIED":
            warn_msg = (
                base.warning
                or f"Coordinate {coord!r} uses an ASSUMED/UNVERIFIED torque capacity prior."
            )
            warnings.append(f"[{coord}] {warn_msg}")

    return CapacityVector(
        coordinate_order=actuated_order,
        tau_min_nm=np.asarray(tau_min_list, dtype=float),
        tau_max_nm=np.asarray(tau_max_list, dtype=float),
        entries=tuple(scaled_entries),
        verification_status=tuple(statuses),
        sources=tuple(sources),
        warnings=tuple(warnings),
    )


__all__ = [
    "CapacitySlotEntry",
    "CapacityStatus",
    "CapacityVector",
    "REFERENCE_HEIGHT_M",
    "REFERENCE_MASS_KG",
    "capacity_for",
]
