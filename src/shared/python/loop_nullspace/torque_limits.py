"""Torque-limit provider for capability ellipsoids (CAP-2, #14041).

Adapts the anthropometry-scaled, directional ``CapacityVector`` (NSX-04) into
``(B, tau_lo, tau_hi)`` for a model's DoF ordering.  Limits are posture-
independent population priors; assumed/verified slot counts travel with them so
a glyph can label itself.  The uniform 100 N m ``JointInfo.torque_limit``
default in ``control_interface`` is a placeholder and is refused here.
Posture-dependent muscle moments are deferred behind HX6 (#13763).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.capacity import (
    DIRECTIONS,
    CapacitySubject,
    CapacityVector,
    _entries,
    capacity_for,
)

REGISTRY_SOURCE_ID = "loop_nullspace_registry"
PLACEHOLDER_SOURCES = frozenset({"control_interface", "placeholder"})
CAPACITY_HANDS: tuple[str, ...] = ("lead", "trail")
CAPACITY_MODES: tuple[str, ...] = ("constrained", "unconstrained")


@dataclass(frozen=True)
class TorqueLimits:
    """Actuated-coordinate selection and torque bounds.

    ``B`` is ``(nv, nu)``; column ``j`` selects the DoF actuated by limit ``j``.
    Postconditions (checked): arrays read-only and finite, ``tau_lo < 0 <
    tau_hi``, no placeholder source, no uniform constant limit.
    """

    B: np.ndarray
    tau_lo: np.ndarray
    tau_hi: np.ndarray
    source_id: str
    assumed_slots: int
    verified_slots: int
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source_id in PLACEHOLDER_SOURCES:
            raise ValueError(
                f"source {self.source_id!r} is a placeholder, not a torque capacity"
            )
        if self.B.ndim != 2 or self.B.shape[1] != self.tau_lo.size:
            raise ValueError(f"B {self.B.shape} does not match nu={self.tau_lo.size}")
        if self.tau_lo.shape != self.tau_hi.shape or self.tau_lo.ndim != 1:
            raise ValueError("tau_lo and tau_hi must be 1-D arrays of equal length")
        for arr in (self.B, self.tau_lo, self.tau_hi):
            if not np.all(np.isfinite(arr)):
                raise ValueError("torque limits and B must be finite")
        if not (np.all(self.tau_lo < 0.0) and np.all(self.tau_hi > 0.0)):
            raise ValueError("require tau_lo < 0 < tau_hi for every actuated DoF")
        both = np.concatenate([-self.tau_lo, self.tau_hi])
        if both.size > 1 and np.ptp(both) == 0.0:
            raise ValueError(
                f"uniform torque limit {both[0]:g} N m is a placeholder, refused"
            )
        for arr in (self.B, self.tau_lo, self.tau_hi):
            arr.setflags(write=False)


def _resolve(
    source: Mapping[str, Any] | CapacitySubject, dof_names: tuple[str, ...]
) -> CapacityVector:
    """Capacity for ``source``: a full-body spec mapping or a bare subject."""
    if isinstance(source, CapacitySubject):
        table = _entries()
        names = tuple(n for n in dof_names if all((n, d) in table for d in DIRECTIONS))
        return capacity_for(dof_names, source, actuated=names)
    if "subject" not in source or "coordinate_order" not in source:
        raise ValueError("spec must have 'subject' and 'coordinate_order'")
    subject = CapacitySubject.from_spec_subject(source["subject"])
    return capacity_for(tuple(source["coordinate_order"]), subject)


def torque_limits_for(
    spec_or_subject: Mapping[str, Any] | CapacitySubject,
    dof_names: Sequence[str],
) -> TorqueLimits:
    """Return ``TorqueLimits`` mapped onto the caller's ``dof_names``.

    A spec mapping actuates its non-translation ``coordinate_order``; a
    ``CapacitySubject`` actuates the ``dof_names`` that have registry entries.
    DoFs without a capacity (root, club, armature-only) get zero rows in ``B``.
    Raises ``ValueError`` for duplicate ``dof_names`` or any actuated coordinate
    missing from them.
    """
    dofs = tuple(dof_names)
    if len(set(dofs)) != len(dofs):
        raise ValueError(f"dof_names has duplicate names: {sorted(set(dofs))}")
    vec = _resolve(spec_or_subject, dofs)
    absent = [n for n in vec.actuated_names if n not in dofs]
    if absent:
        raise ValueError(f"actuated coordinates missing from dof_names: {absent}")
    B = np.zeros((len(dofs), len(vec.actuated_names)))
    for col, name in enumerate(vec.actuated_names):
        B[dofs.index(name), col] = 1.0
    return TorqueLimits(
        B=B,
        tau_lo=-vec.tau_max_negative_nm,
        tau_hi=vec.tau_max_positive_nm.copy(),
        source_id=REGISTRY_SOURCE_ID,
        assumed_slots=len(vec.assumed_slots),
        verified_slots=len(vec.verified_slots),
        warnings=vec.warnings,
    )


def resolve_torque_limits(
    source: TorqueLimits | CapacitySubject | Mapping[str, Any] | None,
    spec: Mapping[str, Any],
    coordinate_order: Sequence[str],
) -> TorqueLimits | None:
    """Resolve optional torque limits from an explicit object, subject, spec, or mapping."""
    if isinstance(source, TorqueLimits):
        return source
    if isinstance(source, CapacitySubject):
        return torque_limits_for(source, coordinate_order)
    if isinstance(source, Mapping):
        return torque_limits_for(source, coordinate_order)
    if "subject" in spec:
        try:
            return torque_limits_for(spec, coordinate_order)
        except Exception:
            return None
    return None


def validate_capacity_hand_mode(
    hand: str,
    mode: str,
    *,
    hands: Sequence[str] = CAPACITY_HANDS,
    modes: Sequence[str] = CAPACITY_MODES,
) -> None:
    """Validate hand and closure mode labels for capacity ellipsoid tasks."""
    if mode not in modes:
        raise ValueError(f"mode must be one of {modes}, got {mode!r}")
    if hand not in hands:
        raise ValueError(f"hand must be one of {hands}, got {hand!r}")


def effective_torque_limits_for_capacity(
    torque_limits: TorqueLimits | None,
    default_limits: TorqueLimits | None,
    *,
    label: str,
) -> TorqueLimits:
    """Return explicit or default torque limits, or raise if neither is set."""
    limits = torque_limits or default_limits
    if limits is None:
        raise ValueError(f"{label} capacity requires torque limits")
    return limits
