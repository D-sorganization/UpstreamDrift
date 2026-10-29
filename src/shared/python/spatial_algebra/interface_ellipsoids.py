"""Engine-neutral contracts for interface (hand-club) ellipsoids.

Radii are always the square roots of the shape-matrix eigenvalues (semi-axes),
as in ``operational_space.velocity_ellipsoid`` / ``force_ellipsoid``.  Capacity
kinds describe a torque *box*, whose attainable set is a zonotope; the
ellipsoid is only an inner approximation of it.  Epic #14038, CAP-0 (#14039).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

PARITY_LEDGER_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "interface_ellipsoid_parity.json"
)
PARITY_STATUSES = ("parity", "gap", "exempt")
CLOSURES = ("constrained", "unconstrained")

# value -> (radius unit, HUD label, is_capacity, requires_torque_limits)
_KIND_INFO: dict[str, tuple[str, str, bool, bool]] = {
    "mobility": ("kg^-1/2", "Mobility", False, False),
    "effective-mass": ("kg^1/2", "Effective mass", False, False),
    "acceleration-capacity": ("m/s^2", "Acceleration capacity", True, True),
    "force-capacity": ("N", "Force capacity", True, True),
}


class InterfaceEllipsoidKind(str, Enum):
    """Kinds of interface ellipsoid; radii are sqrt of shape eigenvalues."""

    MOBILITY = "mobility"
    EFFECTIVE_MASS = "effective-mass"
    ACCELERATION_CAPACITY = "acceleration-capacity"
    FORCE_CAPACITY = "force-capacity"

    @property
    def radius_unit(self) -> str:
        return _KIND_INFO[self.value][0]

    @property
    def label(self) -> str:
        return _KIND_INFO[self.value][1]

    @property
    def is_capacity(self) -> bool:
        return _KIND_INFO[self.value][2]

    @property
    def requires_torque_limits(self) -> bool:
        return _KIND_INFO[self.value][3]

    @property
    def inner_approximation(self) -> bool:
        """Capacity ellipsoids inscribe the true zonotope (torque box)."""
        return self.is_capacity


@dataclass(frozen=True)
class EllipsoidProvenance:
    """Where an ellipsoid came from; capacity kinds must name a limit source."""

    engine: str
    topology: str
    closure: str
    torque_limit_source: str | None
    assumed_slots: int
    verified_slots: int
    kind: InterfaceEllipsoidKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, InterfaceEllipsoidKind):
            raise TypeError("kind must be an InterfaceEllipsoidKind")
        for name in ("engine", "topology"):
            val = getattr(self, name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.closure not in CLOSURES:
            raise ValueError(f"closure must be one of {CLOSURES}, got {self.closure!r}")
        for name in ("assumed_slots", "verified_slots"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"{name} must be an int")
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")
        src = self.torque_limit_source
        if src is not None and not isinstance(src, str):
            raise TypeError("torque_limit_source must be a str or None")
        if self.kind.requires_torque_limits and not (src and src.strip()):
            raise ValueError(f"{self.kind.value} requires a torque_limit_source")

    def label(self) -> str:
        """Short HUD string, e.g. ``Force capacity [mujoco, inner approx., 3 assumed]``."""
        parts = [self.engine, self.closure]
        if self.kind.inner_approximation:
            parts.append("inner approx.")
        if self.kind.requires_torque_limits:
            parts.append(f"{self.assumed_slots} assumed")
        return f"{self.kind.label} [{', '.join(parts)}]"


def load_parity_ledger(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load and validate the engine-capability parity ledger.

    Returns ``{engine: {kind: record}}``.  Raises ``ValueError`` on an unknown
    kind or status, a ``gap`` without an int issue, an ``exempt`` without a
    reason, or an engine that omits a kind.
    """
    data = json.loads(Path(path or PARITY_LEDGER_PATH).read_text(encoding="utf-8"))
    engines = data.get("engines")
    if not isinstance(engines, dict) or not engines:
        raise ValueError("ledger must contain a non-empty 'engines' object")
    kinds = {k.value for k in InterfaceEllipsoidKind}
    for engine, entry in engines.items():
        unknown = set(entry) - kinds
        if unknown:
            raise ValueError(f"{engine}: unknown kind(s) {sorted(unknown)}")
        if set(entry) != kinds:
            raise ValueError(f"{engine}: missing kind(s) {sorted(kinds - set(entry))}")
        for kind, rec in entry.items():
            status = rec.get("status")
            if status not in PARITY_STATUSES:
                raise ValueError(f"{engine}/{kind}: invalid status {status!r}")
            if status == "gap" and not isinstance(rec.get("issue"), int):
                raise ValueError(f"{engine}/{kind}: gap needs an integer issue")
            if status == "exempt" and not str(rec.get("reason", "")).strip():
                raise ValueError(f"{engine}/{kind}: exempt needs a reason")
    return engines
