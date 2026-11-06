"""``PhysiologicalEnvelope``: per-joint torque, range and velocity limits.

Sources (no number is invented here):

* torque: the NSX-04 capacity table (#12553) via
  :func:`src.shared.python.loop_nullspace.capacity.capacity_for`, scaled to the
  subject; unverified table slots are carried through as ASSUMED;
* range of motion: the spec's ``coordinate_ranges_deg`` (coordinates without a
  declared range are listed in ``unbounded_rom_slots``, not given a made-up one);
* angular velocity: a cited YAML (``src/config/impact_envelope_limits.yaml``).

The envelope **fails closed**: an actuated coordinate without a limit in every
category raises ``ValueError`` naming the coordinate.  A binding torque is
reported by joint name (:meth:`binding_joints`), never relaxed silently.

Design by Contract:
    Preconditions: strictly ordered limits (ROM lo < hi, torque and
    velocity finite and > 0); torque vectors match the actuated-joint count.
    Postconditions: arrays are read-only; ``utilisation`` >= 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.shared.python.contracts import ensure, require
from src.shared.python.loop_nullspace.capacity import CapacitySubject, capacity_for

DEFAULT_LIMITS_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "impact_envelope_limits.yaml"
)


@dataclass(frozen=True)
class VelocityLimit:
    """One cited angular-velocity limit (deg/s) for a coordinate."""

    value_deg_s: float
    verification_status: str


#: Allowed registry statuses; only VERIFIED is exempt from assumption-limited reporting.
VERIFICATION_STATUSES: frozenset[str] = frozenset(
    {"VERIFIED", "SOURCED_WITH_ASSUMED_MARGIN", "UNVERIFIED"}
)


def load_limits(path: Path | str = DEFAULT_LIMITS_PATH) -> dict[str, VelocityLimit]:
    """Parse the cited YAML into ``{coordinate: VelocityLimit}``.

    Preconditions: every entry has coordinates, a positive finite value, a
    citation mapping and a ``verification_status``.
    """
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("velocity_limits_deg_s") if isinstance(raw, Mapping) else None
    require(bool(entries), "limits file lacks velocity_limits_deg_s", str(path))
    assert entries is not None  # narrowing for mypy
    out: dict[str, VelocityLimit] = {}
    for e in entries:
        require(
            isinstance(e.get("citation"), Mapping) and "verification_status" in e,
            "each limit needs a citation and verification_status",
            e.get("coordinates"),
        )
        require(
            e["verification_status"] in VERIFICATION_STATUSES,
            "unknown verification_status",
            e["verification_status"],
        )
        value = float(e["value"])
        require(
            math.isfinite(value) and value > 0.0,
            "velocity limit must be positive",
            value,
        )
        for name in e["coordinates"]:
            require(name not in out, "duplicate velocity limit", name)
            out[name] = VelocityLimit(value, str(e["verification_status"]))
    return out


@dataclass(frozen=True)
class DeclaredSide:
    """One bound of a declared range: degrees, source text, verification."""

    value_deg: float
    verification_status: str
    citation: str


@dataclass(frozen=True)
class DeclaredRom:
    """A declared joint range with a source per side (lower < upper)."""

    lower: DeclaredSide
    upper: DeclaredSide


def _citation_text(side: Mapping[str, Any]) -> str:
    cite = side["citation"]
    parts = [str(cite.get(k)) for k in ("reference", "doi", "note") if cite.get(k)]
    return "; ".join(parts)


def _declared_side(coordinate: str, side: Mapping[str, Any]) -> DeclaredSide:
    require(
        isinstance(side.get("citation"), Mapping) and "verification_status" in side,
        "each declared ROM side needs a citation and verification_status",
        coordinate,
    )
    value = float(side["value"])
    require(math.isfinite(value), "declared ROM must be finite", (coordinate, value))
    status = str(side["verification_status"])
    if status in ("VERIFIED", "SOURCED_WITH_ASSUMED_MARGIN"):
        cite = side["citation"]
        require(
            bool(cite.get("reference") and cite.get("doi")),
            "a sourced ROM side needs a reference and a DOI",
            coordinate,
        )
    return DeclaredSide(value, status, _citation_text(side))


def load_declared_rom(
    path: Path | str = DEFAULT_LIMITS_PATH,
) -> dict[str, DeclaredRom]:
    """Parse ``declared_rom_deg`` into ``{coordinate: DeclaredRom}``.

    Preconditions: every entry names one coordinate and gives ``lower`` and
    ``upper`` sides, each with a value, a citation and a ``verification_status``;
    ``lower < upper``.  An absent section yields ``{}`` (callers then fail closed).
    """
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("declared_rom_deg") if isinstance(raw, Mapping) else None
    out: dict[str, DeclaredRom] = {}
    for e in entries or []:
        name = e["coordinate"]
        require(name not in out, "duplicate declared ROM", name)
        lower = _declared_side(name, e["lower"])
        upper = _declared_side(name, e["upper"])
        require(lower.value_deg < upper.value_deg, "declared ROM lower < upper", name)
        out[name] = DeclaredRom(lower, upper)
    return out


def is_assumed(
    joint: str, kind: str, side: str, assumed: frozenset[tuple[str, str]]
) -> bool:
    """Is the ``(joint, kind, side)`` limit in an ``assumed_limits()`` set?

    A whole-joint ROM entry ``(joint, "ROM")`` covers both sides; a per-side
    entry ``(joint, "ROM:lower")`` covers one.  One predicate for every reporter.
    """
    if (joint, kind) in assumed:
        return True
    return kind == "ROM" and bool(side) and (joint, f"ROM:{side}") in assumed


#: Root (pelvis-in-world) rotations: free by definition, so the one explicit
#: exemption from the fail-closed ROM rule.  Every other coordinate needs a
#: declared range or an explicit ``allow_unbounded_rom`` entry.
ROOT_COORDINATES: frozenset[str] = frozenset({"HipInputX", "HipInputY", "HipInputZ"})


@dataclass(frozen=True, eq=False)
class PhysiologicalEnvelope:
    """Per-actuated-joint limits, all arrays aligned with ``names``."""

    names: tuple[str, ...]
    tau_max_negative_nm: np.ndarray
    tau_max_positive_nm: np.ndarray
    rom_deg: np.ndarray  # (n, 2) lower, upper
    vel_max_deg_s: np.ndarray
    assumed_torque_slots: tuple[str, ...]
    assumed_velocity_slots: tuple[str, ...]
    #: Coordinates the spec declares no range for; their ROM is +-inf (never invented).
    unbounded_rom_slots: tuple[str, ...] = ()
    #: Coordinates whose ROM comes from the spec's ``soft_limits_deg`` (a design
    #: range, HRG-D2 #13173), not from a hard ``coordinate_ranges_deg`` bound and
    #: not from a source of its own.  A breach still rejects the rung (D29) but is
    #: reported as assumption-limited (D32), never as probable miscalibration.
    soft_rom_slots: tuple[str, ...] = ()
    #: ``(coordinate, "lower"|"upper")`` bounds from ``declared_rom_deg`` whose
    #: ``verification_status`` is not VERIFIED (no source read in full text).
    assumed_rom_sides: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        n = len(self.names)
        require(n > 0, "envelope needs at least one joint")
        require(self.rom_deg.shape == (n, 2), "rom_deg shape mismatch")
        require(
            bool(
                not np.any(np.isnan(self.rom_deg))
                and np.all(self.rom_deg[:, 0] < self.rom_deg[:, 1])
            ),
            "ROM must be ordered (lower < upper), never NaN",
        )
        for label, arr in (
            ("tau_max_negative_nm", self.tau_max_negative_nm),
            ("tau_max_positive_nm", self.tau_max_positive_nm),
            ("vel_max_deg_s", self.vel_max_deg_s),
        ):
            require(
                arr.shape == (n,) and bool(np.all(np.isfinite(arr) & (arr > 0.0))),
                f"{label} must be positive and finite per joint",
            )
            arr.setflags(write=False)
        self.rom_deg.setflags(write=False)
        require(
            set(self.soft_rom_slots) <= set(self.names)
            and not set(self.soft_rom_slots) & set(self.unbounded_rom_slots),
            "soft_rom_slots must be known joints with a declared (soft) range",
            self.soft_rom_slots,
        )
        require(
            all(
                j in self.names and s in ("lower", "upper")
                for j, s in self.assumed_rom_sides
            ),
            "assumed_rom_sides must be (known joint, 'lower'|'upper')",
            self.assumed_rom_sides,
        )

    def limit_is_assumed(self, joint: str, kind: str, side: str = "") -> bool:
        """True when ``joint``'s ``kind`` limit (on ``side``) is not a cited value.

        ``kind`` is ``"ROM"`` (a soft_limits_deg design range, or a declared
        bound without a read source) or ``"velocity"`` (an UNVERIFIED prior).
        ``side`` (``"lower"``/``"upper"``) matters for ROM declared per side;
        without it only a whole-joint (soft) ROM reads as assumed, never a guess.
        """
        require(kind in ("ROM", "velocity"), "kind must be 'ROM' or 'velocity'", kind)
        return is_assumed(joint, kind, side, self.assumed_limits())

    def assumed_limits(self) -> frozenset[tuple[str, str]]:
        """Every assumed limit as ``(joint, kind)`` or ``(joint, "ROM:<side>")``."""
        return frozenset(
            [(j, "ROM") for j in self.soft_rom_slots]
            + [(j, f"ROM:{s}") for j, s in self.assumed_rom_sides]
            + [(j, "velocity") for j in self.assumed_velocity_slots]
        )

    @classmethod
    def from_spec(
        cls,
        spec: Mapping[str, Any],
        subject: CapacitySubject,
        velocity_limits: Mapping[str, VelocityLimit],
        allow_unbounded_rom: frozenset[str] = frozenset(),
        declared_rom: Mapping[str, DeclaredRom] | None = None,
    ) -> PhysiologicalEnvelope:
        """Build for every actuated (non-translation) coordinate of ``spec``.

        Fails closed (``ValueError``) when a coordinate lacks a torque or
        velocity limit, or a non-root coordinate lacks a declared range and
        the caller did not list it in ``allow_unbounded_rom`` (an unbounded
        joint would make any "no ROM violation" check vacuous).  A coordinate
        with no hard range but a ``soft_limits_deg`` entry takes that range and
        is listed in ``soft_rom_slots`` (hard ranges win when both exist).  A soft
        range gates acceptance of a rung, never IK or an engine limit.  A coordinate
        with neither takes ``declared_rom_deg`` from the limits YAML (a source and
        status per side; pass ``declared_rom={}`` to fail closed instead).
        """
        cap = capacity_for(tuple(spec["coordinate_order"]), subject)
        hard = spec["coordinate_ranges_deg"]
        soft = {
            n: v for n, v in spec.get("soft_limits_deg", {}).items() if n not in hard
        }
        pool = load_declared_rom() if declared_rom is None else declared_rom
        declared = {n: d for n, d in pool.items() if n not in hard and n not in soft}
        declared_ranges = {
            n: (d.lower.value_deg, d.upper.value_deg) for n, d in declared.items()
        }
        # a hard bound wins over a soft one, which wins over a declared one
        ranges = {**declared_ranges, **soft, **hard}
        missing = [
            f"{n}: {kind}"
            for n in cap.actuated_names
            for kind, table in (("velocity", velocity_limits),)
            if n not in table
        ]
        if missing:
            raise ValueError(f"no envelope limit for: {missing}")
        names = cap.actuated_names
        no_rom = [
            n
            for n in names
            if n not in ranges
            and n not in ROOT_COORDINATES
            and n not in allow_unbounded_rom
        ]
        if no_rom:
            raise ValueError(
                f"no declared ROM for: {no_rom}; pass allow_unbounded_rom to accept"
            )
        return cls(
            names=names,
            tau_max_negative_nm=np.array(cap.tau_max_negative_nm, dtype=float),
            tau_max_positive_nm=np.array(cap.tau_max_positive_nm, dtype=float),
            rom_deg=np.array(
                [ranges.get(n, (-np.inf, np.inf)) for n in names], dtype=float
            ),
            unbounded_rom_slots=tuple(n for n in names if n not in ranges),
            soft_rom_slots=tuple(n for n in names if n in soft),
            assumed_rom_sides=tuple(
                (n, s)
                for n in names
                if n in declared
                for s, b in (("lower", declared[n].lower), ("upper", declared[n].upper))
                if b.verification_status != "VERIFIED"
            ),
            vel_max_deg_s=np.array(
                [velocity_limits[n].value_deg_s for n in names], dtype=float
            ),
            assumed_torque_slots=cap.assumed_slots,
            assumed_velocity_slots=tuple(
                n for n in names if velocity_limits[n].verification_status != "VERIFIED"
            ),
        )

    def utilisation(self, tau_nm: np.ndarray) -> np.ndarray:
        """``|tau| / capacity`` in the direction of ``tau`` (1.0 = at the limit)."""
        tau = np.asarray(tau_nm, dtype=float)
        require(tau.shape == (len(self.names),), "torque length mismatch", tau.shape)
        cap = np.where(tau >= 0.0, self.tau_max_positive_nm, self.tau_max_negative_nm)
        u = np.abs(tau) / cap
        ensure(bool(np.all(u >= 0.0)), "utilisation must be non-negative")
        return u

    def torque_excess(self, tau_nm: np.ndarray) -> np.ndarray:
        """Torque magnitude beyond capacity per joint (N m, 0 inside)."""
        tau = np.asarray(tau_nm, dtype=float)
        cap = np.where(tau >= 0.0, self.tau_max_positive_nm, self.tau_max_negative_nm)
        return np.maximum(0.0, np.abs(tau) - cap)

    def binding_joints(
        self, tau_nm: np.ndarray, threshold: float = 0.98
    ) -> tuple[str, ...]:
        """Joints whose torque is at or above ``threshold`` of capacity."""
        require(0.0 < threshold <= 1.0, "threshold must be in (0, 1]", threshold)
        u = self.utilisation(tau_nm)
        return tuple(n for n, ui in zip(self.names, u, strict=True) if ui >= threshold)
