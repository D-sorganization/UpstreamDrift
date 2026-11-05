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
    #: range, report-only per HRG-D2 #13173), not from a hard ``coordinate_ranges_deg``
    #: bound and not from a source of its own; a breach is "soft", never a hard miss.
    soft_rom_slots: tuple[str, ...] = ()

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

    @classmethod
    def from_spec(
        cls,
        spec: Mapping[str, Any],
        subject: CapacitySubject,
        velocity_limits: Mapping[str, VelocityLimit],
        allow_unbounded_rom: frozenset[str] = frozenset(),
    ) -> PhysiologicalEnvelope:
        """Build for every actuated (non-translation) coordinate of ``spec``.

        Fails closed (``ValueError``) when a coordinate lacks a torque or
        velocity limit, or a non-root coordinate lacks a declared range and
        the caller did not list it in ``allow_unbounded_rom`` (an unbounded
        joint would make any "no ROM violation" check vacuous).  A coordinate
        with no hard range but a ``soft_limits_deg`` entry takes that range and
        is listed in ``soft_rom_slots`` (hard ranges win when both exist).
        """
        cap = capacity_for(tuple(spec["coordinate_order"]), subject)
        hard = spec["coordinate_ranges_deg"]
        soft = {
            n: v for n, v in spec.get("soft_limits_deg", {}).items() if n not in hard
        }
        ranges = {**soft, **hard}  # a hard bound always wins over a soft one
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
