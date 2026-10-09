"""Simscape <-> spec URDF exchange diff (#11569 task 3).

``scripts/matlab/export_simscape_urdf_exchange.m`` (R2025b) writes a receipt
with two inventories read by the same MATLAB routine
(``simscape_model_inventory.m``): the canonical ``GolfSwing3D_Kinetic`` and an
``smimport`` of the spec URDF built by the #9965 exporter. ``smexport`` does
not exist in R2025b, so the canonical tree comes from its blocks.

This module diffs that receipt against the URDF on disk:

* round trip: does the ``smimport`` model carry the URDF's coordinates and
  link masses;
* coordinates: canonical Simscape versus spec URDF;
* masses per anatomical group (:data:`SEGMENT_GROUPS`), with Simscape bodies
  whose mass could not be evaluated listed, never counted as zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DefusedET

__all__ = [
    "SCHEMA",
    "SEGMENT_GROUPS",
    "ExchangeReport",
    "GroupMass",
    "UrdfInventory",
    "exchange_report",
    "lf_sha256",
    "load_exchange_receipt",
    "urdf_inventory",
]

SCHEMA = "simscape-urdf-exchange/v1"

#: group -> (canonical Simscape body block names, spec URDF link names).
SEGMENT_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "trunk": (
        ("LowerTorso", "UpperTorsoBase", "UpperTorsoTop"),
        ("pelvis", "lumbar1", "lumbar2", "lumbar3", "thorax1", "thorax2", "thorax3"),
    ),
    "head_neck": (("Head", "Neck"), ("head",)),
    "shoulder_girdle": (("HubtoLS", "HubtoRS"), ("scapula_left", "scapula_right")),
    "upper_arms": (("LUpperArm", "RUpperArm"), ("upper_arm_left", "upper_arm_right")),
    "forearms": (
        ("LUpperForearm", "LLowerForearm", "RUpperForearm", "RLowerForearm"),
        ("forearm_left", "forearm_right"),
    ),
    "hands": (
        ("LHand", "RHand", "LHandStandoff", "RHandStandoff"),
        ("hand_left", "hand_right", "fingers_left", "fingers_right"),
    ),
    "club": (
        ("Clubhead", "Rigid Shaft", '1.5" Bottom', '1.5" Top', '2.5" Butt'),
        ("club_shaft", "club_head"),
    ),
    "legs": (
        (),
        tuple(
            f"{side}_{seg}"
            for side in ("right", "left")
            for seg in ("thigh", "shank", "foot", "toes")
        ),
    ),
}

_URDF_DOF = {"revolute": 1, "continuous": 1, "prismatic": 1, "planar": 3, "floating": 6}


@dataclass(frozen=True)
class UrdfInventory:
    """Movable coordinates and per-link masses of a URDF."""

    n_coordinates: int
    link_masses_kg: dict[str, float]


@dataclass(frozen=True)
class GroupMass:
    """Mass of one anatomical group in each model."""

    simscape_kg: float
    spec_kg: float

    @property
    def delta_kg(self) -> float:
        """Simscape minus spec."""
        return self.simscape_kg - self.spec_kg


@dataclass(frozen=True)
class ExchangeReport:
    """Diff of the R2025b exchange receipt against the URDF on disk."""

    receipt_current: bool
    coordinates: tuple[int, int]
    round_trip_coordinates: tuple[int, int]
    round_trip_mass_error_kg: float
    groups: dict[str, GroupMass]
    simscape_total_kg: float
    spec_total_kg: float
    unavailable_simscape: tuple[str, ...]
    unmatched_simscape: tuple[str, ...]
    unmatched_spec: tuple[str, ...]


def load_exchange_receipt(path: Path) -> dict[str, Any]:
    """Load a receipt; raises ``ValueError`` unless it is :data:`SCHEMA`."""
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise ValueError(
            f"receipt schema must be {SCHEMA!r}, got {receipt.get('schema')!r}"
        )
    return receipt


def urdf_inventory(path: Path) -> UrdfInventory:
    """Movable DOF count and ``<inertial><mass>`` of every link that has one."""
    root = DefusedET.parse(str(path)).getroot()
    masses: dict[str, float] = {}
    for link in root.iter("link"):
        mass = link.find("inertial/mass")
        if mass is not None:
            masses[link.get("name", "")] = float(mass.get("value", "nan"))
    dof = sum(_URDF_DOF.get(j.get("type", ""), 0) for j in root.iter("joint"))
    return UrdfInventory(n_coordinates=dof, link_masses_kg=masses)


def lf_sha256(path: Path) -> str:
    """SHA-256 of a text file with CR bytes removed (CRLF and LF checkouts agree)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r", b"")).hexdigest()


def _items(value: Any) -> list[Mapping[str, Any]]:
    """MATLAB ``jsonencode`` writes a one-element struct array as an object."""
    if isinstance(value, Mapping):
        return [value]
    return list(value or [])


def _leaf(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_intermediate(link: str) -> bool:
    return link.endswith("_intermediate") or "_gimbal_" in link


def _group_of(name: str, side: int) -> str | None:
    for group, members in SEGMENT_GROUPS.items():
        if name in members[side]:
            return group
    return None


def exchange_report(receipt: Mapping[str, Any], urdf_path: Path) -> ExchangeReport:
    """Diff a loaded receipt against the URDF at ``urdf_path``.

    Preconditions: ``receipt`` passed :func:`load_exchange_receipt`; the spec
    import succeeded (``spec_urdf.smimport_ok``).
    Postconditions: group masses sum the Simscape bodies and URDF links listed
    in :data:`SEGMENT_GROUPS`; massless or unevaluated bodies are never counted
    as matched mass.
    """
    spec = receipt["spec_urdf"]
    if not spec.get("smimport_ok"):
        raise ValueError(f"spec URDF smimport failed: {spec.get('smimport_error')!r}")
    urdf = urdf_inventory(urdf_path)
    current = spec["urdf_sha256_lf"] == lf_sha256(urdf_path)
    canonical = receipt["canonical"]["inventory"]
    imported = spec["inventory"]

    sim_mass: dict[str, float] = dict.fromkeys(SEGMENT_GROUPS, 0.0)
    unavailable, unmatched_sim, sim_total = [], [], 0.0
    for body in _items(canonical["bodies"]):
        mass = body.get("mass_kg")
        if mass is None:
            unavailable.append(body["path"])
            continue
        sim_total += mass
        group = _group_of(_leaf(body["path"]), 0)
        if group is not None:
            sim_mass[group] += mass
        elif mass > 0.0:
            unmatched_sim.append(body["path"])

    spec_mass: dict[str, float] = dict.fromkeys(SEGMENT_GROUPS, 0.0)
    unmatched_spec = []
    for link, mass in urdf.link_masses_kg.items():
        group = _group_of(link, 1)
        if group is not None:
            spec_mass[group] += mass
        elif mass > 0.0 and not _is_intermediate(link):
            unmatched_spec.append(link)

    imported_mass = sum(
        b["mass_kg"] or 0.0
        for b in _items(imported["bodies"])
        if b["kind"] == "Inertia"
    )
    spec_total = sum(urdf.link_masses_kg.values())
    return ExchangeReport(
        receipt_current=current,
        coordinates=(int(canonical["n_coordinates"]), urdf.n_coordinates),
        round_trip_coordinates=(int(imported["n_coordinates"]), urdf.n_coordinates),
        round_trip_mass_error_kg=abs(imported_mass - spec_total),
        groups={g: GroupMass(sim_mass[g], spec_mass[g]) for g in SEGMENT_GROUPS},
        simscape_total_kg=sim_total,
        spec_total_kg=spec_total,
        unavailable_simscape=tuple(unavailable),
        unmatched_simscape=tuple(unmatched_sim),
        unmatched_spec=tuple(unmatched_spec),
    )
