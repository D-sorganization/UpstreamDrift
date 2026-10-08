"""Rigid-body description of the club as seen by the grip (issue #11739).

Mass, centre of mass and inertia come from the club solids of the full-body
spec (head, shaft and grip; the hand solids are excluded because they move to
the hand bodies in the bushing model).  All vectors are in the club body frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClubDynamics:
    """Club mass, centre of mass and inertia about the centre of mass."""

    mass_kg: float
    com_m: tuple[float, float, float]
    inertia_com_kg_m2: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError(f"mass_kg must be positive, got {self.mass_kg}")
        inertia = np.asarray(self.inertia_com_kg_m2, dtype=float)
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise ValueError("inertia_com_kg_m2 must be a finite 3x3 matrix")
        if np.any(np.linalg.eigvalsh(0.5 * (inertia + inertia.T)) <= 0.0):
            raise ValueError("inertia_com_kg_m2 must be positive definite")
        object.__setattr__(self, "inertia_com_kg_m2", inertia)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> ClubDynamics:
        """Compose the club solids of ``spec`` (closure ``body_b``), hands excluded.

        Raises:
            ValueError: if the spec has no closure or the club body has no
                non-hand solids.
        """
        closure = spec.get("closure")
        if not closure:
            raise ValueError("spec has no closure")
        body = next((b for b in spec["bodies"] if b["name"] == closure["body_b"]), None)
        solids = [s for s in (body or {}).get("solids", []) if "Hand" not in s["name"]]
        if not solids:
            raise ValueError("club body has no non-hand solids")
        placed = []
        for s in solids:
            t = np.asarray(s["placement"], dtype=float)
            com = t[:3, :3] @ np.asarray(s["com_m"], dtype=float) + t[:3, 3]
            inertia = (
                t[:3, :3]
                @ np.asarray(s["inertia_com_kg_m2"], dtype=float)
                @ t[:3, :3].T
            )
            placed.append((float(s["mass_kg"]), com, inertia))
        mass = sum(m for m, _, _ in placed)
        com = sum(m * c for m, c, _ in placed) / mass
        inertia_tot = np.zeros((3, 3))
        for m, c, inertia in placed:
            d = c - com
            inertia_tot += inertia + m * ((d @ d) * np.eye(3) - np.outer(d, d))
        return cls(mass, (float(com[0]), float(com[1]), float(com[2])), inertia_tot)
