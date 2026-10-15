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

from src.shared.python.biomechanics.club_inertia_partition import (
    club_solids_partition,
)


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

        Delegates to :func:`club_solids_partition` (NSX-01, #12550), the one
        place that separates the bare club from the welded hand solids.

        Raises:
            ValueError: if the spec has no closure, the club body has no
                non-hand solids or the bare mass contradicts ``club.total_mass_kg``.
        """
        bare = club_solids_partition(spec).bare_club
        return cls(bare.mass_kg, bare.com_m, bare.inertia_com_kg_m2)
