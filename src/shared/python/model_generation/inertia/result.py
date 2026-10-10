# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""
Inertia calculation results and mode enumerations.

Provides data structures representing computed inertia tensors, validity checks,
and serialization helpers for downstream URDF and simulation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.model_generation.core.contracts import precondition
from src.shared.python.model_generation.core.types import Inertia


class InertiaMode(Enum):
    """Inertia calculation modes."""

    AUTO = "auto"
    PRIMITIVE = "primitive"
    MESH_UNIFORM_DENSITY = "mesh_uniform"
    MESH_SPECIFIED_MASS = "mesh_mass"
    MANUAL = "manual"
    ANTHROPOMETRIC = "anthropometric"


@dataclass
class InertiaResult:
    """Result of inertia calculation."""

    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    mass: float = 1.0
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume: float | None = None
    mode: InertiaMode = InertiaMode.PRIMITIVE
    is_watertight: bool | None = None
    source: str | None = None

    def to_inertia(self) -> Inertia:
        """Convert to core Inertia type."""
        return Inertia(
            ixx=self.ixx,
            iyy=self.iyy,
            izz=self.izz,
            ixy=self.ixy,
            ixz=self.ixz,
            iyz=self.iyz,
            mass=self.mass,
            center_of_mass=self.center_of_mass,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ixx": self.ixx,
            "iyy": self.iyy,
            "izz": self.izz,
            "ixy": self.ixy,
            "ixz": self.ixz,
            "iyz": self.iyz,
            "mass": self.mass,
            "center_of_mass": list(self.center_of_mass),
            "volume": self.volume,
            "mode": self.mode.value,
            "is_watertight": self.is_watertight,
            "source": self.source,
        }

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching humanoid builder format."""
        return {
            "ixx": self.ixx,
            "iyy": self.iyy,
            "izz": self.izz,
            "ixy": self.ixy,
            "ixz": self.ixz,
            "iyz": self.iyz,
            "center_of_mass": list(self.center_of_mass),
            "volume": self.volume,
            "mass": self.mass,
            "was_watertight": self.is_watertight,
            "mode": self.mode.value,
        }

    def as_matrix(self) -> np.ndarray:
        """Convert to 3x3 symmetric inertia matrix."""
        return self.to_inertia().to_matrix()

    def as_urdf_dict(self) -> dict[str, float]:
        """Convert to URDF inertia dict format."""
        return {
            "ixx": self.ixx,
            "ixy": self.ixy,
            "ixz": self.ixz,
            "iyy": self.iyy,
            "iyz": self.iyz,
            "izz": self.izz,
        }

    def validate_positive_definite(self) -> bool:
        """Check if inertia matrix is positive definite."""
        mat = self.as_matrix()
        if not np.all(np.isfinite(mat)):
            return False
        try:
            return bool(np.all(np.linalg.eigvalsh(mat) > 0))
        except np.linalg.LinAlgError:
            return False

    def is_valid(self) -> bool:
        """Check if inertia values are physically valid."""
        if not np.all(
            np.isfinite([self.ixx, self.iyy, self.izz, self.ixy, self.ixz, self.iyz])
        ):
            return False
        if self.ixx <= 0 or self.iyy <= 0 or self.izz <= 0:
            return False
        if not self.to_inertia().satisfies_triangle_inequality():
            return False
        return self.validate_positive_definite()

    @classmethod
    def create_default(cls, mass: float = 1.0) -> InertiaResult:
        """Create default inertia with small sphere approximation (0.1 * mass)."""
        if mass is None or mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        i_default = 0.1 * mass
        return cls(
            ixx=i_default,
            iyy=i_default,
            izz=i_default,
            mass=mass,
            mode=InertiaMode.PRIMITIVE,
        )

    @precondition(lambda new_mass: new_mass > 0, "New mass must be positive")
    def scale_to_mass(self, new_mass: float) -> InertiaResult:
        """Return new result scaled to different mass."""
        if new_mass <= 0:
            raise ValueError(f"new_mass must be positive, got {new_mass}")
        if self.mass <= 0:
            raise ValueError("Cannot scale from zero or negative mass")
        scale = new_mass / self.mass
        return InertiaResult(
            ixx=self.ixx * scale,
            iyy=self.iyy * scale,
            izz=self.izz * scale,
            ixy=self.ixy * scale,
            ixz=self.ixz * scale,
            iyz=self.iyz * scale,
            mass=new_mass,
            center_of_mass=self.center_of_mass,
            volume=self.volume,
            mode=self.mode,
            is_watertight=self.is_watertight,
            source=self.source,
        )
