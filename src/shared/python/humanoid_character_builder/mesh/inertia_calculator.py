"""
Mesh-based inertia calculation for humanoid character builder.

This module is a thin wrapper around model_generation.inertia.calculator
to maintain backward compatibility while adopting the canonical URDF subsystem.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Re-export core types to maintain compatibility
from model_generation.inertia.calculator import (
    InertiaMode,
    InertiaResult,
    InertiaCalculator as CanonicalInertiaCalculator,
)

logger = logging.getLogger(__name__)


class MeshInertiaCalculator:
    """
    Calculate inertia tensors from mesh geometry.

    This is a compatibility wrapper around model_generation.inertia.calculator.InertiaCalculator.
    """

    DEFAULT_DENSITY = 1050.0

    def __init__(self, default_density: float = DEFAULT_DENSITY) -> None:
        if not (default_density is not None):
            raise ValueError("default_density must be provided")
        self.default_density = default_density
        self._calculator = CanonicalInertiaCalculator(default_density=default_density)
        self._trimesh_available = True

    def compute_from_mesh(
        self,
        mesh_path: Path | str,
        mass: float | None = None,
        density: float | None = None,
        repair_mesh: bool = True,
    ) -> InertiaResult:
        """Compute inertia tensor from mesh file."""
        return self._calculator.compute_from_mesh(
            mesh_path=mesh_path,
            mass=mass,
            density=density,
        )

    def compute_from_trimesh(
        self,
        mesh: Any,
        mass: float | None = None,
        density: float | None = None,
        repair_mesh: bool = True,
    ) -> InertiaResult:
        """Compute inertia tensor from trimesh object."""
        mode = (
            InertiaMode.MESH_SPECIFIED_MASS
            if mass is not None
            else InertiaMode.MESH_UNIFORM_DENSITY
        )
        eff_density = density if density is not None else self.default_density

        if repair_mesh and not getattr(mesh, "is_watertight", True):
            try:
                import trimesh

                trimesh.repair.fill_holes(mesh)
                mesh.fix_normals()
                mesh.remove_degenerate_faces()
                mesh.merge_vertices()
            except Exception:  # noqa: BLE001
                pass

        props = self._calculator._extract_mesh_properties(
            mesh, Path("trimesh"), mode, mass
        )
        if props is None:
            return InertiaResult.create_default(mass or 1.0)

        return self._calculator._scale_and_create_result(
            props, mass, eff_density, mode, "trimesh"
        )

    def compute_from_vertices(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        mass: float | None = None,
        density: float | None = None,
    ) -> InertiaResult:
        """Compute inertia from raw vertices and faces."""
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return self.compute_from_trimesh(mesh, mass=mass, density=density)

    def transform_inertia(
        self,
        inertia: InertiaResult,
        rotation: NDArray[np.float64] | None = None,
        translation: NDArray[np.float64] | None = None,
    ) -> InertiaResult:
        """Transform inertia tensor to a new reference frame."""
        if not (inertia is not None):
            raise ValueError("inertia must be provided")

        I_orig = inertia.as_matrix()
        mass = inertia.mass
        com = np.array(inertia.center_of_mass)

        # Apply rotation (I_rot = R * I * R^T)
        if rotation is not None:
            R = np.asarray(rotation)
            I_rot = R @ I_orig @ R.T
            com = R @ com
        else:
            I_rot = I_orig

        # Apply translation (Parallel Axis Theorem)
        if translation is not None:
            d = np.asarray(translation)
            new_com = com - d
            d_sq = np.dot(new_com, new_com)
            I_final = I_rot + mass * (d_sq * np.eye(3) - np.outer(new_com, new_com))
        else:
            I_final = I_rot
            new_com = com

        return InertiaResult(
            ixx=float(I_final[0, 0]),
            iyy=float(I_final[1, 1]),
            izz=float(I_final[2, 2]),
            ixy=float(I_final[0, 1]),
            ixz=float(I_final[0, 2]),
            iyz=float(I_final[1, 2]),
            center_of_mass=(float(new_com[0]), float(new_com[1]), float(new_com[2])),
            volume=inertia.volume,
            mass=inertia.mass,
            mode=inertia.mode,
            is_watertight=inertia.is_watertight,
            source=inertia.source,
        )

    @staticmethod
    def create_manual_inertia(
        ixx: float,
        iyy: float,
        izz: float,
        mass: float,
        ixy: float = 0.0,
        ixz: float = 0.0,
        iyz: float = 0.0,
        com: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> InertiaResult:
        """Create an InertiaResult from manually specified values."""
        return InertiaResult(
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            mass=mass,
            ixy=ixy,
            ixz=ixz,
            iyz=iyz,
            center_of_mass=com,
            mode=InertiaMode.MANUAL,
        )


def validate_inertia_tensor(inertia_matrix: NDArray[np.float64]) -> list[str]:
    """
    Validate an inertia tensor and return list of issues.
    """
    errors = []
    tensor = np.asarray(inertia_matrix)

    if tensor.shape != (3, 3):
        errors.append(f"Inertia must be 3x3, got {tensor.shape}")
        return errors

    # Check symmetry
    if not np.allclose(tensor, tensor.T, rtol=1e-6):
        errors.append("Inertia tensor is not symmetric")

    # Check positive diagonal
    if np.any(np.diag(tensor) <= 0):
        errors.append("Diagonal elements must be positive")

    # Check positive definite
    try:
        np.linalg.cholesky(tensor)
    except np.linalg.LinAlgError:
        errors.append("Inertia tensor is not positive definite")

    # Check triangle inequality
    ixx, iyy, izz = tensor[0, 0], tensor[1, 1], tensor[2, 2]
    if not (abs(ixx - iyy) <= izz <= ixx + iyy):
        errors.append("Triangle inequality violated: Izz")
    if not (abs(iyy - izz) <= ixx <= iyy + izz):
        errors.append("Triangle inequality violated: Ixx")
    if not (abs(ixx - izz) <= iyy <= ixx + izz):
        errors.append("Triangle inequality violated: Iyy")

    return errors
