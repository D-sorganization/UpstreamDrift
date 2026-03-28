# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Mesh-based inertia calculation for humanoid character builder.

This module provides inertia tensor computation from mesh geometry
using the trimesh library. It supports:
- Uniform density calculation
- Specified mass scaling
- Manual override mode
- Watertight mesh validation and repair
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from humanoid_character_builder.contracts import precondition
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass  # For type hints without runtime import

logger = logging.getLogger(__name__)


class InertiaMode(Enum):
    """Inertia calculation mode."""

    # Compute from mesh with uniform density
    MESH_UNIFORM_DENSITY = "mesh_uniform"

    # Compute from mesh, then scale to match specified mass
    MESH_SPECIFIED_MASS = "mesh_with_mass"

    # Use primitive shape approximation (box, cylinder, etc.)
    PRIMITIVE_APPROXIMATION = "primitive"

    # User specifies all inertia values manually
    MANUAL = "manual"


@dataclass
class InertiaResult:
    """
    Result of inertia calculation.

    The inertia tensor is expressed at the center of mass,
    aligned with the principal axes (or mesh coordinate frame).
    """

    # Principal moments of inertia (kg*m^2)
    ixx: float
    iyy: float
    izz: float

    # Products of inertia (kg*m^2) - often zero for symmetric shapes
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    # Center of mass position (meters)
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Volume (m^3) - useful for density calculations
    volume: float = 0.0

    # Computed or specified mass (kg)
    mass: float = 1.0

    # Whether mesh was watertight
    was_watertight: bool = True

    # Computation mode used
    mode: InertiaMode = InertiaMode.MESH_UNIFORM_DENSITY

    def as_matrix(self) -> NDArray[np.float64]:
        """Return inertia as 3x3 matrix."""
        return np.array(
            [
                [self.ixx, self.ixy, self.ixz],
                [self.ixy, self.iyy, self.iyz],
                [self.ixz, self.iyz, self.izz],
            ]
        )

    def as_urdf_dict(self) -> dict[str, float]:
        """Return inertia values for URDF format."""
        return {
            "ixx": self.ixx,
            "ixy": self.ixy,
            "ixz": self.ixz,
            "iyy": self.iyy,
            "iyz": self.iyz,
            "izz": self.izz,
        }

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
            "was_watertight": self.was_watertight,
            "mode": self.mode.value,
        }

    def is_valid(self) -> bool:
        """Check if inertia values are physically valid."""
        # All diagonal elements must be positive
        if self.ixx <= 0 or self.iyy <= 0 or self.izz <= 0:
            return False

        # Triangle inequality for inertia
        if not (abs(self.ixx - self.iyy) <= self.izz <= self.ixx + self.iyy):
            return False
        if not (abs(self.iyy - self.izz) <= self.ixx <= self.iyy + self.izz):
            return False
        return abs(self.ixx - self.izz) <= self.iyy <= self.ixx + self.izz

    def validate_positive_definite(self) -> bool:
        """Check if inertia matrix is positive definite."""
        try:
            np.linalg.cholesky(self.as_matrix())
            return True
        except np.linalg.LinAlgError:
            return False

    @classmethod
    def create_default(cls, mass: float = 1.0) -> InertiaResult:
        """Create default inertia (small sphere approximation).

        Returns a physically reasonable minimum inertia result for degenerate
        (zero-size) geometry. Volume is set to a small non-zero value to
        prevent downstream division-by-zero in density calculations.
        """
        # Default to 0.1 kg*m^2 (reasonable for small-medium rigid body)
        if not (mass is not None):
            raise ValueError("mass must be provided")
        i_default = 0.1 * mass
        # Use volume of a sphere with 1 cm radius as minimum (~4.2e-6 m³)
        _min_volume = (4.0 / 3.0) * 3.14159265358979 * (0.01**3)
        return cls(
            ixx=i_default,
            iyy=i_default,
            izz=i_default,
            volume=_min_volume,
            mass=mass,
            mode=InertiaMode.PRIMITIVE_APPROXIMATION,
        )


class MeshInertiaCalculator:
    """
    Calculate inertia tensors from mesh geometry.

    This class uses trimesh for mesh processing and inertia computation.
    It handles:
    - Loading various mesh formats (STL, OBJ, PLY, etc.)
    - Validating and repairing non-watertight meshes
    - Computing inertia with uniform density or specified mass
    - Transforming inertia to different reference frames
    """

    # Default tissue density (kg/m^3) - approximately human tissue
    DEFAULT_DENSITY = 1050.0

    def __init__(self, default_density: float = DEFAULT_DENSITY):
        """
        Initialize the inertia calculator.

        Args:
            default_density: Default density in kg/m^3 for uniform density mode
        """
        if not (default_density is not None):
            raise ValueError("default_density must be provided")
        self.default_density = default_density
        self._trimesh_available = self._check_trimesh()

    def _check_trimesh(self) -> bool:
        """Check if trimesh is available."""
        try:
            import trimesh  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "trimesh not available. Mesh-based inertia calculation disabled. "
                "Install with: pip install trimesh"
            )
            return False

    def compute_from_mesh(
        self,
        mesh_path: Path | str,
        mass: float | None = None,
        density: float | None = None,
        repair_mesh: bool = True,
    ) -> InertiaResult:
        """
        Compute inertia tensor from mesh file.

        Args:
            mesh_path: Path to mesh file (STL, OBJ, PLY, etc.)
            mass: Specified mass in kg (scales inertia to match)
            density: Density in kg/m^3 (used if mass not specified)
            repair_mesh: Attempt to repair non-watertight meshes

        Returns:
            InertiaResult with computed inertia tensor

        Raises:
            ImportError: If trimesh is not available
            FileNotFoundError: If mesh file doesn't exist
            ValueError: If mesh processing fails
        """
        if not self._trimesh_available:
            raise ImportError(
                "trimesh is required for mesh-based inertia calculation. "
                "Install with: pip install trimesh"
            )

        import trimesh

        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        # Load mesh
        try:
            mesh = trimesh.load(str(mesh_path))
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to load mesh: {e}") from e

        # Handle scene objects (multiple meshes)
        if isinstance(mesh, trimesh.Scene):
            # Concatenate all meshes in the scene
            meshes = list(mesh.geometry.values())
            if not meshes:
                raise ValueError("Scene contains no geometry")
            mesh = trimesh.util.concatenate(meshes)

        return self.compute_from_trimesh(
            mesh, mass=mass, density=density, repair_mesh=repair_mesh
        )

    def compute_from_trimesh(
        self,
        mesh: Any,  # trimesh.Trimesh
        mass: float | None = None,
        density: float | None = None,
        repair_mesh: bool = True,
    ) -> InertiaResult:
        """
        Compute inertia tensor from trimesh object.

        Args:
            mesh: trimesh.Trimesh object
            mass: Specified mass in kg (scales inertia to match)
            density: Density in kg/m^3 (used if mass not specified)
            repair_mesh: Attempt to repair non-watertight meshes

        Returns:
            InertiaResult with computed inertia tensor
        """
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh, got {type(mesh)}")

        mesh, was_watertight = self._validate_and_repair_mesh(mesh, repair_mesh)
        effective_density = density if density is not None else self.default_density

        mesh_props = self._extract_mesh_mass_properties(mesh, mass)
        if mesh_props is None:
            return InertiaResult.create_default(mass or 1.0)

        return self._create_inertia_result(
            mesh_props, mass, effective_density, was_watertight
        )

    def _validate_and_repair_mesh(
        self, mesh: Any, repair_mesh: bool
    ) -> tuple[Any, bool]:
        """Validate mesh watertightness and optionally repair."""
        if not (repair_mesh is not None):
            raise ValueError("repair_mesh must be provided")
        was_watertight = mesh.is_watertight

        if not was_watertight and repair_mesh:
            mesh = self._repair_mesh(mesh)
            was_watertight = mesh.is_watertight

        if not mesh.is_watertight:
            logger.warning(
                "Mesh is not watertight. Inertia calculation may be inaccurate."
            )

        return mesh, was_watertight

    def _extract_mesh_mass_properties(
        self, mesh: Any, mass: float | None
    ) -> dict[str, Any] | None:
        """Extract volume, center of mass, and inertia from mesh."""
        try:
            volume = mesh.volume
            center_mass = mesh.center_mass

            if volume <= 0:
                logger.warning("Mesh has non-positive volume. Using bounding box.")
                volume = mesh.bounding_box.volume
                center_mass = mesh.bounding_box.centroid

            return {
                "volume": volume,
                "center_mass": center_mass,
                "inertia_unit": mesh.moment_inertia,
            }
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to compute mesh properties: {e}. Using defaults.")
            return None

    def _create_inertia_result(
        self,
        mesh_props: dict[str, Any],
        mass: float | None,
        effective_density: float,
        was_watertight: bool,
    ) -> InertiaResult:
        """Create InertiaResult from mesh properties."""
        if not (mesh_props is not None):
            raise ValueError("mesh_props must be provided")
        volume = mesh_props["volume"]
        center_mass = mesh_props["center_mass"]
        inertia_unit = mesh_props["inertia_unit"]

        inertia, final_mass, mode = self._scale_inertia(
            inertia_unit, volume, mass, effective_density
        )

        return InertiaResult(
            ixx=float(inertia[0, 0]),
            iyy=float(inertia[1, 1]),
            izz=float(inertia[2, 2]),
            ixy=float(inertia[0, 1]),
            ixz=float(inertia[0, 2]),
            iyz=float(inertia[1, 2]),
            center_of_mass=(
                float(center_mass[0]),
                float(center_mass[1]),
                float(center_mass[2]),
            ),
            volume=float(volume),
            mass=float(final_mass),
            was_watertight=was_watertight,
            mode=mode,
        )

    def _scale_inertia(
        self,
        inertia_unit: np.ndarray,
        volume: float,
        mass: float | None,
        effective_density: float,
    ) -> tuple[np.ndarray, float, InertiaMode]:
        """Scale inertia based on mass or density."""
        if not (inertia_unit is not None):
            raise ValueError("inertia_unit must be provided")
        if mass is not None:
            scale_factor = mass / volume if volume > 0 else 1.0
            return (
                inertia_unit * scale_factor,
                mass,
                InertiaMode.MESH_SPECIFIED_MASS,
            )
        final_mass = volume * effective_density
        return (
            inertia_unit * effective_density,
            final_mass,
            InertiaMode.MESH_UNIFORM_DENSITY,
        )

    def _repair_mesh(self, mesh: Any) -> Any:
        """
        Attempt to repair a non-watertight mesh.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            Repaired mesh (may still not be watertight)
        """
        import trimesh

        logger.info("Attempting mesh repair...")

        # Try various repair operations
        try:
            # Fill holes
            trimesh.repair.fill_holes(mesh)

            # Fix normals
            mesh.fix_normals()

            # Remove degenerate faces
            mesh.remove_degenerate_faces()

            # Merge close vertices
            mesh.merge_vertices()

        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning(f"Mesh repair partially failed: {e}")

        return mesh

    def compute_from_vertices(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        mass: float | None = None,
        density: float | None = None,
    ) -> InertiaResult:
        """
        Compute inertia from raw vertices and faces.

        Args:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices
            mass: Specified mass in kg
            density: Density in kg/m^3

        Returns:
            InertiaResult with computed inertia tensor
        """
        if not self._trimesh_available:
            raise ImportError("trimesh is required")

        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return self.compute_from_trimesh(mesh, mass=mass, density=density)

    @precondition(lambda inertia: inertia is not None, "Inertia cannot be None")
    def transform_inertia(
        self,
        inertia: InertiaResult,
        rotation: NDArray[np.float64] | None = None,
        translation: NDArray[np.float64] | None = None,
    ) -> InertiaResult:
        """
        Transform inertia tensor to a new reference frame.

        Args:
            inertia: Original InertiaResult
            rotation: 3x3 rotation matrix (R)
            translation: 3D translation vector from old to new frame origin

        Returns:
            New InertiaResult in transformed frame
        """
        if not (inertia is not None):
            raise ValueError("inertia must be provided")
        I_original = inertia.as_matrix()
        mass = inertia.mass
        com = np.array(inertia.center_of_mass)

        I_rotated, com = self._apply_rotation(I_original, com, rotation)
        I_final, new_com = self._apply_translation(I_rotated, com, mass, translation)

        return self._create_transformed_result(I_final, new_com, inertia)

    def _apply_rotation(
        self,
        inertia_matrix: NDArray[np.float64],
        com: NDArray[np.float64],
        rotation: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply rotation transformation to inertia matrix and COM."""
        if not (inertia_matrix is not None):
            raise ValueError("inertia_matrix must be provided")
        if rotation is not None:
            R = np.asarray(rotation)
            return R @ inertia_matrix @ R.T, R @ com
        return inertia_matrix, com

    def _apply_translation(
        self,
        inertia_matrix: NDArray[np.float64],
        com: NDArray[np.float64],
        mass: float,
        translation: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply parallel axis theorem for translation."""
        if not (inertia_matrix is not None):
            raise ValueError("inertia_matrix must be provided")
        if translation is not None:
            d = np.asarray(translation)
            new_com = com - d
            d_sq = np.dot(new_com, new_com)
            I_translated = inertia_matrix + mass * (
                d_sq * np.eye(3) - np.outer(new_com, new_com)
            )
            return I_translated, new_com
        return inertia_matrix, com

    def _create_transformed_result(
        self,
        inertia_matrix: NDArray[np.float64],
        com: NDArray[np.float64],
        original: InertiaResult,
    ) -> InertiaResult:
        """Create InertiaResult from transformed matrix."""
        return InertiaResult(
            ixx=float(inertia_matrix[0, 0]),
            iyy=float(inertia_matrix[1, 1]),
            izz=float(inertia_matrix[2, 2]),
            ixy=float(inertia_matrix[0, 1]),
            ixz=float(inertia_matrix[0, 2]),
            iyz=float(inertia_matrix[1, 2]),
            center_of_mass=(float(com[0]), float(com[1]), float(com[2])),
            volume=original.volume,
            mass=original.mass,
            was_watertight=original.was_watertight,
            mode=original.mode,
        )

    @staticmethod
    @precondition(lambda ixx: ixx > 0, "ixx must be positive")
    @precondition(lambda iyy: iyy > 0, "iyy must be positive")
    @precondition(lambda izz: izz > 0, "izz must be positive")
    @precondition(lambda mass: mass > 0, "Mass must be positive")
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
        """
        Create an InertiaResult from manually specified values.

        Args:
            ixx, iyy, izz: Principal moments of inertia
            mass: Mass in kg
            ixy, ixz, iyz: Products of inertia (default 0)
            com: Center of mass position

        Returns:
            InertiaResult with manual mode
        """
        return InertiaResult(
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            ixy=ixy,
            ixz=ixz,
            iyz=iyz,
            center_of_mass=com,
            volume=0.0,  # Unknown for manual
            mass=mass,
            was_watertight=True,  # N/A for manual
            mode=InertiaMode.MANUAL,
        )


def validate_inertia_tensor(inertia_matrix: NDArray[np.float64]) -> list[str]:
    """
    Validate an inertia tensor and return list of issues.

    Args:
        inertia_matrix: 3x3 inertia tensor

    Returns:
        List of validation error messages (empty if valid)
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
