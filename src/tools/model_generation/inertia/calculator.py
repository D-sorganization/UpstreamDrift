"""
Unified inertia calculator with multiple computation modes.

This module provides a single entry point for all inertia calculations,
supporting primitive shapes, mesh-based computation, and manual override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from model_generation.core.constants import DEFAULT_DENSITY_KG_M3, DEFAULT_INERTIA_KG_M2
from model_generation.core.types import Geometry, GeometryType, Inertia
from model_generation.inertia.primitives import (
    box_inertia,
    capsule_inertia,
    cylinder_inertia,
    sphere_inertia,
)

logger = logging.getLogger(__name__)


class InertiaMode(Enum):
    """Inertia calculation modes."""

    # Automatic: detect best mode based on input
    AUTO = "auto"

    # Use analytical formulas for primitive shapes
    PRIMITIVE = "primitive"

    # Compute from mesh geometry with uniform density
    MESH_UNIFORM_DENSITY = "mesh_uniform"

    # Compute from mesh geometry, scale to specified mass
    MESH_SPECIFIED_MASS = "mesh_mass"

    # Use manually specified inertia values
    MANUAL = "manual"

    # Use anthropometric data (for humanoid segments)
    ANTHROPOMETRIC = "anthropometric"


@dataclass
class InertiaResult:
    """Result of inertia calculation."""

    # Inertia tensor components (kg*m^2)
    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    # Mass (kg)
    mass: float = 1.0

    # Center of mass relative to frame origin (m)
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Volume if computed (m^3)
    volume: float | None = None

    # Mode used for calculation
    mode: InertiaMode = InertiaMode.PRIMITIVE

    # Whether mesh was watertight (for mesh modes)
    is_watertight: bool | None = None

    # Source information
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

    def is_valid(self) -> bool:
        """Check if inertia values are physically valid."""
        return self.to_inertia().is_positive_definite()

    def scale_to_mass(self, new_mass: float) -> InertiaResult:
        """Return new result scaled to different mass."""
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


@dataclass
class InertiaCalculator:
    """
    Unified inertia calculator supporting multiple computation modes.

    This is the primary interface for computing inertia tensors throughout
    the model_generation package.
    """

    # Default calculation mode
    default_mode: InertiaMode = InertiaMode.AUTO

    # Default density for mesh-based calculation (kg/m^3)
    default_density: float = DEFAULT_DENSITY_KG_M3

    # Cache for mesh computations
    _cache: dict[str, InertiaResult] = field(default_factory=dict)

    def compute(
        self,
        source: str | Path | Geometry | dict[str, Any],
        mass: float | None = None,
        density: float | None = None,
        mode: InertiaMode | None = None,
        dimensions: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> InertiaResult:
        """
        Compute inertia from various sources.

        Args:
            source: One of:
                - Path to mesh file (str or Path)
                - Geometry object (primitive shape)
                - Dict with manual inertia values
            mass: Target mass (for scaling)
            density: Density for uniform density mode (kg/m^3)
            mode: Calculation mode (auto-detected if None)
            dimensions: Shape dimensions for primitive mode
            **kwargs: Additional mode-specific options

        Returns:
            InertiaResult with computed inertia

        Examples:
            # From primitive geometry
            calc.compute(Geometry.cylinder(0.05, 0.4), mass=2.0)

            # From mesh file
            calc.compute("arm.stl", density=1050.0)

            # From mesh with specified mass
            calc.compute("arm.stl", mass=2.0, mode=InertiaMode.MESH_SPECIFIED_MASS)

            # Manual values
            calc.compute({"ixx": 0.1, "iyy": 0.1, "izz": 0.05, "mass": 2.0})
        """
        mode = mode or self.default_mode
        density = density or self.default_density

        # Auto-detect mode
        if mode == InertiaMode.AUTO:
            mode = self._detect_mode(source)

        # Route to appropriate method
        if mode == InertiaMode.MANUAL:
            return self._compute_manual(source, mass)
        elif mode == InertiaMode.PRIMITIVE:
            return self._compute_primitive(source, mass, dimensions)
        elif mode in (
            InertiaMode.MESH_UNIFORM_DENSITY,
            InertiaMode.MESH_SPECIFIED_MASS,
        ):
            return self._compute_mesh(source, mass, density, mode)
        elif mode == InertiaMode.ANTHROPOMETRIC:
            return self._compute_anthropometric(source, mass, dimensions, **kwargs)
        else:
            raise ValueError(f"Unsupported inertia mode: {mode}")

    def compute_from_geometry(
        self,
        geometry: Geometry,
        mass: float,
    ) -> InertiaResult:
        """
        Compute inertia from Geometry object.

        Args:
            geometry: Geometry specification
            mass: Mass in kg

        Returns:
            InertiaResult
        """
        return self.compute(geometry, mass=mass, mode=InertiaMode.PRIMITIVE)

    def compute_from_mesh(
        self,
        mesh_path: str | Path,
        mass: float | None = None,
        density: float | None = None,
    ) -> InertiaResult:
        """
        Compute inertia from mesh file.

        Args:
            mesh_path: Path to mesh file
            mass: If specified, scale to this mass
            density: Density for uniform density calculation

        Returns:
            InertiaResult
        """
        mode = (
            InertiaMode.MESH_SPECIFIED_MASS
            if mass is not None
            else InertiaMode.MESH_UNIFORM_DENSITY
        )
        return self.compute(mesh_path, mass=mass, density=density, mode=mode)

    def compute_from_manual(
        self,
        ixx: float,
        iyy: float,
        izz: float,
        mass: float,
        ixy: float = 0.0,
        ixz: float = 0.0,
        iyz: float = 0.0,
        center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> InertiaResult:
        """
        Create inertia from manual values.

        Args:
            ixx, iyy, izz: Diagonal inertia components
            mass: Mass in kg
            ixy, ixz, iyz: Off-diagonal inertia components
            center_of_mass: COM position

        Returns:
            InertiaResult
        """
        return InertiaResult(
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            ixy=ixy,
            ixz=ixz,
            iyz=iyz,
            mass=mass,
            center_of_mass=center_of_mass,
            mode=InertiaMode.MANUAL,
            source="manual",
        )

    def _detect_mode(self, source: Any) -> InertiaMode:
        """Auto-detect appropriate mode from source type."""
        if isinstance(source, dict):
            if "ixx" in source or "inertia" in source:
                return InertiaMode.MANUAL
            return InertiaMode.PRIMITIVE

        if isinstance(source, Geometry):
            if source.geometry_type == GeometryType.MESH:
                return InertiaMode.MESH_UNIFORM_DENSITY
            return InertiaMode.PRIMITIVE

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() in (".stl", ".obj", ".ply", ".dae", ".glb"):
                return InertiaMode.MESH_UNIFORM_DENSITY
            return InertiaMode.PRIMITIVE

        return InertiaMode.PRIMITIVE

    def _compute_manual(self, source: Any, mass: float | None) -> InertiaResult:
        """Compute from manual specification."""
        if isinstance(source, dict):
            data = source
        else:
            raise ValueError(f"Manual mode requires dict, got {type(source)}")

        # Handle nested inertia dict
        if "inertia" in data:
            inertia_data = data["inertia"]
        else:
            inertia_data = data

        result = InertiaResult(
            ixx=inertia_data.get("ixx", DEFAULT_INERTIA_KG_M2),
            iyy=inertia_data.get("iyy", DEFAULT_INERTIA_KG_M2),
            izz=inertia_data.get("izz", DEFAULT_INERTIA_KG_M2),
            ixy=inertia_data.get("ixy", 0.0),
            ixz=inertia_data.get("ixz", 0.0),
            iyz=inertia_data.get("iyz", 0.0),
            mass=data.get("mass", mass or 1.0),
            center_of_mass=tuple(data.get("center_of_mass", (0.0, 0.0, 0.0))),
            mode=InertiaMode.MANUAL,
            source="manual",
        )

        # Scale to target mass if specified
        if mass is not None and mass != result.mass:
            result = result.scale_to_mass(mass)

        return result

    def _compute_primitive(
        self,
        source: Any,
        mass: float | None,
        dimensions: dict[str, float] | None,
    ) -> InertiaResult:
        """Compute from primitive geometry."""
        if isinstance(source, Geometry):
            geom = source
        elif isinstance(source, dict):
            geom = Geometry.from_dict(source)
        elif dimensions:
            # Use dimensions to create geometry
            geom = self._geometry_from_dimensions(dimensions)
        else:
            raise ValueError("Primitive mode requires Geometry, dict, or dimensions")

        if mass is None:
            mass = 1.0

        # Compute based on geometry type
        if geom.geometry_type == GeometryType.BOX:
            inertia = box_inertia(mass, *geom.dimensions[:3])
        elif geom.geometry_type == GeometryType.CYLINDER:
            inertia = cylinder_inertia(mass, geom.dimensions[0], geom.dimensions[1])
        elif geom.geometry_type == GeometryType.SPHERE:
            inertia = sphere_inertia(mass, geom.dimensions[0])
        elif geom.geometry_type == GeometryType.CAPSULE:
            inertia = capsule_inertia(mass, geom.dimensions[0], geom.dimensions[1])
        else:
            # Default to sphere approximation
            radius = geom.dimensions[0] if geom.dimensions else 0.1
            inertia = sphere_inertia(mass, radius)

        return InertiaResult(
            ixx=inertia["ixx"],
            iyy=inertia["iyy"],
            izz=inertia["izz"],
            ixy=inertia.get("ixy", 0.0),
            ixz=inertia.get("ixz", 0.0),
            iyz=inertia.get("iyz", 0.0),
            mass=mass,
            mode=InertiaMode.PRIMITIVE,
            source=f"primitive:{geom.geometry_type.value}",
        )

    def _compute_mesh(
        self,
        source: Any,
        mass: float | None,
        density: float,
        mode: InertiaMode,
    ) -> InertiaResult:
        """Compute from mesh file using trimesh."""
        if isinstance(source, Geometry) and source.mesh_filename:
            mesh_path = Path(source.mesh_filename)
        elif isinstance(source, (str, Path)):
            mesh_path = Path(source)
        else:
            raise ValueError(f"Mesh mode requires path, got {type(source)}")

        # Check cache
        cache_key = f"{mesh_path}:{density}:{mass}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try to import trimesh
        try:
            import trimesh
        except ImportError:
            logger.warning("trimesh not available, falling back to default inertia")
            return InertiaResult(
                ixx=DEFAULT_INERTIA_KG_M2,
                iyy=DEFAULT_INERTIA_KG_M2,
                izz=DEFAULT_INERTIA_KG_M2,
                mass=mass or 1.0,
                mode=mode,
                source=str(mesh_path),
            )

        # Load mesh
        try:
            mesh = trimesh.load(str(mesh_path))
            if isinstance(mesh, trimesh.Scene):
                meshes = list(mesh.geometry.values())
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    raise ValueError("Scene contains no geometry")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to load mesh {mesh_path}: {e}")
            return InertiaResult(
                ixx=DEFAULT_INERTIA_KG_M2,
                iyy=DEFAULT_INERTIA_KG_M2,
                izz=DEFAULT_INERTIA_KG_M2,
                mass=mass or 1.0,
                mode=mode,
                source=str(mesh_path),
            )

        # Type narrow: after Scene handling, mesh should be a Trimesh
        assert isinstance(mesh, trimesh.Trimesh), f"Expected Trimesh, got {type(mesh)}"

        is_watertight = mesh.is_watertight
        if not is_watertight:
            logger.warning(
                f"Mesh {mesh_path} is not watertight, inertia may be inaccurate"
            )

        # Get inertia at COM (assuming unit density)
        try:
            raw_inertia = mesh.moment_inertia
            volume = float(mesh.volume) if is_watertight else None
            com = mesh.center_mass if is_watertight else mesh.centroid
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to compute mesh properties: {e}")
            return InertiaResult(
                ixx=DEFAULT_INERTIA_KG_M2,
                iyy=DEFAULT_INERTIA_KG_M2,
                izz=DEFAULT_INERTIA_KG_M2,
                mass=mass or 1.0,
                mode=mode,
                source=str(mesh_path),
            )

        # Scale inertia based on mode
        if mode == InertiaMode.MESH_SPECIFIED_MASS and mass is not None:
            # Scale to specified mass
            if volume and volume > 0:
                computed_density = mass / volume
                scaled_inertia = raw_inertia * computed_density
                final_mass = mass
            else:
                # Can't compute density, use raw inertia and scale
                raw_mass = np.trace(raw_inertia) / 3.0  # Rough estimate
                if raw_mass > 0:
                    scale = mass / raw_mass
                    scaled_inertia = raw_inertia * scale
                else:
                    scaled_inertia = raw_inertia
                final_mass = mass
        else:
            # Uniform density mode
            scaled_inertia = raw_inertia * density
            final_mass = volume * density if volume else mass or 1.0

        result = InertiaResult(
            ixx=float(scaled_inertia[0, 0]),
            iyy=float(scaled_inertia[1, 1]),
            izz=float(scaled_inertia[2, 2]),
            ixy=float(scaled_inertia[0, 1]),
            ixz=float(scaled_inertia[0, 2]),
            iyz=float(scaled_inertia[1, 2]),
            mass=final_mass,
            center_of_mass=(float(com[0]), float(com[1]), float(com[2])),
            volume=volume,
            mode=mode,
            is_watertight=is_watertight,
            source=str(mesh_path),
        )

        # Cache result
        self._cache[cache_key] = result

        return result

    def _compute_anthropometric(
        self,
        source: Any,
        mass: float | None,
        dimensions: dict[str, float] | None,
        **kwargs: Any,
    ) -> InertiaResult:
        """Compute using anthropometric data."""
        segment_name = kwargs.get(
            "segment_name", source if isinstance(source, str) else None
        )
        gender_factor = kwargs.get("gender_factor", 0.5)
        length = dimensions.get("length", 0.1) if dimensions else 0.1

        if segment_name is None:
            raise ValueError("Anthropometric mode requires segment_name")

        # Import anthropometry data
        try:
            from model_generation.humanoid.anthropometry import (
                estimate_segment_inertia_from_gyration,
            )

            inertia_dict = estimate_segment_inertia_from_gyration(
                segment_name, mass or 1.0, length, gender_factor
            )

            return InertiaResult(
                ixx=inertia_dict["ixx"],
                iyy=inertia_dict["iyy"],
                izz=inertia_dict["izz"],
                ixy=inertia_dict.get("ixy", 0.0),
                ixz=inertia_dict.get("ixz", 0.0),
                iyz=inertia_dict.get("iyz", 0.0),
                mass=mass or 1.0,
                mode=InertiaMode.ANTHROPOMETRIC,
                source=f"anthropometric:{segment_name}",
            )
        except ImportError:
            logger.warning(
                "Anthropometry data not available, falling back to primitive"
            )
            return self._compute_primitive(
                Geometry.cylinder(length * 0.1, length),
                mass,
                dimensions,
            )

    def _geometry_from_dimensions(self, dimensions: dict[str, float]) -> Geometry:
        """Create geometry from dimensions dict."""
        if "radius" in dimensions and "length" in dimensions:
            return Geometry.cylinder(dimensions["radius"], dimensions["length"])
        elif "length" in dimensions and "width" in dimensions:
            depth = dimensions.get("depth", dimensions["width"])
            return Geometry.box(dimensions["length"], dimensions["width"], depth)
        elif "radius" in dimensions:
            return Geometry.sphere(dimensions["radius"])
        else:
            # Default to small box
            size = dimensions.get("size", 0.1)
            return Geometry.box(size, size, size)

    def clear_cache(self) -> None:
        """Clear the computation cache."""
        self._cache.clear()
