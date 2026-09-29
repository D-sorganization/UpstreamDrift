# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""
Unified inertia calculator with multiple computation modes.

This module provides a single entry point for all inertia calculations,
supporting primitive shapes, mesh-based computation, and manual override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shared.python.model_generation.core.constants import (
    DEFAULT_DENSITY_KG_M3,
    DEFAULT_INERTIA_KG_M2,
)
from shared.python.model_generation.core.types import Geometry, GeometryType
from shared.python.model_generation.inertia.primitives import (
    box_inertia,
    capsule_inertia,
    cylinder_inertia,
    sphere_inertia,
)
from shared.python.model_generation.inertia.result import InertiaMode, InertiaResult

logger = logging.getLogger(__name__)

__all__ = ["InertiaCalculator", "InertiaMode", "InertiaResult"]


@dataclass
class InertiaCalculator:
    """Unified inertia calculator supporting multiple computation modes."""

    default_mode: InertiaMode = InertiaMode.AUTO
    default_density: float = DEFAULT_DENSITY_KG_M3
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
        """Compute inertia from geometry, mesh, dict, or anthropometric source."""
        if source is None:
            raise ValueError("source must be provided")
        resolved_mode = mode or self.default_mode
        resolved_density = density or self.default_density

        if resolved_mode == InertiaMode.AUTO:
            resolved_mode = self._detect_mode(source)

        if resolved_mode == InertiaMode.MANUAL:
            return self._compute_manual(source, mass)
        if resolved_mode == InertiaMode.PRIMITIVE:
            return self._compute_primitive(source, mass, dimensions)
        if resolved_mode in (
            InertiaMode.MESH_UNIFORM_DENSITY,
            InertiaMode.MESH_SPECIFIED_MASS,
        ):
            return self._compute_mesh(source, mass, resolved_density, resolved_mode)
        if resolved_mode == InertiaMode.ANTHROPOMETRIC:
            return self._compute_anthropometric(source, mass, dimensions, **kwargs)
        raise ValueError(f"Unsupported inertia mode: {resolved_mode}")

    def compute_from_geometry(self, geometry: Geometry, mass: float) -> InertiaResult:
        """Compute inertia from Geometry object."""
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        return self.compute(geometry, mass=mass, mode=InertiaMode.PRIMITIVE)

    def compute_from_mesh(
        self,
        mesh_path: str | Path,
        mass: float | None = None,
        density: float | None = None,
    ) -> InertiaResult:
        """Compute inertia from mesh file."""
        if mesh_path is None:
            raise ValueError("mesh_path must be provided")
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
        """Create inertia from manual values."""
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
            return (
                InertiaMode.MANUAL
                if ("ixx" in source or "inertia" in source)
                else InertiaMode.PRIMITIVE
            )
        if isinstance(source, Geometry):
            return (
                InertiaMode.MESH_UNIFORM_DENSITY
                if source.geometry_type == GeometryType.MESH
                else InertiaMode.PRIMITIVE
            )
        if isinstance(source, str | Path):
            if Path(source).suffix.lower() in (
                ".stl",
                ".obj",
                ".ply",
                ".dae",
                ".glb",
            ):
                return InertiaMode.MESH_UNIFORM_DENSITY
            return InertiaMode.PRIMITIVE
        return InertiaMode.PRIMITIVE

    def _compute_manual(self, source: Any, mass: float | None) -> InertiaResult:
        """Compute from manual specification."""
        if not isinstance(source, dict):
            raise ValueError(f"Manual mode requires dict, got {type(source)}")
        data = source.get("inertia", source)
        result = InertiaResult(
            ixx=data.get("ixx", DEFAULT_INERTIA_KG_M2),
            iyy=data.get("iyy", DEFAULT_INERTIA_KG_M2),
            izz=data.get("izz", DEFAULT_INERTIA_KG_M2),
            ixy=data.get("ixy", 0.0),
            ixz=data.get("ixz", 0.0),
            iyz=data.get("iyz", 0.0),
            mass=source.get("mass", mass or 1.0),
            center_of_mass=tuple(source.get("center_of_mass", (0.0, 0.0, 0.0))),
            mode=InertiaMode.MANUAL,
            source="manual",
        )
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
            geom = self._geometry_from_dimensions(dimensions)
        else:
            raise ValueError("Primitive mode requires Geometry, dict, or dimensions")
        target_mass = 1.0 if mass is None else mass

        if geom.geometry_type == GeometryType.BOX:
            inertia = box_inertia(target_mass, *geom.dimensions[:3])
        elif geom.geometry_type == GeometryType.CYLINDER:
            inertia = cylinder_inertia(
                target_mass, geom.dimensions[0], geom.dimensions[1]
            )
        elif geom.geometry_type == GeometryType.SPHERE:
            inertia = sphere_inertia(target_mass, geom.dimensions[0])
        elif geom.geometry_type == GeometryType.CAPSULE:
            inertia = capsule_inertia(
                target_mass, geom.dimensions[0], geom.dimensions[1]
            )
        else:
            radius = geom.dimensions[0] if geom.dimensions else 0.1
            inertia = sphere_inertia(target_mass, radius)

        return InertiaResult(
            ixx=inertia["ixx"],
            iyy=inertia["iyy"],
            izz=inertia["izz"],
            ixy=inertia.get("ixy", 0.0),
            ixz=inertia.get("ixz", 0.0),
            iyz=inertia.get("iyz", 0.0),
            mass=target_mass,
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
        if density is None:
            raise ValueError("density must be provided")
        mesh_path = self._resolve_mesh_path(source)
        cache_key = f"{mesh_path}:{density}:{mass}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        mesh = self._load_mesh(mesh_path, mode, mass)
        if mesh is None:
            return self._create_default_inertia_result(mass, mode, str(mesh_path))

        props = self._extract_mesh_properties(mesh, mesh_path, mode, mass)
        if props is None:
            return self._create_default_inertia_result(mass, mode, str(mesh_path))

        scaled, final_mass = self._compute_scaled_inertia(
            props["raw_inertia"], props["volume"], mass, density, mode
        )
        com = props["com"]
        result = InertiaResult(
            ixx=float(scaled[0, 0]),
            iyy=float(scaled[1, 1]),
            izz=float(scaled[2, 2]),
            ixy=float(scaled[0, 1]),
            ixz=float(scaled[0, 2]),
            iyz=float(scaled[1, 2]),
            mass=final_mass,
            center_of_mass=(float(com[0]), float(com[1]), float(com[2])),
            volume=props["volume"],
            mode=mode,
            is_watertight=props["is_watertight"],
            source=str(mesh_path),
        )
        self._cache[cache_key] = result
        return result

    def _resolve_mesh_path(self, source: Any) -> Path:
        """Resolve source to a mesh file path."""
        if isinstance(source, Geometry) and source.mesh_filename:
            return Path(source.mesh_filename)
        if isinstance(source, str | Path):
            return Path(source)
        raise ValueError(f"Mesh mode requires path, got {type(source)}")

    def _load_mesh(
        self, mesh_path: Path, mode: InertiaMode, mass: float | None
    ) -> Any | None:
        """Load mesh from file, returning None on failure."""
        if mesh_path is None:
            raise ValueError("mesh_path must be provided")
        try:
            import trimesh
        except ImportError:
            logger.warning("trimesh not available, falling back to default inertia")
            return None
        try:
            loaded = trimesh.load(str(mesh_path))
            if isinstance(loaded, trimesh.Scene):
                if not loaded.geometry:
                    raise ValueError("Scene contains no geometry")
                return trimesh.util.concatenate(list(loaded.geometry.values()))
            return loaded
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Failed to load mesh %s: %s", mesh_path, e)
            return None

    def _extract_mesh_properties(
        self, mesh: Any, mesh_path: Path, mode: InertiaMode, mass: float | None
    ) -> dict[str, Any] | None:
        """Extract inertia properties from mesh, returning None on failure."""
        if mesh_path is None:
            raise ValueError("mesh_path must be provided")
        is_watertight = bool(mesh.is_watertight)
        if not is_watertight:
            logger.warning(
                "Mesh %s is not watertight, inertia may be inaccurate",
                mesh_path,
            )
        try:
            return {
                "raw_inertia": mesh.moment_inertia,
                "volume": float(mesh.volume) if is_watertight else None,
                "com": mesh.center_mass if is_watertight else mesh.centroid,
                "is_watertight": is_watertight,
            }
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning("Failed to compute mesh properties: %s", e)
            return None

    def _compute_scaled_inertia(
        self,
        raw_inertia: np.ndarray,
        volume: float | None,
        mass: float | None,
        density: float,
        mode: InertiaMode,
    ) -> tuple[np.ndarray, float]:
        """Compute scaled inertia and final mass based on mode."""
        if mode == InertiaMode.MESH_SPECIFIED_MASS and mass is not None:
            if volume and volume > 0:
                return raw_inertia * (mass / volume), mass
            raw_mass = float(np.trace(raw_inertia) / 3.0)
            return (
                (raw_inertia * (mass / raw_mass), mass)
                if raw_mass > 0
                else (raw_inertia, mass)
            )
        final_mass = volume * density if volume else mass or 1.0
        return raw_inertia * density, final_mass

    def _create_default_inertia_result(
        self, mass: float | None, mode: InertiaMode, source_path: str
    ) -> InertiaResult:
        """Create a default inertia result for fallback cases."""
        return InertiaResult(
            ixx=DEFAULT_INERTIA_KG_M2,
            iyy=DEFAULT_INERTIA_KG_M2,
            izz=DEFAULT_INERTIA_KG_M2,
            mass=mass or 1.0,
            mode=mode,
            source=source_path,
        )

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
        try:
            from shared.python.model_generation.humanoid.anthropometry import (
                estimate_segment_inertia_from_gyration,
            )

            d = estimate_segment_inertia_from_gyration(
                segment_name, mass or 1.0, length, gender_factor
            )
            return InertiaResult(
                ixx=d["ixx"],
                iyy=d["iyy"],
                izz=d["izz"],
                ixy=d.get("ixy", 0.0),
                ixz=d.get("ixz", 0.0),
                iyz=d.get("iyz", 0.0),
                mass=mass or 1.0,
                mode=InertiaMode.ANTHROPOMETRIC,
                source=f"anthropometric:{segment_name}",
            )
        except ImportError:
            logger.warning(
                "Anthropometry data not available, falling back to primitive"
            )
            return self._compute_primitive(
                Geometry.cylinder(length * 0.1, length), mass, dimensions
            )

    def _geometry_from_dimensions(self, dimensions: dict[str, float]) -> Geometry:
        """Create geometry from dimensions dict."""
        if "radius" in dimensions and "length" in dimensions:
            return Geometry.cylinder(dimensions["radius"], dimensions["length"])
        if "length" in dimensions and "width" in dimensions:
            depth = dimensions.get("depth", dimensions["width"])
            return Geometry.box(dimensions["length"], dimensions["width"], depth)
        if "radius" in dimensions:
            return Geometry.sphere(dimensions["radius"])
        size = dimensions.get("size", 0.1)
        return Geometry.box(size, size, size)

    def clear_cache(self) -> None:
        """Clear the computation cache."""
        self._cache.clear()
