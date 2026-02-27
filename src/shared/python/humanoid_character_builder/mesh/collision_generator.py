"""Collision Geometry Generator.

Advanced mesh simplification specifically optimized for collision detection.
Supports multiple methods:
- VHACD (Volumetric Hierarchical Approximate Convex Decomposition)
- Primitive fitting (box, sphere, cylinder, capsule)
- Mesh decimation with quality metrics
- Hybrid approaches

Example:
    generator = CollisionGeometryGenerator()
    result = generator.generate(
        visual_mesh,
        method="auto",
        target_complexity="balanced",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SimplificationMethod(Enum):
    """Available mesh simplification methods."""

    AUTO = auto()
    VHACD = auto()  # Convex decomposition
    PRIMITIVES = auto()  # Fit geometric primitives
    DECIMATION = auto()  # Quadric decimation
    CONVEX_HULL = auto()  # Single convex hull
    HYBRID = auto()  # Combine methods


class ComplexityLevel(Enum):
    """Target complexity levels for collision geometry."""

    MINIMAL = auto()  # Fastest simulation, lowest accuracy
    BALANCED = auto()  # Good tradeoff
    ACCURATE = auto()  # Higher accuracy, slower simulation


@dataclass
class PrimitiveFit:
    """Result of fitting a primitive to a mesh region."""

    primitive_type: str  # "box", "sphere", "cylinder", "capsule"
    center: tuple[float, float, float]
    dimensions: tuple[float, ...]  # Type-specific dimensions
    rotation: tuple[float, float, float, float]  # Quaternion
    volume_ratio: float  # Mesh volume / primitive volume
    error_metric: float  # Hausdorff distance


@dataclass
class CollisionGeometryResult:
    """Result of collision geometry generation."""

    success: bool
    method_used: SimplificationMethod
    components: list[Any]  # Mesh or primitive definitions
    original_triangles: int
    final_triangles: int
    reduction_ratio: float
    volume_preservation: float
    hausdorff_distance: float
    primitive_fits: list[PrimitiveFit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class VHACDParameters:
    """Parameters for VHACD convex decomposition."""

    max_hulls: int = 16
    max_vertices_per_hull: int = 64
    resolution: int = 100000
    concavity: float = 0.001
    plane_downsampling: int = 4
    hull_downsampling: int = 4
    alpha: float = 0.05
    beta: float = 0.05
    mode: int = 0  # 0: voxel-based, 1: tetrahedra-based
    min_volume_per_hull: float = 0.0001


class CollisionGeometryGenerator:
    """Generate optimized collision geometry from visual meshes.

    This class provides various methods to create efficient collision
    representations that balance accuracy with simulation performance.
    """

    # Default parameters for each complexity level
    COMPLEXITY_PRESETS = {
        ComplexityLevel.MINIMAL: {
            "max_primitives": 4,
            "max_triangles": 100,
            "max_hulls": 4,
        },
        ComplexityLevel.BALANCED: {
            "max_primitives": 16,
            "max_triangles": 500,
            "max_hulls": 16,
        },
        ComplexityLevel.ACCURATE: {
            "max_primitives": 64,
            "max_triangles": 2000,
            "max_hulls": 64,
        },
    }

    def __init__(self) -> None:
        """Initialize the collision geometry generator."""
        self._trimesh_available = self._check_trimesh()
        self._vhacd_available = self._check_vhacd()

    @staticmethod
    def _check_trimesh() -> bool:
        """Check if trimesh is available."""
        try:
            import trimesh  # noqa: F401

            return True
        except ImportError:
            logger.warning("trimesh not available - mesh processing limited")
            return False

    @staticmethod
    def _check_vhacd() -> bool:
        """Check if VHACD is available."""
        try:
            # VHACD can be accessed via trimesh or pybullet
            import trimesh

            if hasattr(trimesh.interfaces, "vhacd"):
                return True
        except ImportError:
            pass

        try:
            import pybullet  # noqa: F401

            return True
        except ImportError:
            pass

        logger.warning("VHACD not available - using fallback decomposition")
        return False

    def generate(
        self,
        visual_mesh: Any,
        method: str | SimplificationMethod = "auto",
        target_complexity: str | ComplexityLevel = "balanced",
        max_primitives: int | None = None,
        max_triangles: int | None = None,
        max_hulls: int | None = None,
        vhacd_params: VHACDParameters | None = None,
    ) -> CollisionGeometryResult:
        """Generate optimized collision geometry from visual mesh.

        Args:
            visual_mesh: Input mesh (trimesh.Trimesh or path)
            method: Simplification method (auto, vhacd, primitives, decimation)
            target_complexity: Complexity level (minimal, balanced, accurate)
            max_primitives: Override max primitive count
            max_triangles: Override max triangle count
            max_hulls: Override max convex hulls for VHACD
            vhacd_params: Custom VHACD parameters

        Returns:
            CollisionGeometryResult with generated geometry
        """
        # Convert string enum values
        if isinstance(method, str):
            method = SimplificationMethod[method.upper()]
        if isinstance(target_complexity, str):
            target_complexity = ComplexityLevel[target_complexity.upper()]

        # Get complexity preset
        preset = self.COMPLEXITY_PRESETS[target_complexity]
        max_primitives = max_primitives or preset["max_primitives"]
        max_triangles = max_triangles or preset["max_triangles"]
        max_hulls = max_hulls or preset["max_hulls"]

        # Load mesh if path
        mesh = self._load_mesh(visual_mesh)
        if mesh is None:
            return CollisionGeometryResult(
                success=False,
                method_used=method,
                components=[],
                original_triangles=0,
                final_triangles=0,
                reduction_ratio=1.0,
                volume_preservation=0.0,
                hausdorff_distance=float("inf"),
                errors=["Failed to load mesh"],
            )

        original_triangles = len(mesh.faces) if hasattr(mesh, "faces") else 0
        original_volume = mesh.volume if hasattr(mesh, "volume") else 0.0

        # Select method
        if method == SimplificationMethod.AUTO:
            method = self._select_best_method(mesh, max_primitives, max_triangles)

        # Generate collision geometry
        try:
            if method == SimplificationMethod.VHACD:
                result = self._generate_vhacd(mesh, max_hulls, vhacd_params)
            elif method == SimplificationMethod.PRIMITIVES:
                result = self._generate_primitives(mesh, max_primitives)
            elif method == SimplificationMethod.DECIMATION:
                result = self._generate_decimated(mesh, max_triangles)
            elif method == SimplificationMethod.CONVEX_HULL:
                result = self._generate_convex_hull(mesh)
            elif method == SimplificationMethod.HYBRID:
                result = self._generate_hybrid(mesh, max_primitives, max_triangles)
            else:
                result = self._generate_decimated(mesh, max_triangles)

        except (ValueError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"Collision generation failed: {e}")
            return CollisionGeometryResult(
                success=False,
                method_used=method,
                components=[],
                original_triangles=original_triangles,
                final_triangles=0,
                reduction_ratio=1.0,
                volume_preservation=0.0,
                hausdorff_distance=float("inf"),
                errors=[str(e)],
            )

        # Compute metrics
        final_triangles = self._count_triangles(result.components)
        reduction_ratio = 1.0 - (final_triangles / max(original_triangles, 1))
        volume_preservation = self._compute_volume_preservation(
            mesh, result.components, original_volume
        )
        hausdorff = self._compute_hausdorff_distance(mesh, result.components)

        return CollisionGeometryResult(
            success=True,
            method_used=method,
            components=result.components,
            original_triangles=original_triangles,
            final_triangles=final_triangles,
            reduction_ratio=reduction_ratio,
            volume_preservation=volume_preservation,
            hausdorff_distance=hausdorff,
            primitive_fits=result.primitive_fits,
            warnings=result.warnings,
        )

    def _load_mesh(self, mesh_or_path: Any) -> Any:
        """Load mesh from path or return as-is."""
        if not self._trimesh_available:
            return None

        import trimesh

        if isinstance(mesh_or_path, (str, Path)):
            try:
                return trimesh.load(str(mesh_or_path))
            except (ValueError, KeyError, TypeError) as e:
                logger.error(f"Failed to load mesh: {e}")
                return None

        return mesh_or_path

    def _select_best_method(
        self,
        mesh: Any,
        max_primitives: int,
        max_triangles: int,
    ) -> SimplificationMethod:
        """Automatically select the best simplification method.

        Based on mesh complexity and shape characteristics.
        """
        n_faces = len(mesh.faces) if hasattr(mesh, "faces") else 0

        # Simple meshes: single convex hull
        if n_faces < 100:
            return SimplificationMethod.CONVEX_HULL

        # Check if mesh is roughly convex
        if self._is_roughly_convex(mesh):
            return SimplificationMethod.CONVEX_HULL

        # Check if primitives would work well
        if self._primitives_would_fit(mesh, max_primitives):
            return SimplificationMethod.PRIMITIVES

        # Use VHACD for complex meshes if available
        if self._vhacd_available and n_faces > 500:
            return SimplificationMethod.VHACD

        # Default to decimation
        return SimplificationMethod.DECIMATION

    def _is_roughly_convex(self, mesh: Any, threshold: float = 0.95) -> bool:
        """Check if mesh is approximately convex."""
        try:
            convex = mesh.convex_hull
            volume_ratio = mesh.volume / convex.volume
            return bool(volume_ratio > threshold)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return False

    def _primitives_would_fit(self, mesh: Any, max_primitives: int) -> bool:
        """Estimate if primitive fitting would work well."""
        try:
            extents = mesh.extents
            aspect_ratios = extents / extents.min()

            # If aspect ratios are reasonable, primitives might work
            if all(r < 10 for r in aspect_ratios):
                # Estimate number of primitives needed
                volume = mesh.volume
                bounding_volume = np.prod(extents)
                fill_ratio = volume / bounding_volume

                # Higher fill ratio = fewer primitives needed
                estimated_primitives = int(1 / fill_ratio)
                return estimated_primitives <= max_primitives

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

        return False

    def _generate_vhacd(
        self,
        mesh: Any,
        max_hulls: int,
        vhacd_params: VHACDParameters | None,
    ) -> CollisionGeometryResult:
        """Generate collision geometry using VHACD."""
        import trimesh

        params = vhacd_params or VHACDParameters(max_hulls=max_hulls)

        try:
            # Try trimesh VHACD interface
            if hasattr(trimesh.interfaces, "vhacd"):
                convex_hulls = trimesh.interfaces.vhacd.convex_decomposition(
                    mesh,
                    maxhulls=params.max_hulls,
                    resolution=params.resolution,
                )
            else:
                # Fallback to pybullet VHACD
                convex_hulls = self._vhacd_pybullet(mesh, params)

            if not convex_hulls:
                raise ValueError("VHACD produced no output")

            return CollisionGeometryResult(
                success=True,
                method_used=SimplificationMethod.VHACD,
                components=list(convex_hulls),
                original_triangles=len(mesh.faces),
                final_triangles=sum(len(h.faces) for h in convex_hulls),
                reduction_ratio=0.0,
                volume_preservation=1.0,
                hausdorff_distance=0.0,
            )

        except (ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning(f"VHACD failed, falling back to convex hull: {e}")
            return self._generate_convex_hull(mesh)

    def _vhacd_pybullet(
        self,
        mesh: Any,
        params: VHACDParameters,
    ) -> list[Any]:
        """Use pybullet for VHACD decomposition."""
        import os
        import tempfile

        import pybullet as p
        import trimesh

        # Export mesh temporarily
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.obj")
            output_path = os.path.join(tmpdir, "output.obj")

            mesh.export(input_path)

            # Run VHACD
            p.vhacd(
                input_path,
                output_path,
                os.path.join(tmpdir, "log.txt"),
                maxNumVerticesPerCH=params.max_vertices_per_hull,
                resolution=params.resolution,
                concavity=params.concavity,
            )

            # Load result
            result = trimesh.load(output_path)
            if isinstance(result, trimesh.Scene):
                return list(result.geometry.values())
            return [result]

    def _generate_primitives(
        self,
        mesh: Any,
        max_primitives: int,
    ) -> CollisionGeometryResult:
        """Generate collision geometry using fitted primitives."""
        primitives = []
        primitive_fits = []

        # Try to fit primitives
        fits = [
            self._fit_box(mesh),
            self._fit_sphere(mesh),
            self._fit_cylinder(mesh),
        ]

        # Sort by error metric
        fits.sort(key=lambda f: f.error_metric)

        # Use best fit if good enough
        best_fit = fits[0]
        if best_fit.volume_ratio > 0.7:
            primitives.append(self._primitive_to_mesh(best_fit))
            primitive_fits.append(best_fit)
        else:
            # Fallback to convex hull
            primitives.append(mesh.convex_hull)

        return CollisionGeometryResult(
            success=True,
            method_used=SimplificationMethod.PRIMITIVES,
            components=primitives,
            original_triangles=len(mesh.faces),
            final_triangles=sum(
                len(p.faces) if hasattr(p, "faces") else 0 for p in primitives
            ),
            reduction_ratio=0.0,
            volume_preservation=1.0,
            hausdorff_distance=0.0,
            primitive_fits=primitive_fits,
        )

    def _fit_box(self, mesh: Any) -> PrimitiveFit:
        """Fit an oriented bounding box to the mesh."""
        try:
            obb = mesh.bounding_box_oriented
            center = tuple(obb.centroid.tolist())
            extents = tuple(obb.primitive.extents.tolist())
            transform = obb.primitive.transform

            # Extract rotation as quaternion
            from scipy.spatial.transform import Rotation

            rot = Rotation.from_matrix(transform[:3, :3])
            quat = tuple(rot.as_quat().tolist())

            volume_ratio = mesh.volume / obb.volume
            # Approximate error as fraction not covered
            error = 1.0 - volume_ratio

            return PrimitiveFit(
                primitive_type="box",
                center=center,
                dimensions=extents,
                rotation=quat,
                volume_ratio=volume_ratio,
                error_metric=error,
            )
        except ImportError as e:
            logger.warning(f"Box fitting failed: {e}")
            return PrimitiveFit(
                primitive_type="box",
                center=(0, 0, 0),
                dimensions=(1, 1, 1),
                rotation=(0, 0, 0, 1),
                volume_ratio=0.0,
                error_metric=float("inf"),
            )

    def _fit_sphere(self, mesh: Any) -> PrimitiveFit:
        """Fit a bounding sphere to the mesh."""
        try:
            # Use mesh centroid and max distance to vertex
            center = tuple(mesh.centroid.tolist())
            vertices = mesh.vertices - mesh.centroid
            radius = float(np.max(np.linalg.norm(vertices, axis=1)))

            sphere_volume = (4 / 3) * np.pi * radius**3
            volume_ratio = mesh.volume / sphere_volume

            return PrimitiveFit(
                primitive_type="sphere",
                center=center,
                dimensions=(radius,),
                rotation=(0, 0, 0, 1),
                volume_ratio=volume_ratio,
                error_metric=1.0 - volume_ratio,
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"Sphere fitting failed: {e}")
            return PrimitiveFit(
                primitive_type="sphere",
                center=(0, 0, 0),
                dimensions=(1,),
                rotation=(0, 0, 0, 1),
                volume_ratio=0.0,
                error_metric=float("inf"),
            )

    def _fit_cylinder(self, mesh: Any) -> PrimitiveFit:
        """Fit a cylinder to the mesh."""
        try:
            # Use OBB to determine primary axis
            obb = mesh.bounding_box_oriented
            extents = obb.primitive.extents

            # Longest axis is cylinder axis
            axis_idx = np.argmax(extents)
            height = extents[axis_idx]

            # Radius from the other two dimensions
            other_dims = [extents[i] for i in range(3) if i != axis_idx]
            radius = max(other_dims) / 2

            cylinder_volume = np.pi * radius**2 * height
            volume_ratio = mesh.volume / cylinder_volume

            center = tuple(obb.centroid.tolist())
            transform = obb.primitive.transform
            from scipy.spatial.transform import Rotation

            rot = Rotation.from_matrix(transform[:3, :3])
            quat = tuple(rot.as_quat().tolist())

            return PrimitiveFit(
                primitive_type="cylinder",
                center=center,
                dimensions=(radius, height),
                rotation=quat,
                volume_ratio=volume_ratio,
                error_metric=1.0 - volume_ratio,
            )
        except ImportError as e:
            logger.warning(f"Cylinder fitting failed: {e}")
            return PrimitiveFit(
                primitive_type="cylinder",
                center=(0, 0, 0),
                dimensions=(1, 1),
                rotation=(0, 0, 0, 1),
                volume_ratio=0.0,
                error_metric=float("inf"),
            )

    def _primitive_to_mesh(self, fit: PrimitiveFit) -> Any:
        """Convert primitive fit to mesh."""
        import trimesh

        if fit.primitive_type == "box":
            mesh = trimesh.creation.box(extents=fit.dimensions)
        elif fit.primitive_type == "sphere":
            mesh = trimesh.creation.icosphere(radius=fit.dimensions[0])
        elif fit.primitive_type == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=fit.dimensions[0],
                height=fit.dimensions[1],
            )
        else:
            mesh = trimesh.creation.box(extents=(1, 1, 1))

        # Apply transform
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_quat(fit.rotation)
        transform = np.eye(4)
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = fit.center
        mesh.apply_transform(transform)

        return mesh

    def _generate_decimated(
        self,
        mesh: Any,
        max_triangles: int,
    ) -> CollisionGeometryResult:
        """Generate collision geometry via mesh decimation."""
        if len(mesh.faces) <= max_triangles:
            return CollisionGeometryResult(
                success=True,
                method_used=SimplificationMethod.DECIMATION,
                components=[mesh.copy()],
                original_triangles=len(mesh.faces),
                final_triangles=len(mesh.faces),
                reduction_ratio=0.0,
                volume_preservation=1.0,
                hausdorff_distance=0.0,
            )

        try:
            # Try quadric decimation
            simplified = mesh.simplify_quadric_decimation(max_triangles)
        except (ValueError, RuntimeError, IndexError):
            # Fallback to vertex clustering
            try:
                reduction = max_triangles / len(mesh.faces)
                pitch = mesh.extents.max() * (1 - reduction) / 10
                voxelized = mesh.voxelized(pitch)
                simplified = voxelized.marching_cubes
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                simplified = mesh.copy()

        return CollisionGeometryResult(
            success=True,
            method_used=SimplificationMethod.DECIMATION,
            components=[simplified],
            original_triangles=len(mesh.faces),
            final_triangles=len(simplified.faces),
            reduction_ratio=1.0 - len(simplified.faces) / len(mesh.faces),
            volume_preservation=(
                simplified.volume / mesh.volume if mesh.volume > 0 else 1.0
            ),
            hausdorff_distance=0.0,
        )

    def _generate_convex_hull(self, mesh: Any) -> CollisionGeometryResult:
        """Generate single convex hull."""
        hull = mesh.convex_hull

        return CollisionGeometryResult(
            success=True,
            method_used=SimplificationMethod.CONVEX_HULL,
            components=[hull],
            original_triangles=len(mesh.faces),
            final_triangles=len(hull.faces),
            reduction_ratio=1.0 - len(hull.faces) / len(mesh.faces),
            volume_preservation=mesh.volume / hull.volume if hull.volume > 0 else 1.0,
            hausdorff_distance=0.0,
        )

    def _generate_hybrid(
        self,
        mesh: Any,
        max_primitives: int,
        max_triangles: int,
    ) -> CollisionGeometryResult:
        """Combine primitives and mesh decimation."""
        # Start with primitive fitting
        prim_result = self._generate_primitives(mesh, max_primitives)

        # If primitives fit well, use them
        if (
            prim_result.primitive_fits
            and prim_result.primitive_fits[0].volume_ratio > 0.8
        ):
            return prim_result

        # Otherwise use decimation for remaining detail
        return self._generate_decimated(mesh, max_triangles)

    def _count_triangles(self, components: list[Any]) -> int:
        """Count total triangles in components."""
        total = 0
        for comp in components:
            if hasattr(comp, "faces"):
                total += len(comp.faces)
        return total

    def _compute_volume_preservation(
        self,
        original: Any,
        components: list[Any],
        original_volume: float,
    ) -> float:
        """Compute volume preservation ratio."""
        if original_volume <= 0:
            return 1.0

        total_volume = 0.0
        for comp in components:
            if hasattr(comp, "volume"):
                total_volume += comp.volume

        return min(total_volume / original_volume, 1.0)

    def _compute_hausdorff_distance(
        self,
        original: Any,
        components: list[Any],
    ) -> float:
        """Compute approximate Hausdorff distance."""
        # Simplified: sample points and compute max distance
        try:
            original_points = original.sample(1000)
            max_dist = 0.0

            for comp in components:
                if hasattr(comp, "nearest"):
                    _, distances, _ = comp.nearest.on_surface(original_points)
                    max_dist = max(max_dist, distances.max())

            return float(max_dist)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return float("inf")


__all__ = [
    "CollisionGeometryGenerator",
    "CollisionGeometryResult",
    "SimplificationMethod",
    "ComplexityLevel",
    "VHACDParameters",
    "PrimitiveFit",
]
