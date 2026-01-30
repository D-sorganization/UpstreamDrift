"""
Mesh processing utilities for humanoid character builder.

This module provides mesh loading, segmentation, simplification,
and export capabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MeshSegmentResult:
    """Result of mesh segmentation operation."""

    segment_name: str
    mesh_path: Path | None
    vertex_count: int
    face_count: int
    volume: float
    center_of_mass: tuple[float, float, float]
    bounding_box_min: tuple[float, float, float]
    bounding_box_max: tuple[float, float, float]
    is_watertight: bool
    success: bool
    error_message: str | None = None


@dataclass
class MeshExportConfig:
    """Configuration for mesh export."""

    # Output format
    format: str = "stl"  # stl, obj, ply, dae

    # Simplification
    simplify: bool = False
    target_faces: int | None = None
    simplification_ratio: float = 0.5  # Keep this fraction of faces

    # Processing
    repair: bool = True
    center_at_origin: bool = False
    scale: float = 1.0

    # Collision mesh options
    generate_collision: bool = True
    collision_convex_hull: bool = True
    collision_simplification: float = 0.3


class MeshProcessor:
    """
    Process and manipulate mesh geometry.

    Provides utilities for:
    - Loading meshes from various formats
    - Segmenting meshes by vertex groups
    - Simplifying meshes for collision
    - Generating convex hulls
    - Exporting to various formats
    """

    SUPPORTED_FORMATS = {"stl", "obj", "ply", "dae", "glb", "gltf", "off"}

    def __init__(self):
        """Initialize the mesh processor."""
        self._trimesh_available = self._check_trimesh()

    def _check_trimesh(self) -> bool:
        """Check if trimesh is available."""
        try:
            import trimesh  # noqa: F401

            return True
        except ImportError:
            logger.warning("trimesh not available for mesh processing")
            return False

    def load_mesh(self, mesh_path: Path | str) -> Any:
        """
        Load a mesh from file.

        Args:
            mesh_path: Path to mesh file

        Returns:
            trimesh.Trimesh object

        Raises:
            ImportError: If trimesh not available
            FileNotFoundError: If file doesn't exist
        """
        if not self._trimesh_available:
            raise ImportError("trimesh is required for mesh processing")

        import trimesh

        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        mesh = trimesh.load(str(mesh_path))

        # Handle scene objects
        if isinstance(mesh, trimesh.Scene):
            meshes = list(mesh.geometry.values())
            if not meshes:
                raise ValueError("Scene contains no geometry")
            mesh = trimesh.util.concatenate(meshes)

        return mesh

    def segment_by_vertex_groups(
        self,
        mesh: Any,
        vertex_groups: dict[str, NDArray[np.int64]],
        output_dir: Path,
        config: MeshExportConfig | None = None,
    ) -> dict[str, MeshSegmentResult]:
        """
        Segment a mesh by vertex groups.

        Args:
            mesh: trimesh.Trimesh object
            vertex_groups: Dict mapping group name to vertex indices
            output_dir: Directory for output meshes
            config: Export configuration

        Returns:
            Dict mapping segment name to MeshSegmentResult
        """
        if not self._trimesh_available:
            raise ImportError("trimesh is required")

        import trimesh

        config = config or MeshExportConfig()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for segment_name, vertex_indices in vertex_groups.items():
            try:
                # Extract faces that use these vertices
                vertex_set = set(vertex_indices)
                face_mask = np.array(
                    [
                        all(v in vertex_set for v in face)
                        for face in mesh.faces
                    ]
                )

                if not np.any(face_mask):
                    results[segment_name] = MeshSegmentResult(
                        segment_name=segment_name,
                        mesh_path=None,
                        vertex_count=0,
                        face_count=0,
                        volume=0.0,
                        center_of_mass=(0.0, 0.0, 0.0),
                        bounding_box_min=(0.0, 0.0, 0.0),
                        bounding_box_max=(0.0, 0.0, 0.0),
                        is_watertight=False,
                        success=False,
                        error_message="No faces found for vertex group",
                    )
                    continue

                # Create submesh
                submesh = mesh.submesh([face_mask], append=True)

                # Process mesh
                if config.repair:
                    trimesh.repair.fill_holes(submesh)
                    submesh.fix_normals()

                if config.simplify and config.target_faces:
                    submesh = self._simplify_mesh(submesh, config.target_faces)

                if config.center_at_origin:
                    submesh.vertices -= submesh.center_mass

                if config.scale != 1.0:
                    submesh.vertices *= config.scale

                # Export
                output_path = output_dir / f"{segment_name}.{config.format}"
                submesh.export(str(output_path))

                # Get properties
                bbox = submesh.bounding_box.bounds
                com = submesh.center_mass if submesh.is_watertight else submesh.centroid

                results[segment_name] = MeshSegmentResult(
                    segment_name=segment_name,
                    mesh_path=output_path,
                    vertex_count=len(submesh.vertices),
                    face_count=len(submesh.faces),
                    volume=float(submesh.volume) if submesh.is_watertight else 0.0,
                    center_of_mass=(float(com[0]), float(com[1]), float(com[2])),
                    bounding_box_min=(
                        float(bbox[0, 0]),
                        float(bbox[0, 1]),
                        float(bbox[0, 2]),
                    ),
                    bounding_box_max=(
                        float(bbox[1, 0]),
                        float(bbox[1, 1]),
                        float(bbox[1, 2]),
                    ),
                    is_watertight=submesh.is_watertight,
                    success=True,
                )

            except Exception as e:
                logger.error(f"Failed to segment {segment_name}: {e}")
                results[segment_name] = MeshSegmentResult(
                    segment_name=segment_name,
                    mesh_path=None,
                    vertex_count=0,
                    face_count=0,
                    volume=0.0,
                    center_of_mass=(0.0, 0.0, 0.0),
                    bounding_box_min=(0.0, 0.0, 0.0),
                    bounding_box_max=(0.0, 0.0, 0.0),
                    is_watertight=False,
                    success=False,
                    error_message=str(e),
                )

        return results

    def segment_by_bounding_boxes(
        self,
        mesh: Any,
        segment_boxes: dict[str, tuple[NDArray, NDArray]],
        output_dir: Path,
        config: MeshExportConfig | None = None,
    ) -> dict[str, MeshSegmentResult]:
        """
        Segment a mesh by bounding boxes.

        Args:
            mesh: trimesh.Trimesh object
            segment_boxes: Dict mapping name to (min_corner, max_corner)
            output_dir: Directory for output meshes
            config: Export configuration

        Returns:
            Dict mapping segment name to MeshSegmentResult
        """
        if not self._trimesh_available:
            raise ImportError("trimesh is required")

        # Convert bounding boxes to vertex groups
        vertex_groups = {}
        for name, (box_min, box_max) in segment_boxes.items():
            box_min = np.asarray(box_min)
            box_max = np.asarray(box_max)

            # Find vertices inside box
            inside = np.all(
                (mesh.vertices >= box_min) & (mesh.vertices <= box_max), axis=1
            )
            vertex_groups[name] = np.where(inside)[0]

        return self.segment_by_vertex_groups(mesh, vertex_groups, output_dir, config)

    def create_convex_hull(
        self, mesh: Any, output_path: Path | None = None
    ) -> Any:
        """
        Create convex hull from mesh.

        Args:
            mesh: trimesh.Trimesh object
            output_path: Optional path to save hull

        Returns:
            trimesh.Trimesh convex hull
        """
        if not self._trimesh_available:
            raise ImportError("trimesh is required")

        hull = mesh.convex_hull

        if output_path:
            hull.export(str(output_path))

        return hull

    def simplify_mesh(
        self,
        mesh: Any,
        target_faces: int | None = None,
        ratio: float | None = None,
        output_path: Path | None = None,
    ) -> Any:
        """
        Simplify mesh by reducing face count.

        Args:
            mesh: trimesh.Trimesh object
            target_faces: Target number of faces
            ratio: Fraction of faces to keep (alternative to target_faces)
            output_path: Optional path to save simplified mesh

        Returns:
            Simplified trimesh.Trimesh
        """
        if not self._trimesh_available:
            raise ImportError("trimesh is required")

        if target_faces is None and ratio is not None:
            target_faces = int(len(mesh.faces) * ratio)

        simplified = self._simplify_mesh(mesh, target_faces or len(mesh.faces) // 2)

        if output_path:
            simplified.export(str(output_path))

        return simplified

    def _simplify_mesh(self, mesh: Any, target_faces: int) -> Any:
        """Internal mesh simplification."""
        import trimesh

        # Try quadric decimation if available
        try:
            simplified = mesh.simplify_quadric_decimation(target_faces)
            return simplified
        except Exception:
            pass

        # Fallback: vertex clustering
        try:
            # Estimate cluster size to achieve target face count
            current_faces = len(mesh.faces)
            reduction = target_faces / current_faces
            pitch = mesh.extents.max() * (1 - reduction) / 10

            simplified = trimesh.voxel.ops.points_to_marching_cubes(
                mesh.vertices, pitch=pitch
            )
            return simplified
        except Exception:
            # Return original if simplification fails
            logger.warning("Mesh simplification failed, returning original")
            return mesh

    def create_collision_mesh(
        self,
        visual_mesh: Any,
        config: MeshExportConfig | None = None,
    ) -> Any:
        """
        Create simplified collision mesh from visual mesh.

        Args:
            visual_mesh: trimesh.Trimesh visual mesh
            config: Export configuration

        Returns:
            Simplified collision mesh
        """
        config = config or MeshExportConfig()

        if config.collision_convex_hull:
            return self.create_convex_hull(visual_mesh)
        else:
            target_faces = int(len(visual_mesh.faces) * config.collision_simplification)
            return self.simplify_mesh(visual_mesh, target_faces=target_faces)

    def export_mesh(
        self,
        mesh: Any,
        output_path: Path,
        config: MeshExportConfig | None = None,
    ) -> Path:
        """
        Export mesh to file.

        Args:
            mesh: trimesh.Trimesh object
            output_path: Output file path
            config: Export configuration

        Returns:
            Path to exported file
        """
        config = config or MeshExportConfig()
        output_path = Path(output_path)

        # Ensure format matches extension
        if output_path.suffix[1:].lower() != config.format.lower():
            output_path = output_path.with_suffix(f".{config.format}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))

        return output_path

    def get_mesh_info(self, mesh: Any) -> dict[str, Any]:
        """
        Get information about a mesh.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            Dict with mesh properties
        """
        bbox = mesh.bounding_box.bounds

        return {
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "is_watertight": mesh.is_watertight,
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "surface_area": float(mesh.area),
            "center_mass": (
                tuple(mesh.center_mass) if mesh.is_watertight else tuple(mesh.centroid)
            ),
            "bounding_box_min": (
                float(bbox[0, 0]),
                float(bbox[0, 1]),
                float(bbox[0, 2]),
            ),
            "bounding_box_max": (
                float(bbox[1, 0]),
                float(bbox[1, 1]),
                float(bbox[1, 2]),
            ),
            "extents": tuple(mesh.extents),
        }


@dataclass
class PrimitiveMeshGenerator:
    """Generate primitive meshes programmatically."""

    def create_box(
        self, size_x: float, size_y: float, size_z: float
    ) -> Any:
        """Create a box mesh."""
        if not self._check_trimesh():
            raise ImportError("trimesh is required")

        import trimesh

        return trimesh.creation.box(extents=(size_x, size_y, size_z))

    def create_cylinder(
        self, radius: float, height: float, sections: int = 32
    ) -> Any:
        """Create a cylinder mesh."""
        if not self._check_trimesh():
            raise ImportError("trimesh is required")

        import trimesh

        return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)

    def create_sphere(self, radius: float, subdivisions: int = 3) -> Any:
        """Create a sphere mesh."""
        if not self._check_trimesh():
            raise ImportError("trimesh is required")

        import trimesh

        return trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)

    def create_capsule(
        self, radius: float, height: float, sections: int = 32
    ) -> Any:
        """Create a capsule mesh."""
        if not self._check_trimesh():
            raise ImportError("trimesh is required")

        import trimesh

        return trimesh.creation.capsule(radius=radius, height=height, count=[sections, sections])

    def _check_trimesh(self) -> bool:
        """Check if trimesh is available."""
        try:
            import trimesh  # noqa: F401

            return True
        except ImportError:
            return False
