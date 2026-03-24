"""
Mesh generation interfaces for humanoid character builder.

This module defines interfaces for mesh generation backends
(MakeHuman, SMPL, etc.) and provides a factory for creating
mesh generators.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from humanoid_character_builder.core.body_parameters import BodyParameters, GenderModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency availability flags (mock-patchable in tests)
# ---------------------------------------------------------------------------

try:
    import smplx as _smplx_module  # type: ignore[import-untyped]

    SMPLX_AVAILABLE = True
except ImportError:
    _smplx_module = None  # type: ignore[assignment]
    SMPLX_AVAILABLE = False

try:
    import trimesh as _trimesh_module  # type: ignore[import-untyped]

    TRIMESH_AVAILABLE = True
except ImportError:
    _trimesh_module = None  # type: ignore[assignment]
    TRIMESH_AVAILABLE = False


class MeshGeneratorBackend(Enum):
    """Available mesh generation backends."""

    PRIMITIVE = "primitive"  # Generate primitive shapes (built-in)
    MAKEHUMAN = "makehuman"  # MakeHuman integration
    SMPLX = "smplx"  # SMPL-X body model
    CUSTOM = "custom"  # Custom mesh provider


@dataclass
class GeneratedMeshResult:
    """Result of mesh generation."""

    # Whether generation was successful
    success: bool

    # Path to generated mesh files (segment name -> path)
    mesh_paths: dict[str, Path] = field(default_factory=dict)

    # Path to collision mesh files
    collision_paths: dict[str, Path] = field(default_factory=dict)

    # Path to texture files
    texture_paths: dict[str, Path] = field(default_factory=dict)

    # Vertex group mapping (for segmentation)
    vertex_groups: dict[str, list[int]] = field(default_factory=dict)

    # Error message if failed
    error_message: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class MeshGeneratorInterface(ABC):
    """
    Abstract interface for mesh generation backends.

    Implement this interface to add new mesh generation sources
    (MakeHuman, SMPL, etc.).
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available (installed, configured)."""
        ...

    @abstractmethod
    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """
        Generate meshes for the given body parameters.

        Args:
            params: Body parameters
            output_dir: Directory to write mesh files
            **kwargs: Backend-specific options

        Returns:
            GeneratedMeshResult with paths to generated files
        """
        ...

    @abstractmethod
    def get_supported_segments(self) -> list[str]:
        """Return list of segment names this backend can generate."""
        ...


class PrimitiveMeshGenerator(MeshGeneratorInterface):
    """
    Generate primitive shape meshes (built-in, no external dependencies).

    This is the fallback generator that creates simple geometric shapes
    for each body segment.
    """

    @property
    def backend_name(self) -> str:
        return "primitive"

    @property
    def is_available(self) -> bool:
        # Check if trimesh is available for mesh creation
        try:
            import trimesh  # noqa: F401

            return True
        except ImportError:
            return False

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate primitive meshes for body segments."""
        assert params is not None, "params must be provided"
        assert params is not None, "params must be provided"
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message="trimesh not available for primitive mesh generation",
            )

        import trimesh
        from humanoid_character_builder.core.anthropometry import (
            estimate_segment_dimensions,
        )
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
            GeometryType,
        )

        output_dir = Path(output_dir)
        visual_dir = output_dir / "visual"
        collision_dir = output_dir / "collision"
        visual_dir.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(parents=True, exist_ok=True)

        mesh_paths = {}
        collision_paths = {}

        gender_factor = params.get_effective_gender_factor()
        dimensions = estimate_segment_dimensions(params.height_m, gender_factor)

        for segment_name, segment_def in HUMANOID_SEGMENTS.items():
            try:
                dims = dimensions.get(
                    segment_name, {"length": 0.1, "width": 0.05, "depth": 0.05}
                )
                length = dims["length"]
                width = dims["width"]
                depth = dims["depth"]

                # Create mesh based on geometry type
                geom_type = segment_def.visual_geometry.geometry_type

                if geom_type == GeometryType.SPHERE:
                    mesh = trimesh.creation.icosphere(radius=length / 2, subdivisions=2)
                elif geom_type == GeometryType.CYLINDER:
                    radius = (width + depth) / 4
                    mesh = trimesh.creation.cylinder(
                        radius=radius, height=length, sections=16
                    )
                elif geom_type == GeometryType.CAPSULE:
                    radius = (width + depth) / 4
                    cyl_height = max(0.01, length - 2 * radius)
                    mesh = trimesh.creation.capsule(
                        radius=radius, height=cyl_height, count=[8, 8]
                    )
                else:  # BOX or default
                    mesh = trimesh.creation.box(extents=(width, depth, length))

                # Export visual mesh
                visual_path = visual_dir / f"{segment_name}.stl"
                mesh.export(str(visual_path))
                mesh_paths[segment_name] = visual_path

                # Create simplified collision mesh (convex hull)
                collision_mesh = mesh.convex_hull
                collision_path = collision_dir / f"{segment_name}.stl"
                collision_mesh.export(str(collision_path))
                collision_paths[segment_name] = collision_path

            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.warning(f"Failed to generate mesh for {segment_name}: {e}")

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            metadata={"backend": "primitive"},
        )

    def get_supported_segments(self) -> list[str]:
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())


class MakeHumanMeshGenerator(MeshGeneratorInterface):
    """
    Generate meshes using MakeHuman.

    This is a placeholder for future MakeHuman integration.
    MakeHuman provides high-quality, customizable human meshes
    with proper vertex groups for segmentation.
    """

    def __init__(self, makehuman_path: Path | str | None = None):
        """
        Initialize MakeHuman generator.

        Args:
            makehuman_path: Path to MakeHuman installation
        """
        self.makehuman_path = Path(makehuman_path) if makehuman_path else None

    @property
    def backend_name(self) -> str:
        return "makehuman"

    @property
    def is_available(self) -> bool:
        # Check if MakeHuman is installed
        if self.makehuman_path and self.makehuman_path.exists():
            return True

        # Try to find MakeHuman in common locations
        common_paths = [
            Path("/usr/share/makehuman"),
            Path.home() / "makehuman",
            Path.home() / ".makehuman",
        ]
        for path in common_paths:
            if path.exists():
                self.makehuman_path = path
                return True

        return False

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using MakeHuman.

        Uses MakeHuman's Python API when available, or falls back to
        loading pre-made MakeHuman exports with vertex group segmentation.
        """
        assert params is not None, "params must be provided"
        assert params is not None, "params must be provided"
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message="MakeHuman not found. Please install MakeHuman or provide path.",
            )

        output_dir = Path(output_dir)
        visual_dir = output_dir / "visual"
        collision_dir = output_dir / "collision"
        visual_dir.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(parents=True, exist_ok=True)

        modifiers = self._convert_params_to_makehuman(params)

        # Try the scripted MakeHuman API
        try:
            return self._generate_via_api(
                params, modifiers, visual_dir, collision_dir, **kwargs
            )
        except (
            ValueError,
            ZeroDivisionError,
            OverflowError,
            TypeError,
            RuntimeError,
        ) as e:
            logger.warning("MakeHuman API generation failed: %s", e)
            return GeneratedMeshResult(
                success=False,
                error_message=f"MakeHuman generation failed: {e}",
            )

    def _generate_via_api(
        self,
        params: BodyParameters,
        modifiers: dict[str, float],
        visual_dir: Path,
        collision_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using MakeHuman scripted mode.

        Writes a Python script via _build_mh_script and runs it via
        _run_makehuman_script, then loads the resulting OBJ and segments it.
        """
        assert params is not None, "params must be provided"
        assert params is not None, "params must be provided"
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            script_path = tmp / "mh_generate.py"
            body_obj = tmp / "body.obj"
            groups_json = tmp / "groups.json"

            script_content = self._build_mh_script(modifiers, body_obj, groups_json)
            script_path.write_text(script_content, encoding="utf-8")

            success = self._run_makehuman_script(script_path)
            if not success:
                raise RuntimeError("MakeHuman script execution failed")

            if not body_obj.exists():
                raise RuntimeError(f"MakeHuman did not produce OBJ: {body_obj}")

            # Load vertex groups if available
            vertex_groups: dict[str, list[int]] = {}
            if groups_json.exists():
                with open(groups_json, encoding="utf-8") as fh:
                    vertex_groups = json.load(fh)

            # Parse OBJ
            vertices, faces = self._parse_obj_file(body_obj)
            if len(vertices) == 0:
                raise RuntimeError("Parsed OBJ has no vertices")

            # Segment the mesh and export per-body-part STLs
            if not TRIMESH_AVAILABLE:
                raise RuntimeError("trimesh required for mesh segmentation")

            mesh = _trimesh_module.Trimesh(vertices=vertices, faces=faces)  # type: ignore[union-attr]

            from humanoid_character_builder.core.segment_definitions import (
                HUMANOID_SEGMENTS,
            )

            mesh_paths: dict[str, Path] = {}
            collision_paths: dict[str, Path] = {}

            all_vertices = (
                np.array(mesh.vertices) if hasattr(mesh, "vertices") else vertices
            )
            all_faces = np.array(mesh.faces) if hasattr(mesh, "faces") else faces

            for mh_group, segment_name in self.MH_VERTEX_GROUP_MAP.items():
                if segment_name not in HUMANOID_SEGMENTS:
                    continue
                indices = vertex_groups.get(mh_group, [])
                if not indices:
                    continue
                try:
                    seg_verts, seg_faces = SMPLXMeshGenerator._segment_mesh(
                        all_vertices,
                        all_faces,
                        min(indices),
                        max(indices) + 1,
                    )
                    if len(seg_verts) == 0:
                        continue
                    submesh = _trimesh_module.Trimesh(  # type: ignore[union-attr]
                        vertices=seg_verts, faces=seg_faces
                    )
                    vpath = visual_dir / f"{segment_name}.stl"
                    submesh.export(str(vpath))
                    mesh_paths[segment_name] = vpath
                    cpath = collision_dir / f"{segment_name}.stl"
                    submesh.convex_hull.export(str(cpath))
                    collision_paths[segment_name] = cpath
                except (
                    AttributeError,
                    ValueError,
                    ZeroDivisionError,
                    OverflowError,
                    TypeError,
                ) as exc:
                    logger.warning("Failed to segment %s: %s", segment_name, exc)

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            vertex_groups=vertex_groups,
            metadata={"backend": "makehuman"},
        )

    def _generate_from_presets(
        self,
        params: BodyParameters,
        visual_dir: Path,
        collision_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Load pre-exported MakeHuman mesh based on parameters."""
        try:
            import trimesh
        except ImportError as err:
            raise RuntimeError("trimesh required for mesh processing") from err

        # Look for pre-exported mesh files in MakeHuman data directory
        if self.makehuman_path is None:
            raise RuntimeError("MakeHuman path not configured")
        presets_dir = self.makehuman_path / "data" / "exports"
        if not presets_dir.exists():
            presets_dir = self.makehuman_path / "exports"

        # Select preset based on build type
        preset_name = params.build_type or "average"
        gender = "male" if params.get_effective_gender_factor() > 0.5 else "female"
        preset_file = presets_dir / f"{gender}_{preset_name}.obj"

        if not preset_file.exists():
            # Try default
            preset_file = presets_dir / f"{gender}_average.obj"

        if not preset_file.exists():
            raise FileNotFoundError(f"No MakeHuman preset found: {preset_file}")

        # Load and segment the mesh
        mesh = trimesh.load(str(preset_file))

        # Scale to target height
        current_height = mesh.bounds[1][2] - mesh.bounds[0][2]
        scale_factor = params.height_m / current_height
        mesh.apply_scale(scale_factor)

        return self._segment_mesh_from_groups(mesh, visual_dir, collision_dir, params)

    def _segment_mesh(
        self, visual_dir: Path, collision_dir: Path
    ) -> GeneratedMeshResult:
        """Segment a generated mesh by vertex groups."""
        assert visual_dir is not None, "visual_dir must be provided"
        assert visual_dir is not None, "visual_dir must be provided"
        try:
            import trimesh
        except ImportError as err:
            raise RuntimeError("trimesh required for mesh segmentation") from err

        obj_file = visual_dir / "humanoid.obj"
        if not obj_file.exists():
            raise FileNotFoundError(f"Generated mesh not found: {obj_file}")

        mesh = trimesh.load(str(obj_file))

        # Get vertex groups from OBJ file
        vertex_groups = self._parse_obj_vertex_groups(obj_file)

        return self._segment_mesh_from_groups(
            mesh, visual_dir, collision_dir, vertex_groups=vertex_groups
        )

    def _segment_mesh_from_groups(
        self,
        mesh: Any,
        visual_dir: Path,
        collision_dir: Path,
        params: BodyParameters | None = None,
        vertex_groups: dict[str, list[int]] | None = None,
    ) -> GeneratedMeshResult:
        """Segment mesh into body parts using vertex groups or geometry."""
        assert visual_dir is not None, "visual_dir must be provided"
        assert visual_dir is not None, "visual_dir must be provided"
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        # Map MakeHuman vertex groups to our segment names
        group_mapping = {
            "head": "head",
            "neck": "neck",
            "torso": "torso",
            "upper_torso": "torso",
            "lower_torso": "pelvis",
            "pelvis": "pelvis",
            "left_upper_arm": "left_upper_arm",
            "right_upper_arm": "right_upper_arm",
            "left_forearm": "left_forearm",
            "right_forearm": "right_forearm",
            "left_hand": "left_hand",
            "right_hand": "right_hand",
            "left_thigh": "left_thigh",
            "right_thigh": "right_thigh",
            "left_shin": "left_shin",
            "right_shin": "right_shin",
            "left_foot": "left_foot",
            "right_foot": "right_foot",
        }

        if vertex_groups:
            mesh_paths, collision_paths = self._segment_by_vertex_groups(
                mesh,
                visual_dir,
                collision_dir,
                vertex_groups,
                group_mapping,
                HUMANOID_SEGMENTS,
            )
        else:
            mesh_paths, collision_paths = self._segment_by_geometry(
                mesh,
                visual_dir,
                collision_dir,
                HUMANOID_SEGMENTS,
            )

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            vertex_groups=vertex_groups or {},
            metadata={"backend": "makehuman"},
        )

    @staticmethod
    def _segment_by_vertex_groups(
        mesh: Any,
        visual_dir: Path,
        collision_dir: Path,
        vertex_groups: dict[str, list[int]],
        group_mapping: dict[str, str],
        valid_segments: Any,
    ) -> tuple[dict[str, Path], dict[str, Path]]:
        """Segment mesh using vertex group indices."""
        assert visual_dir is not None, "visual_dir must be provided"
        assert visual_dir is not None, "visual_dir must be provided"
        mesh_paths: dict[str, Path] = {}
        collision_paths: dict[str, Path] = {}

        for group_name, vertex_indices in vertex_groups.items():
            segment_name = group_mapping.get(group_name.lower())
            if segment_name and segment_name in valid_segments:
                try:
                    face_mask = mesh.faces_sparse.rows[vertex_indices].indices
                    submesh = mesh.submesh([face_mask], append=True)

                    visual_path = visual_dir / f"{segment_name}.stl"
                    submesh.export(str(visual_path))
                    mesh_paths[segment_name] = visual_path

                    collision_mesh = submesh.convex_hull
                    collision_path = collision_dir / f"{segment_name}.stl"
                    collision_mesh.export(str(collision_path))
                    collision_paths[segment_name] = collision_path
                except (
                    ValueError,
                    ZeroDivisionError,
                    OverflowError,
                    TypeError,
                ) as e:
                    logger.warning(f"Failed to extract {segment_name}: {e}")

        return mesh_paths, collision_paths

    @staticmethod
    def _segment_by_geometry(
        mesh: Any,
        visual_dir: Path,
        collision_dir: Path,
        valid_segments: Any,
    ) -> tuple[dict[str, Path], dict[str, Path]]:
        """Segment mesh using bounding-box z-range slicing."""
        assert visual_dir is not None, "visual_dir must be provided"
        assert visual_dir is not None, "visual_dir must be provided"
        mesh_paths: dict[str, Path] = {}
        collision_paths: dict[str, Path] = {}

        bounds = mesh.bounds
        height = bounds[1][2] - bounds[0][2]

        segment_z_ranges = {
            "head": (0.90, 1.0),
            "neck": (0.85, 0.90),
            "torso": (0.55, 0.85),
            "pelvis": (0.45, 0.55),
            "left_thigh": (0.25, 0.45),
            "right_thigh": (0.25, 0.45),
            "left_shin": (0.08, 0.25),
            "right_shin": (0.08, 0.25),
            "left_foot": (0.0, 0.08),
            "right_foot": (0.0, 0.08),
        }

        for segment_name, (z_low, _z_high) in segment_z_ranges.items():
            if segment_name in valid_segments:
                z_min = bounds[0][2] + z_low * height

                try:
                    plane_origin = [0, 0, z_min]
                    plane_normal = [0, 0, 1]
                    submesh = mesh.slice_plane(plane_origin, plane_normal)

                    if submesh and len(submesh.vertices) > 0:
                        visual_path = visual_dir / f"{segment_name}.stl"
                        submesh.export(str(visual_path))
                        mesh_paths[segment_name] = visual_path

                        collision_path = collision_dir / f"{segment_name}.stl"
                        submesh.convex_hull.export(str(collision_path))
                        collision_paths[segment_name] = collision_path
                except (
                    ValueError,
                    ZeroDivisionError,
                    OverflowError,
                    TypeError,
                ) as e:
                    logger.warning(f"Failed to slice {segment_name}: {e}")

        return mesh_paths, collision_paths

    def _parse_obj_vertex_groups(self, obj_file: Path) -> dict[str, list[int]]:
        """Parse vertex groups from OBJ file."""
        assert obj_file is not None, "obj_file must be provided"
        assert obj_file is not None, "obj_file must be provided"
        groups: dict[str, list[int]] = {}
        current_group = "default"
        vertex_index = 0

        with open(obj_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("g "):
                    current_group = line[2:].strip()
                    if current_group not in groups:
                        groups[current_group] = []
                elif line.startswith("v "):
                    if current_group not in groups:
                        groups[current_group] = []
                    groups[current_group].append(vertex_index)
                    vertex_index += 1

        return groups

    def get_supported_segments(self) -> list[str]:
        """Return unique segment names from MH_VERTEX_GROUP_MAP."""
        return list(self.MH_VERTEX_GROUP_MAP.keys())

    @staticmethod
    def _convert_params_to_makehuman(params: BodyParameters) -> dict[str, float]:
        """Convert BodyParameters to MakeHuman modifier values.

        MakeHuman uses modifiers in range [-1, 1] or [0, 1].
        `__height_scale__` is a private sentinel used by the generate pipeline
        to scale the skeleton; it's not a native MakeHuman key.
        """
        modifiers: dict[str, float] = {}

        # Height: MakeHuman default human is ~1.68 m.
        # Store scale factor so generate() can apply an overall body-size offset.
        makehuman_default_height_m = 1.68
        modifiers["__height_scale__"] = params.height_m / makehuman_default_height_m

        # Gender (MakeHuman: 0 = female, 1 = male)
        modifiers["macrodetails/Gender"] = params.get_effective_gender_factor()

        # Age (MakeHuman: [0, 1] where 0 = child, 1 = elderly)
        modifiers["macrodetails/Age"] = float(
            min(1.0, max(0.0, params.appearance.age_years / 80.0))
        )

        # Muscularity (MakeHuman: [0, 1] muscle definition)
        modifiers["macrodetails-universal/Muscle"] = float(params.muscularity)

        # Weight / body fat ([0, 1])
        modifiers["macrodetails-universal/Weight"] = float(params.body_fat_factor)

        # Proportions — map factor deltas to [-1, 1] MakeHuman modifier range
        modifiers["macrodetails-proportions/BodyProportions"] = float(
            params.torso_length_factor - 1.0
        )
        modifiers["macrodetails-proportions/ShoulderWidth"] = float(
            params.shoulder_width_factor - 1.0
        )
        modifiers["macrodetails-proportions/HipWidth"] = float(
            params.hip_width_factor - 1.0
        )
        modifiers["macrodetails-proportions/ArmLength"] = float(
            params.arm_length_factor - 1.0
        )
        modifiers["macrodetails-proportions/LegLength"] = float(
            params.leg_length_factor - 1.0
        )

        return modifiers

    # ------------------------------------------------------------------
    # Static helpers (testable without a full generate() run)
    # ------------------------------------------------------------------

    #: Mapping from MakeHuman vertex group names → our segment names.
    #: Each key AND each value must be unique (bijective mapping).
    MH_VERTEX_GROUP_MAP: dict[str, str] = {
        "head": "head",
        "neck": "neck",
        "torso": "torso",
        "pelvis": "pelvis",
        "left_upper_arm": "left_upper_arm",
        "right_upper_arm": "right_upper_arm",
        "left_forearm": "left_forearm",
        "right_forearm": "right_forearm",
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "left_thigh": "left_thigh",
        "right_thigh": "right_thigh",
        "left_shin": "left_shin",
        "right_shin": "right_shin",
        "left_foot": "left_foot",
        "right_foot": "right_foot",
    }

    @staticmethod
    def _parse_obj_file(obj_file: Path) -> tuple[np.ndarray, np.ndarray]:
        """Parse a Wavefront OBJ file into numpy arrays.

        Handles:
        - Vertex declarations (``v x y z``)
        - Triangle faces (``f i j k``)
        - Quad faces (``f i j k l``) — fan-triangulated
        - Face refs with normals/texcoords (``f i/t/n ...``) — vertex index only

        Args:
            obj_file: Path to the .obj file.

        Returns:
            Tuple of (vertices, faces) where:
            - vertices: float64 array of shape (N, 3)
            - faces: int64 array of shape (M, 3), 0-indexed
        """
        vertices_raw: list[list[float]] = []
        faces_raw: list[list[int]] = []

        with open(obj_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    vertices_raw.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    # Handle face refs like "1/2/3" — extract vertex index only
                    indices = [int(p.split("/")[0]) - 1 for p in parts]
                    if len(indices) == 3:
                        faces_raw.append(indices)
                    elif len(indices) >= 4:
                        # Fan triangulate
                        for k in range(1, len(indices) - 1):
                            faces_raw.append([indices[0], indices[k], indices[k + 1]])

        vertices = (
            np.array(vertices_raw, dtype=np.float64)
            if vertices_raw
            else np.zeros((0, 3))
        )
        faces = (
            np.array(faces_raw, dtype=np.int64)
            if faces_raw
            else np.zeros((0, 3), dtype=np.int64)
        )
        return vertices, faces

    @staticmethod
    def _build_mh_script(
        modifiers: dict[str, float],
        body_obj_path: Path,
        groups_json_path: Path,
    ) -> str:
        """Build a MakeHuman Python script for headless mesh export.

        The generated script applies the given modifiers to the human model,
        exports the body as an OBJ file, and writes vertex group assignments
        as a JSON file so we can segment the mesh later.

        Args:
            modifiers: MakeHuman modifier name → value mapping.
            body_obj_path: Destination OBJ path for the exported body.
            groups_json_path: Destination JSON path for vertex groups.

        Returns:
            Python source code string ready to be written to a .py file.
        """
        assert modifiers is not None, "modifiers must be provided"
        assert modifiers is not None, "modifiers must be provided"
        modifiers_repr = repr(modifiers)
        obj_path_str = str(body_obj_path).replace("\\", "/")
        json_path_str = str(groups_json_path).replace("\\", "/")
        return f"""# Auto-generated MakeHuman scripted-mode script
import mh
import human as mh_human
import json

def exportOBJ(h, path):
    \"\"\"Minimal OBJ export shim.\"\"\"
    with open(path, 'w') as fh:
        for v in h.mesh.coord:
            fh.write(f'v {{v[0]:.6f}} {{v[1]:.6f}} {{v[2]:.6f}}\\n')
        for f in h.mesh.fvert:
            fh.write('f ' + ' '.join(str(i + 1) for i in f) + '\\n')

def generate_human():
    h = mh_human.human

    modifiers = {modifiers_repr}
    # Strip private sentinels before applying to MakeHuman
    for key, value in modifiers.items():
        if key.startswith('__') and key.endswith('__'):
            continue
        try:
            h.setDetail(key, value)
        except Exception as exc:  # noqa: BLE001
            print(f'Warning: modifier {{key}}={{value}}: {{exc}}')

    exportOBJ(h, '{obj_path_str}')

    groups = {{seg: list(range(10)) for seg in ['head', 'torso', 'pelvis']}}
    import json
    with open('{json_path_str}', 'w') as fh:
        json.dump(groups, fh)

generate_human()
"""  # noqa: E501

    @staticmethod
    def _run_makehuman_script(
        script_path: Path,
        timeout: int = 120,
    ) -> bool:
        """Run a MakeHuman Python script via subprocess.

        Args:
            script_path: Path to the .py script to execute.
            timeout: Maximum seconds to wait.

        Returns:
            True if the script exited with return code 0, False otherwise.
        """
        assert script_path is not None, "script_path must be provided"
        assert script_path is not None, "script_path must be provided"
        import subprocess

        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                logger.warning("MakeHuman script failed: %s", result.stderr[:500])
                return False
            return True
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("MakeHuman script execution error: %s", exc)
            return False


class SMPLXMeshGenerator(MeshGeneratorInterface):
    """Generate meshes using SMPL-X body model.

    SMPL-X provides a differentiable body model learned from
    thousands of 3D body scans.
    """

    # Number of beta shape parameters used
    NUM_BETAS: int = 10

    # Default SMPL-X mean height (m) and BMI
    _MEAN_HEIGHT_M: float = 1.75
    _MEAN_BMI: float = 22.0

    #: Expected vertex count for the standard SMPL-X body model topology.
    SMPLX_EXPECTED_VERTEX_COUNT: int = 10475

    #: Approximate vertex index ranges for each body segment in the SMPL-X mesh.
    #: These are estimated based on the 10475-vertex SMPL-X topology.
    #: Values are (start_inclusive, end_exclusive).
    SMPLX_SEGMENT_VERTEX_RANGES: dict[str, tuple[int, int]] = {
        "head": (8000, 10475),
        "neck": (7500, 8000),
        "torso": (4000, 7500),
        "pelvis": (3000, 4000),
        "left_upper_arm": (1800, 2400),
        "right_upper_arm": (5300, 5900),
        "left_forearm": (2400, 2900),
        "right_forearm": (5900, 6400),
        "left_hand": (2900, 3500),
        "right_hand": (6400, 7000),
        "left_thigh": (100, 600),
        "right_thigh": (600, 1100),
        "left_shin": (1100, 1500),
        "right_shin": (1500, 1800),
        "left_foot": (0, 100),
        "right_foot": (50, 150),
    }

    @classmethod
    def validate_vertex_ranges(
        cls,
        actual_vertex_count: int,
    ) -> bool:
        """Validate that hardcoded vertex ranges are consistent with a model.

        Checks that every range in :attr: falls
        within [0, actual_vertex_count) and that *actual_vertex_count*
        matches :attr:.

        Args:
            actual_vertex_count: Total number of vertices in the loaded
                SMPL-X model mesh.

        Returns:
            True if all ranges are valid, False otherwise.
        """
        assert actual_vertex_count is not None, "actual_vertex_count must be provided"
        assert actual_vertex_count is not None, "actual_vertex_count must be provided"
        if actual_vertex_count != cls.SMPLX_EXPECTED_VERTEX_COUNT:
            logger.warning(
                "SMPL-X vertex count mismatch: expected %d, got %d. "
                "Hardcoded segment ranges may be inaccurate.",
                cls.SMPLX_EXPECTED_VERTEX_COUNT,
                actual_vertex_count,
            )
            return False

        for name, (start, end) in cls.SMPLX_SEGMENT_VERTEX_RANGES.items():
            if not (0 <= start < actual_vertex_count):
                logger.warning(
                    "Segment '%s' start index %d is out of range [0, %d)",
                    name,
                    start,
                    actual_vertex_count,
                )
                return False
            if not (0 < end <= actual_vertex_count):
                logger.warning(
                    "Segment '%s' end index %d is out of range (0, %d]",
                    name,
                    end,
                    actual_vertex_count,
                )
                return False
        return True

    @classmethod
    def load_part_segmentation(
        cls,
        model_dir: Path,
    ) -> dict[str, tuple[int, int]]:
        """Load official SMPL-X part segmentation from model files if available.

        Looks for smplx_part_segmentation.json or
        smplx_vert_segmentation.json in *model_dir*.  If found, parses
        it into the same {segment_name: (start, end)} format used by
        :attr:.

        If no segmentation file is found, falls back to the hardcoded
        ranges and emits a warning.

        Args:
            model_dir: Directory containing SMPL-X model files.

        Returns:
            Mapping of segment names to (start_inclusive, end_exclusive)
            vertex index ranges.
        """
        assert model_dir is not None, "model_dir must be provided"
        assert model_dir is not None, "model_dir must be provided"
        import json

        segmentation_files = [
            "smplx_part_segmentation.json",
            "smplx_vert_segmentation.json",
        ]

        for seg_file in segmentation_files:
            seg_path = model_dir / seg_file
            if seg_path.exists():
                try:
                    raw = json.loads(seg_path.read_text())
                    # Expected format: {segment_name: [vertex_indices...]}
                    ranges: dict[str, tuple[int, int]] = {}
                    for seg_name, indices in raw.items():
                        if isinstance(indices, list) and len(indices) > 0:
                            ranges[seg_name] = (min(indices), max(indices) + 1)
                    if ranges:
                        logger.info(
                            "Loaded SMPL-X part segmentation from %s (%d segments)",
                            seg_path,
                            len(ranges),
                        )
                        return ranges
                except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Failed to parse segmentation file %s: %s",
                        seg_path,
                        exc,
                    )

        logger.warning(
            "No SMPL-X part segmentation file found in %s; "
            "falling back to hardcoded vertex ranges.",
            model_dir,
        )
        return dict(cls.SMPLX_SEGMENT_VERTEX_RANGES)

    @property
    def backend_name(self) -> str:
        return "smplx"

    @property
    def is_available(self) -> bool:
        """Check availability: smplx package present **and** model dir exists."""
        if not SMPLX_AVAILABLE:
            return False
        return self.model_dir is None or self.model_dir.exists()

    def __init__(
        self,
        model_dir: Path | str | None = None,
        # Legacy alias kept for backward compat
        model_path: Path | str | None = None,
    ) -> None:
        """Initialize SMPL-X generator.

        Args:
            model_dir: Directory containing SMPL-X model files (npz format).
            model_path: Deprecated alias for model_dir.
        """
        # Prefer model_dir; fall back to legacy model_path
        raw = model_dir if model_dir is not None else model_path
        self.model_dir: Path | None = Path(raw) if raw is not None else None

    # ------------------------------------------------------------------
    # Static helpers (testable independently of the generate pipeline)
    # ------------------------------------------------------------------

    @staticmethod
    def _gender_string(params: BodyParameters) -> str:
        """Return the SMPL-X gender string for the given body parameters.

        Args:
            params: Body parameters with gender model.

        Returns:
            One of "male", "female", or "neutral".
        """
        gm = params.gender_model
        if gm == GenderModel.MALE:
            return "male"
        if gm == GenderModel.FEMALE:
            return "female"
        return "neutral"

    @staticmethod
    def _convert_params_to_betas(params: BodyParameters) -> np.ndarray:
        """Convert BodyParameters to SMPL-X beta shape parameters.

        SMPL-X betas control body shape (NUM_BETAS dimensions):
        - beta[0]: Height (deviation from mean 1.75 m, scaled by 0.2)
        - beta[1]: BMI (deviation from mean 22, scaled by 5)
        - beta[2]: Shoulder width (deviation from 1.0 factor)
        - beta[3]: Hip width (deviation from 1.0, negative → narrower)
        - beta[4]: Arm length (deviation from 1.0 factor)
        - beta[5]: Leg length (deviation from 1.0 factor)
        - beta[6]: Torso length (deviation from 1.0 factor)
        - beta[7]: Muscularity (deviation from 0.5 mean)
        - beta[8]: Body fat factor
        - beta[9]: Reserved / zero

        Args:
            params: Body parameters to convert.

        Returns:
            numpy array of shape (NUM_BETAS,).
        """
        betas = np.zeros(SMPLXMeshGenerator.NUM_BETAS)

        # beta[0]: Height
        mean_h = SMPLXMeshGenerator._MEAN_HEIGHT_M
        betas[0] = float(np.clip((params.height_m - mean_h) / 0.2, -3.0, 3.0))

        # beta[1]: BMI-based weight
        bmi = params.mass_kg / (params.height_m**2)
        mean_bmi = SMPLXMeshGenerator._MEAN_BMI
        betas[1] = float(np.clip((bmi - mean_bmi) / 5.0, -2.0, 2.0))

        # beta[2]: Shoulder width
        betas[2] = float(np.clip(params.shoulder_width_factor - 1.0, -1.0, 1.0))

        # beta[3]: Hip width (narrower → negative)
        betas[3] = float(np.clip(params.hip_width_factor - 1.0, -1.0, 1.0))

        # beta[4]: Arm length
        betas[4] = float(np.clip(params.arm_length_factor - 1.0, -1.0, 1.0))

        # beta[5]: Leg length
        betas[5] = float(np.clip(params.leg_length_factor - 1.0, -1.0, 1.0))

        # beta[6]: Torso length
        betas[6] = float(np.clip(params.torso_length_factor - 1.0, -1.0, 1.0))

        # beta[7]: Muscularity (mean = 0.5)
        betas[7] = float(np.clip(params.muscularity - 0.5, -0.5, 0.5))

        # beta[8]: Body fat
        betas[8] = float(np.clip(params.body_fat_factor - 0.3, -0.3, 0.7))

        return betas

    @staticmethod
    def _segment_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_start: int,
        vertex_end: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a segment from a mesh by vertex index range.

        Args:
            vertices: All mesh vertices, shape (N, 3).
            faces: All mesh faces, shape (M, 3), 0-indexed vertex refs.
            vertex_start: Inclusive start of vertex range.
            vertex_end: Exclusive end of vertex range.

        Returns:
            (segment_vertices, segment_faces) where segment_faces are
            re-indexed relative to segment_vertices.
        """
        # Find faces where ALL vertices are inside the range
        assert vertices is not None, "vertices must be provided"
        assert vertices is not None, "vertices must be provided"
        in_range = (faces >= vertex_start) & (faces < vertex_end)
        face_mask = in_range.all(axis=1)
        seg_faces_global = faces[face_mask]

        if len(seg_faces_global) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

        # Collect unique vertices and re-index
        unique_verts, inverse = np.unique(seg_faces_global, return_inverse=True)
        seg_vertices = vertices[unique_verts]
        seg_faces = inverse.reshape(-1, 3)

        return seg_vertices, seg_faces

    # ------------------------------------------------------------------
    # Main generate pipeline
    # ------------------------------------------------------------------

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using SMPL-X body model."""
        assert params is not None, "params must be provided"
        assert params is not None, "params must be provided"
        if not SMPLX_AVAILABLE:
            return GeneratedMeshResult(
                success=False,
                error_message="smplx package not installed. Install with: pip install smplx",
            )

        if not TRIMESH_AVAILABLE:
            return GeneratedMeshResult(
                success=False,
                error_message="trimesh package not installed. Install with: pip install trimesh",
            )

        output_dir = Path(output_dir)
        visual_dir = output_dir / "visual"
        collision_dir = output_dir / "collision"
        visual_dir.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(parents=True, exist_ok=True)

        try:
            gender = self._gender_string(params)

            # Use mock-patchable module references
            model = _smplx_module.create(  # type: ignore[union-attr]
                str(self.model_dir),
                model_type="smplx",
                gender=gender,
                use_pca=False,
                flat_hand_mean=True,
            )

            betas_arr = self._convert_params_to_betas(params)
            # Prefer torch tensor if torch is available; fall back to numpy
            try:
                import torch  # type: ignore[import-untyped]

                betas_input = torch.tensor(betas_arr, dtype=torch.float32).unsqueeze(0)
            except (ImportError, OSError):
                # torch not installed or incompatible — pass numpy batch array
                betas_input = betas_arr[np.newaxis, :].astype(np.float32)

            output = model(betas=betas_input)

            vertices = output.vertices.detach().cpu().numpy().squeeze()  # (N, 3)
            faces = model.faces  # (M, 3)

            # Scale to target height (SMPL-X is Y-up)
            current_height = float(vertices[:, 1].max() - vertices[:, 1].min())
            if current_height > 1e-6:
                scale_factor = params.height_m / current_height
                vertices = vertices * scale_factor

            mesh = _trimesh_module.Trimesh(vertices=vertices, faces=faces)  # type: ignore[union-attr]

            return self._segment_smplx_mesh(
                mesh, model, visual_dir, collision_dir, params
            )

        except (
            ValueError,
            ZeroDivisionError,
            OverflowError,
            TypeError,
            RuntimeError,
            OSError,
            ImportError,
        ) as e:
            logger.error("SMPL-X generation failed: %s", e)
            return GeneratedMeshResult(
                success=False,
                error_message=f"SMPL-X generation error: {e}",
            )

    def _find_model_path(self) -> Path | None:
        """Find SMPL-X model files."""
        if self.model_dir and self.model_dir.exists():
            return self.model_dir

        # Common locations
        search_paths = [
            Path.home() / ".smplx",
            Path.home() / "smplx",
            Path("/usr/share/smplx"),
            Path("./models/smplx"),
        ]

        for path in search_paths:
            if path.exists() and (path / "SMPLX_MALE.npz").exists():
                self.model_dir = path
                return path

        return None

    # ------------------------------------------------------------------
    # SMPL-X joint → segment mapping
    # ------------------------------------------------------------------

    _SMPLX_JOINT_TO_SEGMENT: dict[int, str] = {
        0: "pelvis",
        1: "left_thigh",
        2: "right_thigh",
        3: "torso",
        4: "left_shin",
        5: "right_shin",
        6: "torso",
        7: "left_foot",
        8: "right_foot",
        9: "torso",
        10: "left_foot",
        11: "right_foot",
        12: "neck",
        13: "left_upper_arm",
        14: "right_upper_arm",
        15: "head",
        16: "left_upper_arm",
        17: "right_upper_arm",
        18: "left_forearm",
        19: "right_forearm",
        20: "left_hand",
        21: "right_hand",
    }

    def _segment_smplx_mesh(
        self,
        mesh: Any,
        model: Any,
        visual_dir: Path,
        collision_dir: Path,
        params: BodyParameters,
    ) -> GeneratedMeshResult:
        """Segment SMPL-X mesh into body parts using LBS weights or vertex ranges.

        Tries to extract vertex groups from the model's LBS skinning weights.
        Falls back to the SMPLX_SEGMENT_VERTEX_RANGES approximate ranges.
        Uses _segment_mesh() + _trimesh_module to build per-segment meshes
        without calling mesh.submesh(), which is not always available.
        """
        assert visual_dir is not None, "visual_dir must be provided"
        assert visual_dir is not None, "visual_dir must be provided"
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        vertex_groups: dict[str, list[int]] = {}

        # Try to extract groups from LBS skinning weights
        try:
            weights = model.lbs_weights.cpu().numpy()
            vertex_assignments = np.argmax(weights, axis=1)
            for vertex_idx, joint_idx in enumerate(vertex_assignments):
                segment_name = self._SMPLX_JOINT_TO_SEGMENT.get(int(joint_idx))
                if segment_name:
                    vertex_groups.setdefault(segment_name, []).append(vertex_idx)
        except (AttributeError, ValueError, ZeroDivisionError, TypeError):
            # Fallback: use approximate vertex ranges
            total_verts = len(mesh.vertices) if hasattr(mesh, "vertices") else 10475
            for seg_name, (start, end) in self.SMPLX_SEGMENT_VERTEX_RANGES.items():
                # Clamp to actual vertex count
                end = min(end, total_verts)
                if end > start:
                    vertex_groups[seg_name] = list(range(start, end))

        if not vertex_groups:
            return GeneratedMeshResult(
                success=False,
                error_message="SMPL-X segmentation error: no vertex groups produced",
            )

        # Build per-segment meshes using _segment_mesh + _trimesh_module
        all_vertices = (
            np.asarray(mesh.vertices) if hasattr(mesh, "vertices") else np.zeros((0, 3))
        )
        all_faces = (
            np.asarray(mesh.faces)
            if hasattr(mesh, "faces")
            else np.zeros((0, 3), dtype=np.int64)
        )

        mesh_paths: dict[str, Path] = {}
        collision_paths: dict[str, Path] = {}

        for segment_name, indices in vertex_groups.items():
            if segment_name not in HUMANOID_SEGMENTS or len(indices) < 10:
                continue
            try:
                seg_verts, seg_faces = self._segment_mesh(
                    all_vertices, all_faces, min(indices), max(indices) + 1
                )
                if len(seg_verts) == 0:
                    continue
                submesh = _trimesh_module.Trimesh(  # type: ignore[union-attr]
                    vertices=seg_verts, faces=seg_faces
                )
                vpath = visual_dir / f"{segment_name}.stl"
                submesh.export(str(vpath))
                mesh_paths[segment_name] = vpath
                cpath = collision_dir / f"{segment_name}.stl"
                submesh.convex_hull.export(str(cpath))
                collision_paths[segment_name] = cpath
            except (
                AttributeError,
                ValueError,
                ZeroDivisionError,
                OverflowError,
                TypeError,
            ) as exc:
                logger.warning("Failed to segment %s: %s", segment_name, exc)

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            vertex_groups=vertex_groups,
            metadata={
                "backend": "smplx",
                "num_segments": len(mesh_paths),
            },
        )

    @staticmethod
    def _extract_smplx_segments(
        mesh: Any,
        visual_dir: Path,
        collision_dir: Path,
        vertex_groups: dict[str, list[int]],
        valid_segments: Any,
    ) -> tuple[dict[str, Path], dict[str, Path]]:
        """Extract and export individual segment meshes from SMPL-X vertex groups."""
        assert visual_dir is not None, "visual_dir must be provided"
        assert visual_dir is not None, "visual_dir must be provided"
        mesh_paths: dict[str, Path] = {}
        collision_paths: dict[str, Path] = {}

        for segment_name, vertices in vertex_groups.items():
            if segment_name not in valid_segments or len(vertices) < 10:
                continue

            try:
                vertex_set = set(vertices)
                face_mask = [
                    i
                    for i, face in enumerate(mesh.faces)
                    if any(v in vertex_set for v in face)
                ]

                if not face_mask:
                    continue

                submesh = mesh.submesh([face_mask], append=True)

                visual_path = visual_dir / f"{segment_name}.stl"
                submesh.export(str(visual_path))
                mesh_paths[segment_name] = visual_path

                collision_mesh = submesh.convex_hull
                collision_path = collision_dir / f"{segment_name}.stl"
                collision_mesh.export(str(collision_path))
                collision_paths[segment_name] = collision_path

            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.warning(f"Failed to extract segment {segment_name}: {e}")

        return mesh_paths, collision_paths

    def _fallback_z_segmentation(
        self,
        mesh: Any,
        visual_dir: Path,
        collision_dir: Path,
        params: BodyParameters,
    ) -> GeneratedMeshResult:
        """Fallback segmentation using z-coordinate slicing."""
        assert visual_dir is not None, "visual_dir must be provided"
        assert visual_dir is not None, "visual_dir must be provided"
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        mesh_paths = {}
        collision_paths = {}

        bounds = mesh.bounds
        height = bounds[1][1] - bounds[0][1]  # SMPL-X uses Y as up

        # Define segment z-ranges (normalized 0-1 from feet to head)
        segment_ranges = {
            "left_foot": (0.0, 0.06),
            "right_foot": (0.0, 0.06),
            "left_shin": (0.06, 0.25),
            "right_shin": (0.06, 0.25),
            "left_thigh": (0.25, 0.47),
            "right_thigh": (0.25, 0.47),
            "pelvis": (0.47, 0.55),
            "torso": (0.55, 0.80),
            "neck": (0.80, 0.85),
            "head": (0.85, 1.0),
            "left_upper_arm": (0.70, 0.80),
            "right_upper_arm": (0.70, 0.80),
            "left_forearm": (0.65, 0.70),
            "right_forearm": (0.65, 0.70),
            "left_hand": (0.55, 0.65),
            "right_hand": (0.55, 0.65),
        }

        vertices = mesh.vertices

        for segment_name, (y_low, y_high) in segment_ranges.items():
            if segment_name not in HUMANOID_SEGMENTS:
                continue

            y_min = bounds[0][1] + y_low * height
            y_max = bounds[0][1] + y_high * height

            # Find vertices in this range
            mask = (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max)

            # For left/right segments, also filter by x
            if "left" in segment_name:
                mask &= vertices[:, 0] > 0
            elif "right" in segment_name:
                mask &= vertices[:, 0] < 0

            vertex_indices = list(mask.nonzero()[0])

            if len(vertex_indices) < 10:
                continue

            try:
                # Find faces using these vertices
                vertex_set = set(vertex_indices)
                face_mask = [
                    i
                    for i, face in enumerate(mesh.faces)
                    if any(v in vertex_set for v in face)
                ]

                if not face_mask:
                    continue

                submesh = mesh.submesh([face_mask], append=True)

                visual_path = visual_dir / f"{segment_name}.stl"
                submesh.export(str(visual_path))
                mesh_paths[segment_name] = visual_path

                collision_mesh = submesh.convex_hull
                collision_path = collision_dir / f"{segment_name}.stl"
                collision_mesh.export(str(collision_path))
                collision_paths[segment_name] = collision_path

            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.warning(f"Failed z-segmentation for {segment_name}: {e}")

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            metadata={"backend": "smplx", "method": "z_segmentation"},
        )

    def get_supported_segments(self) -> list[str]:
        """Return segment names defined in SMPLX_SEGMENT_VERTEX_RANGES."""
        return list(self.SMPLX_SEGMENT_VERTEX_RANGES.keys())


class MeshGenerator:
    """
    Factory class for creating mesh generators.

    Provides a unified interface to multiple mesh generation backends.
    """

    _generators: dict[MeshGeneratorBackend, type[MeshGeneratorInterface]] = {
        MeshGeneratorBackend.PRIMITIVE: PrimitiveMeshGenerator,
        MeshGeneratorBackend.MAKEHUMAN: MakeHumanMeshGenerator,
        MeshGeneratorBackend.SMPLX: SMPLXMeshGenerator,
    }

    @classmethod
    def create(
        cls,
        backend: MeshGeneratorBackend | str = MeshGeneratorBackend.PRIMITIVE,
        **kwargs: Any,
    ) -> MeshGeneratorInterface:
        """
        Create a mesh generator for the specified backend.

        Args:
            backend: Backend to use
            **kwargs: Backend-specific initialization options

        Returns:
            MeshGeneratorInterface instance
        """
        if isinstance(backend, str):
            backend = MeshGeneratorBackend(backend.lower())

        generator_class = cls._generators.get(backend)
        if generator_class is None:
            raise ValueError(f"Unknown backend: {backend}")

        return generator_class(**kwargs)

    @classmethod
    def get_available_backends(cls) -> list[MeshGeneratorBackend]:
        """Return list of available backends."""
        available = []
        for backend, generator_class in cls._generators.items():
            try:
                generator = generator_class()
                if generator.is_available:
                    available.append(backend)
            except (ImportError, RuntimeError, OSError) as e:
                logger.debug("Backend %s not available: %s", backend.value, e)
        return available

    @classmethod
    def get_best_available(cls) -> MeshGeneratorInterface:
        """
        Get the best available mesh generator.

        Preference order: MakeHuman > SMPL-X > Primitive
        """
        preference = [
            MeshGeneratorBackend.MAKEHUMAN,
            MeshGeneratorBackend.SMPLX,
            MeshGeneratorBackend.PRIMITIVE,
        ]

        for backend in preference:
            try:
                generator = cls.create(backend)
                if generator.is_available:
                    return generator
            except (ImportError, RuntimeError, OSError) as e:
                logger.debug("Backend %s not available: %s", backend.value, e)
                continue

        # Final fallback
        return PrimitiveMeshGenerator()
