"""
Mesh generation interfaces for humanoid character builder.

This module defines interfaces for mesh generation backends
(MakeHuman, SMPL-X, etc.) and provides a factory for creating
mesh generators.

SMPL-X integration: See issue #980
MakeHuman integration: See issue #979
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from humanoid_character_builder.core.body_parameters import BodyParameters

logger = logging.getLogger(__name__)

# Graceful imports for optional dependencies

try:
    import smplx as _smplx_module

    SMPLX_AVAILABLE = True
except ImportError:
    _smplx_module = None  # type: ignore[assignment]
    SMPLX_AVAILABLE = False

try:
    import trimesh as _trimesh_module

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

            except Exception as e:
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
    Generate meshes using MakeHuman in scripted (headless) mode.

    MakeHuman provides high-quality, customizable human meshes with proper
    vertex groups for body-part segmentation.  This generator:

    1. Converts :class:`BodyParameters` to MakeHuman modifier values.
    2. Writes a MakeHuman Python script that applies modifiers and exports
       an OBJ mesh with vertex groups.
    3. Runs MakeHuman in ``--nogui`` scripted mode.
    4. Parses the exported OBJ/vertex-group data and splits the mesh into
       per-segment STL files.

    See issue #979
    """

    # MakeHuman vertex-group names mapped to our segment names
    MH_VERTEX_GROUP_MAP: dict[str, str] = {
        "head": "head",
        "neck": "neck",
        "spine-upper": "thorax",
        "spine-lower": "lumbar",
        "pelvis": "pelvis",
        "shoulder.L": "left_shoulder",
        "shoulder.R": "right_shoulder",
        "upperarm.L": "left_upper_arm",
        "upperarm.R": "right_upper_arm",
        "forearm.L": "left_forearm",
        "forearm.R": "right_forearm",
        "hand.L": "left_hand",
        "hand.R": "right_hand",
        "thigh.L": "left_thigh",
        "thigh.R": "right_thigh",
        "shin.L": "left_shin",
        "shin.R": "right_shin",
        "foot.L": "left_foot",
        "foot.R": "right_foot",
    }

    def __init__(self, makehuman_path: Path | str | None = None) -> None:
        """
        Initialise MakeHuman generator.

        Args:
            makehuman_path: Path to the MakeHuman installation directory
                (the folder containing ``makehuman.py`` or ``makehuman``
                executable).  If *None*, common locations are searched.
        """
        self.makehuman_path = Path(makehuman_path) if makehuman_path else None

    @property
    def backend_name(self) -> str:
        return "makehuman"

    @property
    def is_available(self) -> bool:
        return self._find_makehuman() is not None

    # ------------------------------------------------------------------
    # Installation discovery
    # ------------------------------------------------------------------

    def _find_makehuman(self) -> Path | None:
        """Locate a usable MakeHuman installation."""
        import os
        import shutil

        if self.makehuman_path and self.makehuman_path.exists():
            return self.makehuman_path

        env_path = os.environ.get("MAKEHUMAN_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                self.makehuman_path = p
                return p

        # Common install locations
        candidates = [
            Path("/usr/share/makehuman"),
            Path.home() / "makehuman",
            Path.home() / ".makehuman",
            Path("C:/Program Files/MakeHuman"),
            Path("C:/Program Files (x86)/MakeHuman"),
        ]
        for candidate in candidates:
            if candidate.exists():
                self.makehuman_path = candidate
                return candidate

        # Check if ``makehuman`` is on PATH
        mh_exe = shutil.which("makehuman")
        if mh_exe:
            self.makehuman_path = Path(mh_exe).parent
            return self.makehuman_path

        return None

    # ------------------------------------------------------------------
    # Parameter conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_params_to_makehuman(params: BodyParameters) -> dict[str, float]:
        """Convert :class:`BodyParameters` to MakeHuman modifier values.

        MakeHuman modifiers are typically in the range ``[0, 1]`` or
        ``[-1, 1]``.
        """
        modifiers: dict[str, float] = {}

        # Height handled by overall scale after export
        # MakeHuman default height is ~1.68 m
        modifiers["__height_scale__"] = params.height_m / 1.68

        # Gender  (MakeHuman: 0 = female, 1 = male)
        modifiers["macrodetails/Gender"] = params.get_effective_gender_factor()

        # Age (normalised to 0..1 where 80 years = 1.0)
        modifiers["macrodetails/Age"] = min(
            1.0, max(0.0, params.appearance.age_years / 80.0)
        )

        # Muscularity
        modifiers["macrodetails-universal/Muscle"] = params.muscularity

        # Weight / body fat
        modifiers["macrodetails-universal/Weight"] = params.body_fat_factor

        # Proportions
        modifiers["macrodetails-proportions/BodyProportions"] = (
            params.torso_length_factor - 1.0
        )

        # Shoulder width
        modifiers["macrodetails-proportions/ShoulderWidth"] = (
            params.shoulder_width_factor - 1.0
        )

        # Hip width
        modifiers["macrodetails-proportions/HipWidth"] = params.hip_width_factor - 1.0

        # Arm length
        modifiers["macrodetails-proportions/ArmLength"] = params.arm_length_factor - 1.0

        # Leg length
        modifiers["macrodetails-proportions/LegLength"] = params.leg_length_factor - 1.0

        return modifiers

    # ------------------------------------------------------------------
    # MakeHuman scripted-mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mh_script(
        modifiers: dict[str, float],
        output_obj: Path,
        output_groups_json: Path,
    ) -> str:
        """Build a Python script to run inside MakeHuman scripted mode.

        The script applies *modifiers*, exports an OBJ mesh, and writes a
        JSON file mapping vertex-group names to vertex index lists.
        """
        # Filter out our synthetic __height_scale__ key
        mh_mods = {k: v for k, v in modifiers.items() if not k.startswith("__")}
        mod_json = json.dumps(mh_mods)

        return f"""\
import json, G
human = G.app.selectedHuman

# Apply modifiers
for name, value in {mod_json}.items():
    mod = human.getModifier(name)
    if mod is not None:
        mod.setValue(value)
human.applyAllTargets()

# Export OBJ
from export import exportOBJ
exportOBJ.exportOBJ(human, "{output_obj.as_posix()}")

# Export vertex groups
groups = {{}}
mesh = human.meshData
if hasattr(mesh, "getVertexGroups"):
    for gname, indices in mesh.getVertexGroups().items():
        groups[gname] = list(int(i) for i in indices)
with open("{output_groups_json.as_posix()}", "w") as fp:
    json.dump(groups, fp)
"""

    def _run_makehuman_script(self, script_path: Path, timeout: int = 120) -> bool:
        """Run a Python script inside MakeHuman's ``--nogui`` mode.

        Returns True on success, False on failure.
        """
        mh_dir = self._find_makehuman()
        if mh_dir is None:
            return False

        mh_exe = mh_dir / "makehuman.py"
        if not mh_exe.exists():
            mh_exe = mh_dir / "makehuman"

        cmd = [
            "python",
            str(mh_exe),
            "--nogui",
            "--script",
            str(script_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(mh_dir),
            )
            if result.returncode != 0:
                logger.warning("MakeHuman script failed: %s", result.stderr[:500])
                return False
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("MakeHuman execution failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # OBJ parsing / segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_obj_file(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Parse a Wavefront OBJ file into vertices and faces.

        Returns:
            Tuple of (vertices (N,3), faces (M,3)) numpy arrays.
        """
        vertices_list: list[list[float]] = []
        faces_list: list[list[int]] = []

        with open(obj_path, encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v" and len(parts) >= 4:
                    vertices_list.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
                elif parts[0] == "f":
                    # OBJ faces can be "f v1 v2 v3" or "f v1/vt1/vn1 ..."
                    face_verts = []
                    for token in parts[1:]:
                        idx = int(token.split("/")[0]) - 1  # OBJ is 1-indexed
                        face_verts.append(idx)
                    if len(face_verts) >= 3:
                        # Triangulate quads by fan
                        for i in range(1, len(face_verts) - 1):
                            faces_list.append(
                                [face_verts[0], face_verts[i], face_verts[i + 1]]
                            )

        return (
            np.array(vertices_list, dtype=np.float64),
            np.array(faces_list, dtype=np.int64),
        )

    def _segment_by_vertex_groups(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        groups_json_path: Path,
        output_dir: Path,
        height_scale: float,
    ) -> tuple[dict[str, Path], dict[str, Path], dict[str, list[int]]]:
        """Split a full-body mesh into segments using vertex group data.

        Returns:
            Tuple of (mesh_paths, collision_paths, vertex_groups).
        """
        mesh_paths: dict[str, Path] = {}
        collision_paths: dict[str, Path] = {}
        vertex_groups: dict[str, list[int]] = {}

        visual_dir = output_dir / "visual"
        collision_dir = output_dir / "collision"
        visual_dir.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(parents=True, exist_ok=True)

        # Apply height scale
        scaled_verts = vertices * height_scale

        # Load vertex group assignments
        if groups_json_path.exists():
            with open(groups_json_path, encoding="utf-8") as fp:
                raw_groups: dict[str, list[int]] = json.load(fp)
        else:
            logger.warning("Vertex groups file not found: %s", groups_json_path)
            raw_groups = {}

        if not TRIMESH_AVAILABLE:
            logger.warning("trimesh unavailable, cannot export segment meshes")
            return mesh_paths, collision_paths, vertex_groups

        for mh_group, segment_name in self.MH_VERTEX_GROUP_MAP.items():
            indices = raw_groups.get(mh_group, [])
            if not indices:
                continue

            start = min(indices)
            end = max(indices) + 1

            seg_verts, seg_faces = SMPLXMeshGenerator._segment_mesh(
                scaled_verts, faces, start, end
            )
            if len(seg_faces) == 0:
                continue

            mesh = _trimesh_module.Trimesh(vertices=seg_verts, faces=seg_faces)

            visual_path = visual_dir / f"{segment_name}.stl"
            mesh.export(str(visual_path))
            mesh_paths[segment_name] = visual_path

            try:
                collision_mesh = mesh.convex_hull
            except Exception:
                collision_mesh = mesh
            collision_path = collision_dir / f"{segment_name}.stl"
            collision_mesh.export(str(collision_path))
            collision_paths[segment_name] = collision_path

            vertex_groups[segment_name] = indices

        return mesh_paths, collision_paths, vertex_groups

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate per-segment meshes via MakeHuman scripted mode.

        See issue #979
        """
        mh_dir = self._find_makehuman()
        if mh_dir is None:
            return GeneratedMeshResult(
                success=False,
                error_message=(
                    "MakeHuman not found. Install MakeHuman and set "
                    "MAKEHUMAN_PATH or provide makehuman_path to constructor."
                ),
            )

        modifiers = self._convert_params_to_makehuman(params)
        height_scale = modifiers.pop("__height_scale__", 1.0)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._generate_impl(modifiers, height_scale, output_dir, **kwargs)
        except Exception as exc:
            logger.exception("MakeHuman mesh generation failed")
            return GeneratedMeshResult(
                success=False,
                error_message=f"MakeHuman generation error: {exc}",
            )

    def _generate_impl(
        self,
        modifiers: dict[str, float],
        height_scale: float,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Internal generation logic."""
        timeout = kwargs.get("timeout", 120)

        with tempfile.TemporaryDirectory(prefix="mh_gen_") as tmpdir:
            tmp = Path(tmpdir)
            obj_path = tmp / "body.obj"
            groups_path = tmp / "groups.json"
            script_path = tmp / "generate.py"

            script = self._build_mh_script(modifiers, obj_path, groups_path)
            script_path.write_text(script, encoding="utf-8")

            success = self._run_makehuman_script(script_path, timeout=timeout)
            if not success:
                return GeneratedMeshResult(
                    success=False,
                    error_message=(
                        "MakeHuman scripted export failed. "
                        "Check MakeHuman installation and logs."
                    ),
                )

            if not obj_path.exists():
                return GeneratedMeshResult(
                    success=False,
                    error_message="MakeHuman did not produce an OBJ file.",
                )

            vertices, faces = self._parse_obj_file(obj_path)
            mesh_paths, collision_paths, vertex_groups = self._segment_by_vertex_groups(
                vertices, faces, groups_path, output_dir, height_scale
            )

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            vertex_groups=vertex_groups,
            metadata={
                "backend": "makehuman",
                "height_scale": height_scale,
                "modifiers_applied": len(modifiers),
                "total_vertices": int(vertices.shape[0]),
                "total_faces": int(faces.shape[0]),
            },
        )

    def get_supported_segments(self) -> list[str]:
        return list(self.MH_VERTEX_GROUP_MAP.values())


class SMPLXMeshGenerator(MeshGeneratorInterface):
    """
    Generate meshes using the SMPL-X body model.

    SMPL-X provides a differentiable body model learned from thousands of
    3D body scans.  When the ``smplx`` package and a model directory are
    available, this generator:

    1. Converts :class:`BodyParameters` into SMPL-X shape (betas) and pose
       parameters.
    2. Runs a forward pass to obtain the posed mesh vertices / faces.
    3. Segments the full-body mesh by vertex index ranges into per-body-part
       sub-meshes.
    4. Exports each segment as an STL file together with a simplified
       collision mesh.

    See issue #980
    """

    # SMPL-X default body-part vertex ranges (approximate, 10475 vertices)
    # These correspond to the SMPL-X part segmentation.
    SMPLX_SEGMENT_VERTEX_RANGES: dict[str, tuple[int, int]] = {
        "head": (0, 600),
        "neck": (600, 800),
        "thorax": (800, 2000),
        "lumbar": (2000, 2800),
        "pelvis": (2800, 3600),
        "left_shoulder": (3600, 3900),
        "right_shoulder": (3900, 4200),
        "left_upper_arm": (4200, 4800),
        "right_upper_arm": (4800, 5400),
        "left_forearm": (5400, 5900),
        "right_forearm": (5900, 6400),
        "left_hand": (6400, 7200),
        "right_hand": (7200, 8000),
        "left_thigh": (8000, 8600),
        "right_thigh": (8600, 9200),
        "left_shin": (9200, 9600),
        "right_shin": (9600, 10000),
        "left_foot": (10000, 10200),
        "right_foot": (10200, 10475),
    }

    # Number of SMPL-X shape parameters (betas) we populate
    NUM_BETAS = 10

    def __init__(self, model_dir: Path | str | None = None) -> None:
        """Initialise the SMPL-X generator.

        Args:
            model_dir: Path to the directory containing SMPL-X model files
                (``SMPLX_NEUTRAL.npz`` etc.).  If *None*, the generator checks
                the ``SMPLX_MODEL_DIR`` environment variable and several
                conventional locations.
        """
        self.model_dir = Path(model_dir) if model_dir else None

    @property
    def backend_name(self) -> str:
        return "smplx"

    @property
    def is_available(self) -> bool:
        if not SMPLX_AVAILABLE:
            return False
        return self._find_model_dir() is not None

    # ------------------------------------------------------------------
    # BodyParameters -> SMPL-X parameter conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_params_to_betas(params: BodyParameters) -> np.ndarray:
        """Map :class:`BodyParameters` to SMPL-X shape coefficients (betas).

        The first 10 beta components roughly correspond to overall size,
        weight, limb proportions, shoulder/hip width, etc.  We use a
        simple linear mapping from the human-readable parameters.

        Returns:
            1-D numpy array of shape ``(NUM_BETAS,)``.
        """
        betas = np.zeros(SMPLXMeshGenerator.NUM_BETAS, dtype=np.float32)

        # beta[0] ~ overall height deviation from mean (~1.70 m)
        betas[0] = (params.height_m - 1.70) * 5.0

        # beta[1] ~ mass / body volume
        bmi = params.mass_kg / (params.height_m**2)
        betas[1] = (bmi - 22.0) * 0.5

        # beta[2] ~ shoulder width
        betas[2] = (params.shoulder_width_factor - 1.0) * 3.0

        # beta[3] ~ hip width
        betas[3] = (params.hip_width_factor - 1.0) * 3.0

        # beta[4] ~ arm length
        betas[4] = (params.arm_length_factor - 1.0) * 3.0

        # beta[5] ~ leg length
        betas[5] = (params.leg_length_factor - 1.0) * 3.0

        # beta[6] ~ torso length
        betas[6] = (params.torso_length_factor - 1.0) * 3.0

        # beta[7] ~ muscularity / body fat
        betas[7] = (params.muscularity - 0.5) * 2.0

        # beta[8] ~ body fat
        betas[8] = (params.body_fat_factor - 0.2) * 2.0

        # beta[9] reserved
        return betas

    @staticmethod
    def _gender_string(params: BodyParameters) -> str:
        """Return ``'male'``, ``'female'``, or ``'neutral'``."""
        factor = params.get_effective_gender_factor()
        if factor > 0.7:
            return "male"
        elif factor < 0.3:
            return "female"
        return "neutral"

    # ------------------------------------------------------------------
    # Model directory discovery
    # ------------------------------------------------------------------

    def _find_model_dir(self) -> Path | None:
        """Locate the SMPL-X model files on disk."""
        import os

        if self.model_dir and self.model_dir.exists():
            return self.model_dir

        env_dir = os.environ.get("SMPLX_MODEL_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists():
                self.model_dir = p
                return p

        # Conventional locations
        candidates = [
            Path.home() / "body_models" / "smplx",
            Path.home() / ".smplx",
            Path.home() / "models" / "smplx",
        ]
        for candidate in candidates:
            if candidate.exists():
                self.model_dir = candidate
                return candidate

        return None

    # ------------------------------------------------------------------
    # Mesh segmentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a sub-mesh for a body-part vertex range.

        Args:
            vertices: Full-body vertices (V, 3).
            faces: Full-body face indices (F, 3).
            start: First vertex index (inclusive).
            end: Last vertex index (exclusive).

        Returns:
            Tuple of (segment_vertices, segment_faces) with re-indexed faces.
        """
        mask = np.zeros(len(vertices), dtype=bool)
        mask[start:end] = True

        # Keep faces where all three vertices are in the segment
        face_mask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
        seg_faces = faces[face_mask]

        # Re-index vertices
        old_to_new = np.full(len(vertices), -1, dtype=np.int64)
        old_to_new[start:end] = np.arange(end - start)

        seg_vertices = vertices[start:end]
        seg_faces = old_to_new[seg_faces]

        return seg_vertices, seg_faces

    # ------------------------------------------------------------------
    # Main generation entry-point
    # ------------------------------------------------------------------

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate per-segment STL meshes from SMPL-X.

        Args:
            params: Body parameters to convert.
            output_dir: Directory where STL files are written.
            **kwargs: Extra options.  Recognised keys:

                * ``num_betas`` (int): override for the number of betas.

        Returns:
            A :class:`GeneratedMeshResult` with paths and vertex groups.
        """
        if not SMPLX_AVAILABLE:
            return GeneratedMeshResult(
                success=False,
                error_message=(
                    "smplx package not installed. Install with: pip install smplx"
                ),
            )

        if not TRIMESH_AVAILABLE:
            return GeneratedMeshResult(
                success=False,
                error_message=(
                    "trimesh package required for STL export. "
                    "Install with: pip install trimesh"
                ),
            )

        model_dir = self._find_model_dir()
        if model_dir is None:
            return GeneratedMeshResult(
                success=False,
                error_message=(
                    "SMPL-X model files not found. Set SMPLX_MODEL_DIR or "
                    "place models in ~/body_models/smplx/"
                ),
            )

        try:
            return self._generate_impl(params, Path(output_dir), model_dir, **kwargs)
        except Exception as exc:
            logger.exception("SMPL-X mesh generation failed")
            return GeneratedMeshResult(
                success=False,
                error_message=f"SMPL-X generation error: {exc}",
            )

    def _generate_impl(
        self,
        params: BodyParameters,
        output_dir: Path,
        model_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Internal implementation that assumes dependencies are available."""
        import torch

        num_betas = kwargs.get("num_betas", self.NUM_BETAS)
        gender = self._gender_string(params)

        # Build SMPL-X model
        model = _smplx_module.create(
            model_path=str(model_dir),
            model_type="smplx",
            gender=gender,
            num_betas=num_betas,
            use_pca=False,
            flat_hand_mean=True,
        )

        betas = self._convert_params_to_betas(params)
        betas_tensor = torch.tensor(betas[:num_betas], dtype=torch.float32).unsqueeze(0)

        # Forward pass (neutral pose)
        output = model(betas=betas_tensor)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        faces = model.faces.astype(np.int64)

        # Prepare output directories
        visual_dir = output_dir / "visual"
        collision_dir = output_dir / "collision"
        visual_dir.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(parents=True, exist_ok=True)

        mesh_paths: dict[str, Path] = {}
        collision_paths: dict[str, Path] = {}
        vertex_groups: dict[str, list[int]] = {}

        for segment_name, (start, end) in self.SMPLX_SEGMENT_VERTEX_RANGES.items():
            seg_verts, seg_faces = self._segment_mesh(vertices, faces, start, end)

            if len(seg_faces) == 0:
                logger.debug("No faces for segment %s, skipping", segment_name)
                continue

            # Visual mesh
            mesh = _trimesh_module.Trimesh(vertices=seg_verts, faces=seg_faces)
            visual_path = visual_dir / f"{segment_name}.stl"
            mesh.export(str(visual_path))
            mesh_paths[segment_name] = visual_path

            # Collision mesh (convex hull)
            try:
                collision_mesh = mesh.convex_hull
            except Exception:
                collision_mesh = mesh
            collision_path = collision_dir / f"{segment_name}.stl"
            collision_mesh.export(str(collision_path))
            collision_paths[segment_name] = collision_path

            # Vertex group indices (relative to full body)
            vertex_groups[segment_name] = list(range(start, end))

        # Also export the full-body mesh
        full_mesh = _trimesh_module.Trimesh(vertices=vertices, faces=faces)
        full_path = visual_dir / "full_body.stl"
        full_mesh.export(str(full_path))
        mesh_paths["full_body"] = full_path

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            vertex_groups=vertex_groups,
            metadata={
                "backend": "smplx",
                "gender": gender,
                "num_betas": num_betas,
                "total_vertices": int(vertices.shape[0]),
                "total_faces": int(faces.shape[0]),
            },
        )

    def get_supported_segments(self) -> list[str]:
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
            except Exception:
                pass
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
            except Exception:
                continue

        # Final fallback
        return PrimitiveMeshGenerator()
