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

from humanoid_character_builder.core.body_parameters import BodyParameters

logger = logging.getLogger(__name__)


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

        # Convert body parameters to MakeHuman modifiers
        modifiers = self._convert_params_to_makehuman(params)

        # Try to use MakeHuman scripting API
        try:
            return self._generate_via_api(
                params, modifiers, visual_dir, collision_dir, **kwargs
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"MakeHuman API generation failed: {e}")

        # Fallback: Try to load pre-exported MakeHuman mesh
        try:
            return self._generate_from_presets(
                params, visual_dir, collision_dir, **kwargs
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"MakeHuman preset loading failed: {e}")

        # Final fallback to primitive generator
        logger.warning("Falling back to primitive mesh generation")
        primitive_gen = PrimitiveMeshGenerator()
        return primitive_gen.generate(params, output_dir, **kwargs)

    def _generate_via_api(
        self,
        params: BodyParameters,
        modifiers: dict[str, float],
        visual_dir: Path,
        collision_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using MakeHuman Python API.

        Requires MakeHuman to be installed and accessible via Python.
        """
        import subprocess
        import tempfile

        # Create MakeHuman script
        script_content = self._create_makehuman_script(modifiers, visual_dir)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(script_content)
            script_path = script_file.name

        try:
            # Run MakeHuman in scripted mode
            if self.makehuman_path is None:
                raise RuntimeError("MakeHuman path not configured")
            mh_executable = self.makehuman_path / "makehuman.py"
            if not mh_executable.exists():
                mh_executable = self.makehuman_path / "makehuman"

            result = subprocess.run(
                ["python", str(mh_executable), "--nogui", "--script", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(f"MakeHuman failed: {result.stderr}")

            # Load generated mesh and segment it
            return self._segment_mesh(visual_dir, collision_dir)

        finally:
            Path(script_path).unlink(missing_ok=True)

    def _create_makehuman_script(
        self, modifiers: dict[str, float], output_dir: Path
    ) -> str:
        """Create a MakeHuman Python script for mesh generation."""
        script = f"""
import mh
import human
import export

def generate_human():
    # Get human object
    h = human.human

    # Apply modifiers
    modifiers = {repr(modifiers)}
    for key, value in modifiers.items():
        try:
            h.setDetail(key, value)
        except Exception as exc:
            logger.warning(f"Failed to set modifier {{key}}={{value}}: {{exc}}")

    # Export as OBJ with vertex groups
    export_path = "{output_dir}/humanoid.obj"
    export.exportObj(h, export_path, config={{
        'exportGroups': True,
        'helper': False,
        'scale': 1.0,
    }})

generate_human()
"""
        return script

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
        # MakeHuman supports all standard humanoid segments
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())

    def _convert_params_to_makehuman(self, params: BodyParameters) -> dict[str, float]:
        """Convert BodyParameters to MakeHuman modifier values."""
        # MakeHuman uses modifiers in range [-1, 1] or [0, 1]
        modifiers = {}

        # Height is handled by overall scale
        # MakeHuman default is ~1.68m, adjust proportionally
        # height_scale = params.height_m / 1.68

        # Gender (MakeHuman: 0 = female, 1 = male)
        modifiers["macrodetails/Gender"] = params.get_effective_gender_factor()

        # Age (MakeHuman: range depends on modifier)
        modifiers["macrodetails/Age"] = min(
            1.0, max(0.0, params.appearance.age_years / 80.0)
        )

        # Muscularity (MakeHuman: muscle definition)
        modifiers["macrodetails-universal/Muscle"] = params.muscularity

        # Weight/body fat
        modifiers["macrodetails-universal/Weight"] = params.body_fat_factor

        # Proportions
        modifiers["macrodetails-proportions/BodyProportions"] = (
            params.torso_length_factor - 1.0
        )

        return modifiers


class SMPLXMeshGenerator(MeshGeneratorInterface):
    """
    Generate meshes using SMPL-X body model.

    This is a placeholder for future SMPL-X integration.
    SMPL-X provides a differentiable body model learned from
    thousands of 3D body scans.
    """

    @property
    def backend_name(self) -> str:
        return "smplx"

    @property
    def is_available(self) -> bool:
        try:
            import smplx  # noqa: F401

            return True
        except ImportError:
            return False

    def __init__(self, model_path: Path | str | None = None):
        """Initialize SMPL-X generator.

        Args:
            model_path: Path to SMPL-X model files (npz format)
        """
        self.model_path = Path(model_path) if model_path else None
        self._model = None

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using SMPL-X body model.

        SMPL-X provides a differentiable body model with:
        - 10 body shape parameters (betas)
        - 55 pose parameters for body joints
        - Hand and face parameters

        We map our BodyParameters to SMPL-X betas and pose.
        """
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message="smplx package not installed. Install with: pip install smplx",
            )

        try:
            import smplx
            import torch
            import trimesh
        except ImportError as e:
            return GeneratedMeshResult(
                success=False,
                error_message=f"Missing dependency: {e}",
            )

        output_dir = Path(output_dir)
        visual_dir = output_dir / "visual"
        collision_dir = output_dir / "collision"
        visual_dir.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(parents=True, exist_ok=True)

        # Find model path
        model_folder = self._find_model_path()
        if not model_folder:
            logger.warning("SMPL-X model not found, falling back to primitives")
            primitive_gen = PrimitiveMeshGenerator()
            return primitive_gen.generate(params, output_dir, **kwargs)

        try:
            # Determine gender
            gender = "male" if params.get_effective_gender_factor() > 0.5 else "female"

            # Create SMPL-X model
            model = smplx.create(
                model_folder,
                model_type="smplx",
                gender=gender,
                use_pca=False,
                flat_hand_mean=True,
            )

            # Convert body parameters to SMPL-X betas
            betas = self._params_to_betas(params)
            betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)

            # Generate body with neutral pose
            output = model(betas=betas_tensor)

            # Get vertices and faces
            vertices = output.vertices.detach().cpu().numpy()[0]
            faces = model.faces

            # Scale to target height
            current_height = vertices[:, 1].max() - vertices[:, 1].min()
            scale_factor = params.height_m / current_height
            vertices *= scale_factor

            # Create trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Export full mesh
            full_mesh_path = visual_dir / "humanoid_full.obj"
            mesh.export(str(full_mesh_path))

            # Segment mesh using SMPL-X vertex groups
            return self._segment_smplx_mesh(
                mesh, model, visual_dir, collision_dir, params
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error(f"SMPL-X generation failed: {e}")
            # Fallback to primitive generator
            primitive_gen = PrimitiveMeshGenerator()
            return primitive_gen.generate(params, output_dir, **kwargs)

    def _find_model_path(self) -> Path | None:
        """Find SMPL-X model files."""
        if self.model_path and self.model_path.exists():
            return self.model_path

        # Common locations
        search_paths = [
            Path.home() / ".smplx",
            Path.home() / "smplx",
            Path("/usr/share/smplx"),
            Path("./models/smplx"),
        ]

        for path in search_paths:
            if path.exists() and (path / "SMPLX_MALE.npz").exists():
                self.model_path = path
                return path

        return None

    def _params_to_betas(self, params: BodyParameters) -> list[float]:
        """Convert BodyParameters to SMPL-X beta shape parameters.

        SMPL-X betas control body shape (10 dimensions):
        - beta[0]: Overall size/height
        - beta[1]: Weight/heaviness
        - beta[2]: Arm/leg length ratio
        - beta[3]: Shoulder width
        - beta[4]: Hip width
        - etc.
        """
        import numpy as np

        betas = np.zeros(10)

        # Height deviation from mean (betas[0] affects overall size)
        # Mean SMPL-X height is ~1.7m
        height_deviation = (params.height_m - 1.7) / 0.2
        betas[0] = np.clip(height_deviation, -3, 3)

        # Weight/body composition (beta[1])
        weight_deviation = (params.mass_kg - 70) / 20
        betas[1] = np.clip(weight_deviation * 0.5, -2, 2)

        # Muscularity affects shape
        betas[2] = np.clip(params.muscularity - 0.5, -1, 1)

        # Limb proportions
        betas[3] = np.clip(params.torso_length_factor - 1.0, -0.5, 0.5) * 2
        betas[4] = np.clip(params.leg_length_factor - 1.0, -0.5, 0.5) * 2

        # Shoulder width
        betas[5] = np.clip(params.shoulder_width_factor - 1.0, -0.5, 0.5) * 2

        # Hip width
        betas[6] = np.clip(params.hip_width_factor - 1.0, -0.5, 0.5) * 2

        return list(betas.tolist())

    # SMPL-X joint indices to segment name mapping
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
        """Segment SMPL-X mesh into body parts using joint positions."""
        import numpy as np
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        vertex_groups: dict[str, list[int]] = {}

        try:
            weights = model.lbs_weights.cpu().numpy()
            vertex_assignments = np.argmax(weights, axis=1)

            for vertex_idx, joint_idx in enumerate(vertex_assignments):
                segment_name = self._SMPLX_JOINT_TO_SEGMENT.get(joint_idx)
                if segment_name:
                    if segment_name not in vertex_groups:
                        vertex_groups[segment_name] = []
                    vertex_groups[segment_name].append(vertex_idx)

            mesh_paths, collision_paths = self._extract_smplx_segments(
                mesh,
                visual_dir,
                collision_dir,
                vertex_groups,
                HUMANOID_SEGMENTS,
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"Vertex group extraction failed: {e}")
            return self._fallback_z_segmentation(
                mesh, visual_dir, collision_dir, params
            )

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
        # SMPL-X provides full body mesh, needs segmentation
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())


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
