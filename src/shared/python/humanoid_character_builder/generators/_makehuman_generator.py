"""MakeHuman mesh generator for humanoid character builder.

Internal submodule extracted from mesh_generator.py to keep file size
within the line budget. Import via ``mesh_generator`` (the public module).
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,
)

from ._mesh_export import (
    validate_makehuman_script_inputs,
    validate_output_path_within_base,
)
from ._mesh_types import GeneratedMeshResult, MeshGeneratorInterface
from ._primitive_generator import PrimitiveMeshGenerator

logger = logging.getLogger(__name__)
_MAKEHUMAN_MODIFIER_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def _single_quoted_python_literal(value: str) -> str:
    """Serialize *value* as a single-quoted Python string literal.

    All characters that would otherwise be syntactically significant inside a
    Python string literal — backslashes, single quotes, carriage returns,
    newlines, and null bytes — are replaced with their ``\\`` escape
    sequences.  The result is always a syntactically valid Python expression
    that can be embedded verbatim in generated source code.

    Args:
        value: Arbitrary string to serialize.

    Returns:
        A ``'…'`` Python string literal whose ``ast.literal_eval`` round-trip
        recovers the original *value*.
    """
    escaped = (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\x00", "\\x00")
    )
    return f"'{escaped}'"


class MakeHumanMeshGenerator(MeshGeneratorInterface):
    """
    Generate meshes using MakeHuman.

    This is a placeholder for future MakeHuman integration.
    MakeHuman provides high-quality, customizable human meshes
    with proper vertex groups for segmentation.
    """

    def __init__(self, makehuman_path: Path | str | None = None) -> None:
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
        if self.makehuman_path and self.makehuman_path.exists():
            return True

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
        if params is None:
            raise ValueError("params must be provided")
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message=(
                    "MakeHuman not found. Please install MakeHuman or provide path."
                ),
            )

        output_dir = Path(output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError("output_dir must be a directory")
        visual_dir = output_dir / "visual"
        collision_dir = output_dir / "collision"
        visual_dir.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(parents=True, exist_ok=True)

        modifiers = self._convert_params_to_makehuman(params)

        try:
            return self._generate_via_api(
                params, modifiers, visual_dir, collision_dir, output_dir, **kwargs
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"MakeHuman API generation failed: {e}")

        try:
            return self._generate_from_presets(
                params, visual_dir, collision_dir, **kwargs
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"MakeHuman preset loading failed: {e}")

        logger.warning("Falling back to primitive mesh generation")
        primitive_gen = PrimitiveMeshGenerator()
        return primitive_gen.generate(params, output_dir, **kwargs)

    def _generate_via_api(
        self,
        params: BodyParameters,
        modifiers: dict[str, float],
        visual_dir: Path,
        collision_dir: Path,
        base_output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using MakeHuman Python API."""
        if params is None:
            raise ValueError("params must be provided")
        import subprocess
        import tempfile

        script_content = self._create_makehuman_script(
            modifiers, visual_dir, base_output_dir=base_output_dir
        )
        script_path = ""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(script_content)
            script_path = script_file.name

        try:
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

            return self._segment_mesh(visual_dir, collision_dir)

        finally:
            Path(script_path).unlink(missing_ok=True)

    def _create_makehuman_script(
        self,
        modifiers: dict[str, float],
        output_dir: Path,
        base_output_dir: Path | None = None,
    ) -> str:
        """Create a MakeHuman Python script for mesh generation.

        All caller-controlled values embedded in the generated script are
        serialized via :func:`_single_quoted_python_literal` so that shell
        metacharacters, quote characters, and newlines cannot escape the
        string literal context.

        Args:
            modifiers: Validated MakeHuman modifier key/value pairs.
            output_dir: Directory where the generated OBJ will be written.
                The export path embedded in the script is derived from this
                value via ``Path.resolve()``.
            base_output_dir: When provided, the resolved *output_dir* must be
                contained within this directory.  Pass the caller-supplied root
                ``output_dir`` to prevent path-traversal escapes from reaching
                the generated script.

        Returns:
            Python source code string safe to write to a temporary file and
            executed by the MakeHuman Python interpreter.
        """
        if modifiers is None:
            raise ValueError("modifiers must be provided")
        self._validate_makehuman_script_inputs(
            modifiers, output_dir, base_output_dir=base_output_dir
        )
        export_path = str((output_dir / "humanoid.obj").resolve())
        export_path_literal = _single_quoted_python_literal(export_path)
        script = f"""
import mh
import human
import export

def generate_human():
    h = human.human

    modifiers = {repr(modifiers)}
    for key, value in modifiers.items():
        try:
            h.setDetail(key, value)
        except Exception:
            pass

    export_path = {export_path_literal}
    export.exportObj(h, export_path, config={{
        'exportGroups': True,
        'helper': False,
        'scale': 1.0,
    }})

generate_human()
"""
        return script

    _validate_output_path_within_base = staticmethod(validate_output_path_within_base)
    _validate_makehuman_script_inputs = staticmethod(validate_makehuman_script_inputs)

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

        if self.makehuman_path is None:
            raise RuntimeError("MakeHuman path not configured")
        presets_dir = self.makehuman_path / "data" / "exports"
        if not presets_dir.exists():
            presets_dir = self.makehuman_path / "exports"

        preset_name = params.build_type.value
        gender = "male" if params.get_effective_gender_factor() > 0.5 else "female"
        preset_file = presets_dir / f"{gender}_{preset_name}.obj"

        if not preset_file.exists():
            preset_file = presets_dir / f"{gender}_average.obj"

        if not preset_file.exists():
            raise FileNotFoundError(f"No MakeHuman preset found: {preset_file}")

        mesh = trimesh.load(str(preset_file))

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
        if visual_dir is None:
            raise ValueError("visual_dir must be provided")
        from shared.python.humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

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
        if visual_dir is None:
            raise ValueError("visual_dir must be provided")
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
        if visual_dir is None:
            raise ValueError("visual_dir must be provided")
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
        if obj_file is None:
            raise ValueError("obj_file must be provided")
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
        from shared.python.humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())

    def _convert_params_to_makehuman(self, params: BodyParameters) -> dict[str, float]:
        """Convert BodyParameters to MakeHuman modifier values."""
        if params is None:
            raise ValueError("params must be provided")
        modifiers: dict[str, float] = {}

        modifiers["macrodetails/Gender"] = params.get_effective_gender_factor()
        modifiers["macrodetails/Age"] = min(
            1.0, max(0.0, params.appearance.age_years / 80.0)
        )
        modifiers["macrodetails-universal/Muscle"] = params.muscularity
        modifiers["macrodetails-universal/Weight"] = params.body_fat_factor
        modifiers["macrodetails-proportions/BodyProportions"] = (
            params.torso_length_factor - 1.0
        )

        return modifiers
