# ruff: noqa: E501
# ARCHITECTURE_DEBT — tracked as GitHub issue #1937
# This file is 1,173 lines and contains 4 distinct mesh generator backends
# (Primitive, MakeHuman, SMPLX) plus the public MeshGenerator facade.
# Recommended split:
#   mesh_generator_primitive.py  — PrimitiveMeshGenerator
#   mesh_generator_makehuman.py  — MakeHumanMeshGenerator
#   mesh_generator_smplx.py      — SMPLXMeshGenerator
#   mesh_generator.py            — MeshGenerator facade + MeshGeneratorBackend enum
# Risk: low-medium — backends are independent; facade is the only public API.
# Prerequisite: parametrize existing tests over all backends before splitting.

"""
Mesh generation interfaces for humanoid character builder.

This module defines interfaces for mesh generation backends
(MakeHuman, SMPL, etc.) and provides a factory for creating
mesh generators.
"""

from __future__ import annotations  # noqa: E402, F404

import logging  # noqa: E402
from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from enum import Enum  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,  # noqa: E402
)

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
        if params is None:
            raise ValueError("params must be provided")
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message="smplx package not installed. Install with: pip install smplx",  # noqa: E501
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
        if params is None:
            raise ValueError("params must be provided")
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
        if visual_dir is None:
            raise ValueError("visual_dir must be provided")
        import numpy as np

        from shared.python.humanoid_character_builder.core.segment_definitions import (
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
        if visual_dir is None:
            raise ValueError("visual_dir must be provided")
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
        if visual_dir is None:
            raise ValueError("visual_dir must be provided")
        from shared.python.humanoid_character_builder.core.segment_definitions import (
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
        from shared.python.humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())
