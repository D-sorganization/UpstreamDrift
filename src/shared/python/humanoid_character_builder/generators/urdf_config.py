"""
URDF generator configuration dataclass.

Extracted from urdf_generator.py to isolate configuration concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from humanoid_character_builder.mesh.inertia_calculator import InertiaMode

if TYPE_CHECKING:
    from src.shared.python.body_part_viz.asset_library import ShapeLibrary


@dataclass
class URDFGeneratorConfig:
    """Configuration for URDF generation."""

    # Inertia calculation mode
    inertia_mode: InertiaMode = InertiaMode.PRIMITIVE

    # Density for uniform density calculation (kg/m^3)
    default_density: float = 1050.0

    # Mesh paths (relative to URDF or package://)
    mesh_package_name: str | None = None  # e.g., "humanoid_model"
    visual_mesh_dir: str = "meshes/visual"
    collision_mesh_dir: str = "meshes/collision"

    # Use mesh for visual geometry (vs primitives)
    use_mesh_visual: bool = False

    # Use mesh for collision geometry (vs primitives)
    use_mesh_collision: bool = False

    # Generate collision geometry
    generate_collision: bool = True

    # Joint configuration
    default_joint_damping: float = 0.5
    default_joint_friction: float = 0.0

    # URDF formatting
    pretty_print: bool = True
    indent: str = "  "

    # Expand composite joints (gimbal/universal) to multiple revolute joints
    expand_composite_joints: bool = True

    # Include comments in URDF
    include_comments: bool = True

    # Optional body_part_viz ShapeLibrary supplying per-link visuals.
    # Issue #4765: when provided, named links draw their <visual> from the
    # library instead of the legacy SegmentDefinition-derived primitives.
    shape_library: ShapeLibrary | None = None
