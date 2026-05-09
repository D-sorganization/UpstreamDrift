from __future__ import annotations

from pathlib import Path
from typing import Any

from humanoid_character_builder.core.anthropometry import get_com_location
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.core.model import GeneratedLink
from humanoid_character_builder.core.segment_definitions import SegmentDefinition
from humanoid_character_builder.generators._geometry_helpers import create_geometry_dict
from humanoid_character_builder.generators._inertia_helpers import (
    compute_segment_inertia,
)
from humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    MeshInertiaCalculator,
)
from humanoid_character_builder.mesh.primitive_inertia import PrimitiveInertiaCalculator
from src.shared.python.body_part_viz.contracts import BodyPartShape


def apply_proportion_factors(
    dimensions: dict[str, dict[str, float]],
    params: BodyParameters,
) -> dict[str, dict[str, float]]:
    scaled = {}

    for seg_name, dims in dimensions.items():
        scaled_dims = dims.copy()

        seg_lower = seg_name.lower()

        if "arm" in seg_lower or "hand" in seg_lower:
            scaled_dims["length"] *= params.arm_length_factor
        elif "thigh" in seg_lower or "shin" in seg_lower or "foot" in seg_lower:
            scaled_dims["length"] *= params.leg_length_factor
        elif "thorax" in seg_lower or "lumbar" in seg_lower:
            scaled_dims["length"] *= params.torso_length_factor
            scaled_dims["width"] *= params.shoulder_width_factor
        elif "pelvis" in seg_lower:
            scaled_dims["width"] *= params.hip_width_factor
        elif "head" in seg_lower:
            for key in scaled_dims:
                scaled_dims[key] *= params.head_scale_factor
        elif "neck" in seg_lower:
            scaled_dims["length"] *= params.neck_length_factor

        width_factor = 1.0 + 0.2 * params.muscularity + 0.3 * params.body_fat_factor
        scaled_dims["width"] = scaled_dims.get("width", 0.05) * width_factor
        scaled_dims["depth"] = scaled_dims.get("depth", 0.05) * width_factor

        seg_params = params.get_segment_params(seg_name)
        scale = seg_params.scale.as_tuple()
        scaled_dims["width"] *= scale[0]
        scaled_dims["depth"] *= scale[1]
        scaled_dims["length"] *= scale[2]

        scaled[seg_name] = scaled_dims

    return scaled


def generate_materials(
    params: BodyParameters,
) -> dict[str, tuple[float, float, float, float]]:
    materials: dict[str, tuple[float, float, float, float]] = {}
    skin = params.appearance.skin_tone
    materials["skin"] = skin.as_tuple()
    materials["default"] = (0.7, 0.7, 0.7, 1.0)
    return materials


def generate_link(
    segment_name: str,
    segment_def: SegmentDefinition,
    params: BodyParameters,
    mass: float,
    dimensions: dict[str, float],
    gender_factor: float,
    mesh_dir: Path | str | None,
    inertia_mode: InertiaMode,
    mesh_inertia_calc: MeshInertiaCalculator,
    primitive_inertia_calc: PrimitiveInertiaCalculator,
    generate_collision: bool,
    visual_shape: BodyPartShape | None = None,
) -> GeneratedLink:
    seg_params = params.get_segment_params(segment_name)

    final_mass = seg_params.mass_kg if seg_params.has_mass_override() else mass

    inertia = compute_segment_inertia(
        segment_name,
        seg_params,
        final_mass,
        dimensions,
        inertia_mode,
        mesh_inertia_calc,
        primitive_inertia_calc,
        mesh_dir,
    )

    visual_geom = create_geometry_dict(
        segment_def, dimensions, is_collision=False, shape=visual_shape
    )
    collision_geom: dict[str, Any] | None = None
    if generate_collision:
        # Collision geometry stays on the legacy path: URDF collision
        # checkers strongly prefer fast primitives, and reusing the
        # visual mesh would explode collision-pair workload.
        collision_geom = create_geometry_dict(
            segment_def, dimensions, is_collision=True
        )

    length = dimensions.get("length", 0.1)
    com = get_com_location(segment_name, length, gender_factor)

    return GeneratedLink(
        name=segment_name,
        mass=final_mass,
        inertia=inertia,
        visual_geometry=visual_geom,
        collision_geometry=collision_geom,
        origin_xyz=com,
        origin_rpy=(0.0, 0.0, 0.0),
    )
