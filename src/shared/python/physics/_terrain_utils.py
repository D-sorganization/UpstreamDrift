from __future__ import annotations

from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.physics.terrain import (
    MATERIALS,
    TERRAIN_MATERIAL_MAP,
    Terrain,
)

logger = get_logger(__name__)


def apply_terrain_to_engine(
    engine: Any,
    terrain: Terrain,
    x: float,
    y: float,
) -> None:
    """Apply terrain properties to a physics engine at a position.

    This is a convenience function for engines that support
    position-based ground property updates.

    Args:
        engine: Physics engine (must have set_ground_properties method)
        terrain: Terrain configuration
        x: X position (meters)
        y: Y position (meters)
    """
    if terrain is None:
        raise ValueError("terrain must be provided")
    height = terrain.get_elevation(x, y)
    material = terrain.get_material(x, y)

    if hasattr(engine, "set_ground_properties"):
        engine.set_ground_properties(
            height=height,
            friction=material.friction_coefficient,
            restitution=material.restitution,
        )
    else:
        logger.warning(
            f"Engine {type(engine).__name__} does not support set_ground_properties"
        )


def validate_terrain(  # noqa: C901
    terrain: Terrain,
    warn_low_resolution: bool = False,
) -> list[str]:
    """Validate terrain configuration.

    Args:
        terrain: Terrain to validate
        warn_low_resolution: Include warnings for low resolution

    Returns:
        List of error/warning messages (empty if valid)
    """
    if terrain is None:
        raise ValueError("terrain must be provided")
    messages = []

    elev = terrain.elevation

    # Check dimensions
    if elev.width <= 0:
        messages.append("Terrain width must be positive")
    if elev.length <= 0:
        messages.append("Terrain length must be positive")
    if elev.resolution <= 0:
        messages.append("Terrain resolution must be positive")

    # Check patches within bounds
    for i, patch in enumerate(terrain.patches):
        if patch.x_min < elev.origin_x or patch.x_max > elev.origin_x + elev.width:
            messages.append(
                f"Patch {i} ({patch.terrain_type.name}) X bounds exceed terrain bounds"
            )
        if patch.y_min < elev.origin_y or patch.y_max > elev.origin_y + elev.length:
            messages.append(
                f"Patch {i} ({patch.terrain_type.name}) Y bounds exceed terrain bounds"
            )

    # Resolution warnings
    if warn_low_resolution:
        min_dimension = min(elev.width, elev.length)
        if elev.resolution > min_dimension / 10:
            messages.append(
                f"Low terrain resolution ({elev.resolution}m) may cause inaccurate simulation"
            )

    return messages


def register_terrain_parameters() -> None:
    """Register terrain-related parameters with the physics registry."""
    from src.shared.python.physics.physics_parameters import (
        ParameterCategory,
        PhysicsParameter,
        get_parameter_registry,
    )

    registry = get_parameter_registry()

    # Friction parameters
    for terrain_type, material_name in TERRAIN_MATERIAL_MAP.items():
        if material_name in MATERIALS:
            material = MATERIALS[material_name]
            param_name = f"TERRAIN_FRICTION_{terrain_type.name}"

            registry.register(
                PhysicsParameter(
                    name=param_name,
                    value=material.friction_coefficient,
                    unit="dimensionless",
                    category=ParameterCategory.ENVIRONMENT,
                    description=f"Friction coefficient for {terrain_type.name.lower()}",
                    source="Golf course material properties",
                    min_value=0.0,
                    max_value=2.0,
                    is_constant=False,
                )
            )

            # Restitution parameters
            restitution_name = f"TERRAIN_RESTITUTION_{terrain_type.name}"
            registry.register(
                PhysicsParameter(
                    name=restitution_name,
                    value=material.restitution,
                    unit="dimensionless",
                    category=ParameterCategory.ENVIRONMENT,
                    description=f"Coefficient of restitution for {terrain_type.name.lower()}",
                    source="Golf course material properties",
                    min_value=0.0,
                    max_value=1.0,
                    is_constant=False,
                )
            )

    logger.info("Terrain parameters registered with physics registry")
