# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Terrain-aware physics engine integration.

Provides terrain support for all physics engines including:
- Ground height queries based on elevation maps
- Contact normal calculations for sloped terrain
- Friction and restitution based on terrain type
- Contact force computation for terrain interaction
- Geometry generation for engine-specific formats

Design by Contract:
    Preconditions:
        - Terrain must be set before querying properties
        - Positions must be within terrain bounds

    Postconditions:
        - Normal vectors are unit vectors
        - Contact forces are physically valid
        - Generated geometry is valid for target engine
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.physics._terrain_geometry import TerrainGeometryGenerator
from src.shared.python.physics._terrain_physics import (
    CompressibleTurfModel,
    TerrainContactModel,
)
from src.shared.python.physics._terrain_utils import (
    apply_terrain_to_engine,
    register_terrain_parameters,
    validate_terrain,
)
from src.shared.python.physics.terrain import (
    Terrain,
    TerrainType,
)

logger = get_logger(__name__)


class PhysicsEngineProtocol(Protocol):
    """Protocol for physics engines that support terrain."""

    def set_ground_properties(
        self,
        height: float,
        friction: float,
        restitution: float,
    ) -> None:
        """Set ground contact properties."""
        ...


class TerrainAwareEngine:
    """Terrain-aware physics engine wrapper.

    Provides terrain queries and contact calculations for physics simulations.
    Can wrap any physics engine to add terrain support.

    Attributes:
        terrain: The terrain configuration
        default_stiffness: Default contact stiffness (N/m)
        default_damping: Default contact damping (N*s/m)
    """

    def __init__(
        self,
        terrain: Terrain | None = None,
        stiffness: float = 1e5,
        damping: float = 1e3,
    ) -> None:
        """Initialize terrain-aware engine.

        Args:
            terrain: Optional terrain configuration
            stiffness: Contact stiffness (N/m)
            damping: Contact damping (N*s/m)
        """
        if stiffness is None:
            raise ValueError("stiffness must be provided")
        self.terrain: Terrain | None = terrain
        self.default_stiffness = stiffness
        self.default_damping = damping

    def set_terrain(self, terrain: Terrain) -> None:
        """Set the terrain configuration.

        Args:
            terrain: Terrain configuration
        """
        if terrain is None:
            raise ValueError("terrain must be provided")
        self.terrain = terrain
        logger.info(f"Terrain set: {terrain.name}")

    def get_ground_height(self, x: float, y: float) -> float:
        """Get ground height at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Ground height (meters)
        """
        if x is None:
            raise ValueError("x must be provided")
        if self.terrain is None:
            return 0.0

        return self.terrain.get_elevation(x, y)

    def get_contact_normal(self, x: float, y: float) -> np.ndarray:
        """Get terrain contact normal at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Unit normal vector (3,)
        """
        if x is None:
            raise ValueError("x must be provided")
        if self.terrain is None:
            return np.array([0.0, 0.0, 1.0])

        try:
            return self.terrain.get_normal(x, y)
        except ValueError:
            return np.array([0.0, 0.0, 1.0])

    def get_friction(self, x: float, y: float) -> float:
        """Get friction coefficient at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Friction coefficient
        """
        if x is None:
            raise ValueError("x must be provided")
        if self.terrain is None:
            return 0.5

        material = self.terrain.get_material(x, y)
        return material.friction_coefficient

    def get_restitution(self, x: float, y: float) -> float:
        """Get restitution coefficient at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Coefficient of restitution
        """
        if x is None:
            raise ValueError("x must be provided")
        if self.terrain is None:
            return 0.6

        material = self.terrain.get_material(x, y)
        return material.restitution

    def get_terrain_properties(self, x: float, y: float) -> dict[str, Any]:
        """Get all terrain properties at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Dictionary of terrain properties
        """
        if x is None:
            raise ValueError("x must be provided")
        if self.terrain is None:
            return {
                "elevation": 0.0,
                "normal": np.array([0.0, 0.0, 1.0]),
                "terrain_type": TerrainType.FAIRWAY,
                "friction": 0.5,
                "restitution": 0.6,
            }

        props = self.terrain.get_properties_at(x, y)
        return {
            "elevation": props["elevation"],
            "normal": props["normal"],
            "terrain_type": props["terrain_type"],
            "friction": props["material"].friction_coefficient,
            "restitution": props["material"].restitution,
        }


__all__ = [
    "PhysicsEngineProtocol",
    "TerrainAwareEngine",
    "TerrainContactModel",
    "CompressibleTurfModel",
    "TerrainGeometryGenerator",
    "apply_terrain_to_engine",
    "validate_terrain",
    "register_terrain_parameters",
]

# Auto-register on import
try:
    register_terrain_parameters()
except (RuntimeError, ValueError, OSError) as e:
    logger.debug(f"Could not register terrain parameters: {e}")
