"""Terrain mixin for physics engines.

Provides terrain support that can be mixed into any physics engine class.
Following the Pragmatic Programmer principle of orthogonal, composable code.

Usage:
    class MyPhysicsEngine(TerrainMixin, BasePhysicsEngine):
        pass

    engine = MyPhysicsEngine()
    engine.set_terrain(terrain)
    engine.step(0.001)  # Terrain contact forces applied automatically
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Protocol

import numpy as np

from src.shared.python.logging_config import get_logger
from src.shared.python.terrain import Terrain, TerrainType
from src.shared.python.terrain_engine import (
    CompressibleTurfModel,
    TerrainContactModel,
    TerrainGeometryGenerator,
)

logger = get_logger(__name__)


class HasPosition(Protocol):
    """Protocol for objects that have a position."""

    def get_position(self) -> np.ndarray:
        """Get 3D position."""
        ...


class TerrainMixin(ABC):
    """Mixin class that adds terrain support to physics engines.

    This mixin provides:
    - Terrain configuration
    - Ground height queries
    - Contact force calculations
    - Terrain-specific friction/restitution
    - Compressible turf support

    The mixin is designed to be composable with any physics engine
    that follows the PhysicsEngine protocol.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize terrain mixin."""
        super().__init__(*args, **kwargs)
        self._terrain: Terrain | None = None
        self._contact_model: TerrainContactModel | None = None
        self._turf_model: CompressibleTurfModel | None = None
        self._terrain_enabled: bool = True
        self._use_compressible_turf: bool = True

    @property
    def terrain(self) -> Terrain | None:
        """Get the current terrain configuration."""
        return self._terrain

    @property
    def terrain_enabled(self) -> bool:
        """Check if terrain contact is enabled."""
        return self._terrain_enabled

    def set_terrain(
        self,
        terrain: Terrain,
        use_compressible_turf: bool = True,
    ) -> None:
        """Set the terrain configuration.

        Args:
            terrain: Terrain configuration
            use_compressible_turf: Whether to use compressible turf model
        """
        self._terrain = terrain
        self._use_compressible_turf = use_compressible_turf

        # Create contact models
        self._contact_model = TerrainContactModel(terrain)
        if use_compressible_turf:
            self._turf_model = CompressibleTurfModel(terrain)

        logger.info(
            f"Terrain set: {terrain.name} (compressible={use_compressible_turf})"
        )

    def enable_terrain(self, enabled: bool = True) -> None:
        """Enable or disable terrain contact.

        Args:
            enabled: Whether terrain contact is enabled
        """
        self._terrain_enabled = enabled

    def get_ground_height(self, x: float, y: float) -> float:
        """Get ground height at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Ground height (meters), or 0.0 if no terrain
        """
        if self._terrain is None:
            return 0.0

        try:
            return self._terrain.elevation.get_elevation(x, y)
        except ValueError:
            return 0.0

    def get_terrain_normal(self, x: float, y: float) -> np.ndarray:
        """Get terrain surface normal at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Unit normal vector (3,)
        """
        if self._terrain is None:
            return np.array([0.0, 0.0, 1.0])

        try:
            return self._terrain.elevation.get_normal(x, y)
        except ValueError:
            return np.array([0.0, 0.0, 1.0])

    def get_terrain_type(self, x: float, y: float) -> TerrainType:
        """Get terrain type at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Terrain type at the position
        """
        if self._terrain is None:
            return TerrainType.FAIRWAY

        return self._terrain.get_terrain_type(x, y)

    def get_terrain_friction(self, x: float, y: float) -> float:
        """Get friction coefficient at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Friction coefficient
        """
        if self._terrain is None:
            return 0.5

        material = self._terrain.get_material(x, y)
        return material.friction_coefficient

    def get_terrain_restitution(self, x: float, y: float) -> float:
        """Get coefficient of restitution at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Coefficient of restitution
        """
        if self._terrain is None:
            return 0.6

        material = self._terrain.get_material(x, y)
        return material.restitution

    def compute_terrain_contact_force(
        self,
        x: float,
        y: float,
        z: float,
        radius: float = 0.0,
        velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute contact force from terrain.

        Uses either the compressible turf model or simple spring-damper
        depending on configuration.

        Args:
            x: X position (meters)
            y: Y position (meters)
            z: Z position (meters)
            radius: Object radius (meters)
            velocity: Object velocity (3,) [m/s], optional

        Returns:
            Contact force vector (3,) [N]
        """
        if not self._terrain_enabled or self._terrain is None:
            return np.zeros(3)

        if self._use_compressible_turf and self._turf_model is not None:
            return self._turf_model.compute_turf_contact_force(
                x, y, z, radius, velocity
            )
        elif self._contact_model is not None:
            return self._contact_model.compute_contact_force(x, y, z, radius, velocity)

        return np.zeros(3)

    def compute_terrain_friction_force(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """Compute friction force from terrain contact.

        Args:
            x: X position (meters)
            y: Y position (meters)
            z: Z position (meters)
            radius: Object radius (meters)
            velocity: Object velocity (3,) [m/s]

        Returns:
            Friction force vector (3,) [N]
        """
        if not self._terrain_enabled or self._contact_model is None:
            return np.zeros(3)

        return self._contact_model.compute_friction_force(x, y, z, radius, velocity)

    def is_on_terrain(
        self,
        x: float,
        y: float,
        z: float,
        radius: float = 0.0,
    ) -> bool:
        """Check if object is in contact with terrain.

        Args:
            x: X position (meters)
            y: Y position (meters)
            z: Z position (meters)
            radius: Object radius (meters)

        Returns:
            True if in contact with terrain
        """
        if self._terrain is None or self._contact_model is None:
            return False

        return self._contact_model.is_in_contact(x, y, z, radius)

    def get_lie_quality(
        self,
        x: float,
        y: float,
        ball_radius: float = 0.02135,
    ) -> dict[str, Any]:
        """Get golf ball lie quality at a position.

        Args:
            x: X position (meters)
            y: Y position (meters)
            ball_radius: Golf ball radius (meters)

        Returns:
            Dictionary with lie quality metrics
        """
        if self._turf_model is None:
            return {
                "lie_type": "unknown",
                "sitting_depth": 0.0,
                "grass_interference": 0.0,
                "playability_factor": 1.0,
            }

        return self._turf_model.compute_lie_quality(x, y, ball_radius)

    def get_terrain_contact_params(
        self,
        x: float,
        y: float,
    ) -> dict[str, float]:
        """Get physics contact parameters for terrain at position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Dictionary with friction, restitution, stiffness, damping
        """
        if self._terrain is None:
            return {
                "friction": 0.5,
                "restitution": 0.6,
                "stiffness": 1e5,
                "damping": 1e3,
            }

        return self._terrain.get_contact_params(x, y)

    def generate_terrain_xml(self, name: str = "terrain") -> str:
        """Generate MuJoCo XML for terrain.

        Args:
            name: Name for terrain geometry

        Returns:
            XML string for MuJoCo model
        """
        if self._terrain is None:
            return ""

        generator = TerrainGeometryGenerator(self._terrain)
        return generator.generate_mujoco_xml(name)

    def generate_terrain_mesh(self) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """Generate terrain mesh for visualization or physics.

        Returns:
            Tuple of (vertices, triangles)
        """
        if self._terrain is None:
            return np.array([]), []

        generator = TerrainGeometryGenerator(self._terrain)
        return generator.generate_mesh()


class TerrainAwareSimulation:
    """Helper class for running simulations with terrain.

    Provides convenience methods for applying terrain forces to objects
    during simulation.
    """

    def __init__(
        self,
        terrain: Terrain,
        use_compressible_turf: bool = True,
    ) -> None:
        """Initialize terrain-aware simulation.

        Args:
            terrain: Terrain configuration
            use_compressible_turf: Whether to use compressible turf model
        """
        self.terrain = terrain
        self.contact_model = TerrainContactModel(terrain)
        self.turf_model = (
            CompressibleTurfModel(terrain) if use_compressible_turf else None
        )

    def compute_object_terrain_force(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float,
        use_turf: bool = True,
    ) -> np.ndarray:
        """Compute total terrain force on an object.

        Args:
            position: Object position (3,) [x, y, z]
            velocity: Object velocity (3,) [vx, vy, vz]
            radius: Object radius (meters)
            use_turf: Whether to use compressible turf model

        Returns:
            Total force vector (3,) [N]
        """
        x, y, z = position

        # Contact force (normal)
        if use_turf and self.turf_model is not None:
            contact_force = self.turf_model.compute_turf_contact_force(
                x, y, z, radius, velocity
            )
        else:
            contact_force = self.contact_model.compute_contact_force(
                x, y, z, radius, velocity
            )

        # Friction force (tangential)
        friction_force = self.contact_model.compute_friction_force(
            x, y, z, radius, velocity
        )

        return contact_force + friction_force

    def simulate_ball_landing(
        self,
        impact_position: np.ndarray,
        impact_velocity: np.ndarray,
        ball_mass: float = 0.04593,
        ball_radius: float = 0.02135,
    ) -> dict[str, Any]:
        """Simulate ball landing on terrain.

        Simple analytical model for ball landing behavior.

        Args:
            impact_position: Position at impact (3,)
            impact_velocity: Velocity at impact (3,)
            ball_mass: Ball mass (kg)
            ball_radius: Ball radius (meters)

        Returns:
            Dictionary with landing results
        """
        x, y, z = impact_position

        # Get terrain properties
        terrain_type = self.terrain.get_terrain_type(x, y)
        material = self.terrain.get_material(x, y)
        normal = self.terrain.elevation.get_normal(x, y)

        # Decompose velocity into normal and tangential
        v_normal_mag = abs(np.dot(impact_velocity, normal))
        v_tangent = impact_velocity - np.dot(impact_velocity, normal) * normal
        v_tangent_mag = np.linalg.norm(v_tangent)

        # Compute energy absorption if using turf model
        if self.turf_model is not None:
            energy = self.turf_model.compute_energy_absorption(x, y, impact_velocity)
            energy_absorbed = energy["absorbed_energy"]
            energy_remaining = energy["remaining_energy"]
        else:
            energy_absorbed = 0.0
            energy_remaining = 0.5 * ball_mass * np.linalg.norm(impact_velocity) ** 2

        # Estimate bounce velocity
        rebound_v_normal = v_normal_mag * material.restitution
        rebound_v_tangent_mag = v_tangent_mag * (1.0 - material.rolling_resistance)

        # Get lie quality
        if self.turf_model is not None:
            lie = self.turf_model.compute_lie_quality(x, y, ball_radius)
        else:
            lie = {"lie_type": "normal", "playability_factor": 1.0}

        return {
            "terrain_type": terrain_type,
            "impact_speed": np.linalg.norm(impact_velocity),
            "rebound_speed": np.sqrt(rebound_v_normal**2 + rebound_v_tangent_mag**2),
            "energy_absorbed": energy_absorbed,
            "energy_remaining": energy_remaining,
            "lie_quality": lie,
            "material": material.name,
        }
