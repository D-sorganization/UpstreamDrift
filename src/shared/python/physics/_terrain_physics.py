from __future__ import annotations
import math

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.core.constants import GRAVITY
from src.shared.python.physics.terrain import Terrain


@dataclass
class TerrainContactModel:
    """Contact physics model for terrain interaction.

    Implements spring-damper contact model with terrain-specific
    friction and restitution.

    Attributes:
        terrain: Terrain configuration
        stiffness: Contact stiffness (N/m)
        damping: Contact damping (N*s/m)
    """

    terrain: Terrain
    stiffness: float = 1e5
    damping: float = 1e3

    def is_in_contact(
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
            z: Z position (object center height, meters)
            radius: Object radius for collision (meters)

        Returns:
            True if in contact
        """
        if x is None:
            raise ValueError("x must be provided")
        ground_height = self.terrain.get_elevation(x, y)
        contact_height = z - radius

        return contact_height <= ground_height

    def compute_penetration(
        self,
        x: float,
        y: float,
        z: float,
        radius: float = 0.0,
    ) -> float:
        """Compute penetration depth into terrain.

        Args:
            x: X position (meters)
            y: Y position (meters)
            z: Z position (meters)
            radius: Object radius (meters)

        Returns:
            Penetration depth (positive when penetrating, meters)
        """
        if x is None:
            raise ValueError("x must be provided")
        ground_height = self.terrain.get_elevation(x, y)
        contact_height = z - radius

        return max(0.0, ground_height - contact_height)

    def compute_contact_force(
        self,
        x: float,
        y: float,
        z: float,
        radius: float = 0.0,
        velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute contact force from terrain.

        Uses spring-damper model: F = k*d + c*v_n

        Args:
            x: X position (meters)
            y: Y position (meters)
            z: Z position (meters)
            radius: Object radius (meters)
            velocity: Object velocity (3,) [m/s], optional

        Returns:
            Contact force vector (3,) [N]
        """
        if x is None:
            raise ValueError("x must be provided")
        penetration = self.compute_penetration(x, y, z, radius)

        if penetration <= 0:
            return np.zeros(3)

        # Get terrain normal
        normal = self.terrain.get_normal(x, y)

        # Get terrain-specific stiffness from material
        contact_params = self.terrain.get_contact_params(x, y)
        stiffness = contact_params.get("stiffness", self.stiffness)
        damping = contact_params.get("damping", self.damping)

        # Spring force
        spring_force = stiffness * penetration

        # Damping force (if velocity provided)
        damping_force = 0.0
        if velocity is not None:
            # Velocity component in normal direction
            v_normal = np.dot(velocity, normal)
            # Only damp if moving into surface
            if v_normal < 0:
                damping_force = -damping * v_normal

        # Total normal force magnitude
        force_magnitude = spring_force + damping_force

        # Force acts in normal direction
        return force_magnitude * normal

    def compute_friction_force(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        velocity: np.ndarray,
        normal_force: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute friction force from terrain contact.

        Uses Coulomb friction model: F_f = mu * F_n * (-v_t / |v_t|)

        Args:
            x: X position (meters)
            y: Y position (meters)
            z: Z position (meters)
            radius: Object radius (meters)
            velocity: Object velocity (3,) [m/s]
            normal_force: Optional normal force (3,) [N]

        Returns:
            Friction force vector (3,) [N]
        """
        # Get normal force if not provided
        if x is None:
            raise ValueError("x must be provided")
        if normal_force is None:
            normal_force = self.compute_contact_force(x, y, z, radius, velocity)

        normal_force_magnitude = math.sqrt(
            np.dot(normal_force, normal_force)
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        if normal_force_magnitude < 1e-6:
            return np.zeros(3)

        # Get terrain normal
        normal = self.terrain.get_normal(x, y)

        # Get tangential velocity (perpendicular to normal)
        v_normal_component = np.dot(velocity, normal) * normal
        v_tangent = velocity - v_normal_component

        v_tangent_magnitude = math.sqrt(
            np.dot(v_tangent, v_tangent)
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        if v_tangent_magnitude < 1e-6:
            return np.zeros(3)

        # Get friction coefficient
        mu = self.terrain.get_material(x, y).friction_coefficient

        # Coulomb friction (kinetic)
        friction_magnitude = mu * normal_force_magnitude

        # Direction opposes motion
        friction_direction = -v_tangent / v_tangent_magnitude

        return friction_magnitude * friction_direction


@dataclass
class CompressibleTurfModel:
    """Contact model for compressible turf/grass surfaces.

    Models the non-linear compression behavior of turf, including:
    - Progressive stiffening as compression increases
    - Grass bending and matting
    - Moisture effects on compression
    - Energy absorption during impact

    Attributes:
        terrain: Terrain configuration
        base_stiffness: Base stiffness for rigid surfaces (N/m)
        base_damping: Base damping coefficient (N*s/m)
    """

    terrain: Terrain
    base_stiffness: float = 1e5
    base_damping: float = 1e3

    def get_compression_state(
        self,
        x: float,
        y: float,
        z: float,
        radius: float = 0.0,
    ) -> dict[str, float]:
        """Get compression state at a position.

        Args:
            x: X position (meters)
            y: Y position (meters)
            z: Z position (meters)
            radius: Object radius (meters)

        Returns:
            Dictionary with compression_depth, effective_stiffness,
            max_compression, and compression_ratio
        """
        if x is None:
            raise ValueError("x must be provided")
        material = self.terrain.get_material(x, y)
        ground_height = self.terrain.get_elevation(x, y)

        # Contact point
        contact_z = z - radius

        # Raw penetration into nominal ground surface
        nominal_penetration = max(0.0, ground_height - contact_z)

        # Maximum compression depth for this material
        max_compression = material.get_max_compression_depth()

        # Effective compression (limited by max)
        compression_depth = min(nominal_penetration, max_compression)

        # Compression ratio (0 = no compression, 1 = max compression)
        compression_ratio = (
            compression_depth / max_compression if max_compression > 0 else 0.0
        )

        # Effective stiffness (non-linear: increases with compression)
        # Uses progressive stiffening model
        base_eff_stiffness = material.get_effective_stiffness(self.base_stiffness)
        stiffness_multiplier = 1.0 + 2.0 * compression_ratio**2
        effective_stiffness = base_eff_stiffness * stiffness_multiplier

        return {
            "compression_depth": compression_depth,
            "effective_stiffness": effective_stiffness,
            "max_compression": max_compression,
            "compression_ratio": compression_ratio,
            "nominal_penetration": nominal_penetration,
        }

    def compute_turf_contact_force(
        self,
        x: float,
        y: float,
        z: float,
        radius: float = 0.0,
        velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute contact force from compressible turf.

        Uses non-linear spring-damper model with progressive stiffening.

        Args:
            x: X position (meters)
            y: Y position (meters)
            z: Z position (meters)
            radius: Object radius (meters)
            velocity: Object velocity (3,) [m/s], optional

        Returns:
            Contact force vector (3,) [N]
        """
        if x is None:
            raise ValueError("x must be provided")
        material = self.terrain.get_material(x, y)
        state = self.get_compression_state(x, y, z, radius)

        compression = state["compression_depth"]
        if compression <= 0:
            return np.zeros(3)

        # Get terrain normal
        normal = self.terrain.get_normal(x, y)

        # Spring force with effective stiffness
        spring_force = state["effective_stiffness"] * compression

        # Damping force
        damping_force = 0.0
        if velocity is not None:
            v_normal = np.dot(velocity, normal)
            # Damping increases with compression (energy absorption)
            effective_damping = (
                self.base_damping
                * (1.0 - 0.7 * material.compressibility)
                * (1.0 + material.compression_damping)
            )
            if v_normal < 0:  # Moving into surface
                damping_force = -effective_damping * v_normal

        # Total force magnitude
        force_magnitude = spring_force + damping_force

        # Grass resistance (additional resistance from grass blades)
        if material.grass_height_m > 0 and material.turf_density > 0:
            grass_resistance = (
                0.1 * material.turf_density * material.grass_height_m * compression
            )
            force_magnitude += grass_resistance

        return force_magnitude * normal

    def compute_lie_quality(
        self,
        x: float,
        y: float,
        ball_radius: float = 0.02135,
    ) -> dict[str, Any]:
        """Compute golf ball lie quality at a position.

        Determines how the ball sits in the turf, affecting
        the quality of contact for the next shot.

        Args:
            x: X position (meters)
            y: Y position (meters)
            ball_radius: Golf ball radius (meters)

        Returns:
            Dictionary with lie_type, sitting_depth, grass_interference,
            and playability_factor
        """
        if x is None:
            raise ValueError("x must be provided")
        material = self.terrain.get_material(x, y)
        terrain_type = self.terrain.get_terrain_type(x, y)

        # Ball weight creates compression
        ball_weight = 0.04593 * GRAVITY  # Golf ball weight in N

        # Effective sitting depth based on compression
        max_compression = material.get_max_compression_depth()
        effective_stiffness = material.get_effective_stiffness(self.base_stiffness)

        # Static equilibrium: F = k * x
        sitting_depth = min(
            ball_weight / effective_stiffness if effective_stiffness > 0 else 0,
            max_compression,
        )

        # Grass interference (how much grass surrounds the ball)
        grass_height = material.grass_height_m
        if grass_height > 0:
            interference_ratio = min(1.0, sitting_depth / grass_height)
        else:
            interference_ratio = 0.0

        # Playability factor (1.0 = perfect, 0.0 = unplayable)
        # Based on how much of the ball is above grass level
        visible_height = 2 * ball_radius - sitting_depth
        if grass_height > 0:
            playability = max(0.0, min(1.0, visible_height / (2 * ball_radius)))
        else:
            playability = 1.0

        # Determine lie type
        if sitting_depth < 0.002:
            lie_type = "tight"
        elif sitting_depth < 0.005:
            lie_type = "normal"
        elif sitting_depth < 0.010:
            lie_type = "sitting_down"
        elif sitting_depth < 0.020:
            lie_type = "plugged"
        else:
            lie_type = "buried"

        return {
            "lie_type": lie_type,
            "sitting_depth": sitting_depth,
            "grass_interference": interference_ratio,
            "playability_factor": playability,
            "grass_height": grass_height,
            "terrain_type": terrain_type,
        }

    def compute_energy_absorption(
        self,
        x: float,
        y: float,
        impact_velocity: np.ndarray,
        mass: float = 0.04593,  # Golf ball mass
        radius: float = 0.02135,
    ) -> dict[str, float]:
        """Compute energy absorbed by turf during impact.

        Args:
            x: X position (meters)
            y: Y position (meters)
            impact_velocity: Velocity at impact (3,) [m/s]
            mass: Object mass (kg)
            radius: Object radius (meters)

        Returns:
            Dictionary with kinetic_energy, absorbed_energy,
            remaining_energy, and energy_absorption_ratio
        """
        if x is None:
            raise ValueError("x must be provided")
        material = self.terrain.get_material(x, y)
        normal = self.terrain.get_normal(x, y)

        # Kinetic energy
        speed = math.sqrt(
            np.dot(impact_velocity, impact_velocity)
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        kinetic_energy = 0.5 * mass * speed**2

        # Normal velocity component
        v_normal = abs(np.dot(impact_velocity, normal))

        # Energy absorbed depends on compressibility and damping
        # Higher compressibility = more energy absorption
        absorption_factor = (
            material.compressibility * 0.5
            + material.compression_damping * 0.3
            + (1.0 - material.restitution) * 0.2
        )

        # Normal component energy
        normal_energy = 0.5 * mass * v_normal**2

        # Absorbed energy (mostly from normal component)
        absorbed_energy = normal_energy * absorption_factor

        remaining_energy = kinetic_energy - absorbed_energy

        return {
            "kinetic_energy": float(kinetic_energy),
            "absorbed_energy": float(absorbed_energy),
            "remaining_energy": float(max(0.0, remaining_energy)),
            "energy_absorption_ratio": float(
                absorbed_energy / kinetic_energy if kinetic_energy > 0 else 0.0
            ),
        }
