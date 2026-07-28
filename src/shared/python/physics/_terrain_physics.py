from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.core.constants import GRAVITY
from src.shared.python.physics.terrain import Terrain


def _magnitude(vec: np.ndarray) -> float:
    """Euclidean norm via ``math.sqrt(dot)`` (≈3x faster than ``np.linalg.norm``).

    The vector is cast to ``float`` first so an integer-dtype input cannot
    overflow inside ``np.dot`` before the square root (#7022).
    """
    arr = np.asarray(vec, dtype=float)
    return math.sqrt(float(np.dot(arr, arr)))


def _validate_finite_scalar(name: str, value: float) -> float:
    if value is None:
        raise ValueError(f"{name} must be provided")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_nonnegative_scalar(name: str, value: float) -> float:
    scalar = _validate_finite_scalar(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _validate_positive_scalar(name: str, value: float) -> float:
    scalar = _validate_finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _validate_xy(x: float, y: float) -> tuple[float, float]:
    return _validate_finite_scalar("x", x), _validate_finite_scalar("y", y)


def _validate_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (
        _validate_finite_scalar("x", x),
        _validate_finite_scalar("y", y),
        _validate_finite_scalar("z", z),
    )


def _validate_vector3(name: str, value: np.ndarray) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} must be provided")
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3-vector") from exc
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a finite 3-vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _ensure_finite_vector3(name: str, value: np.ndarray) -> np.ndarray:
    vector = _validate_vector3(name, value)
    return vector


def _ensure_finite_energy_payload(payload: dict[str, float]) -> dict[str, float]:
    for name, value in payload.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    return payload


# Energy-absorption blend weights (#7055).
#
# The fraction of normal-impact kinetic energy absorbed by turf is modelled as
# a CONVEX COMBINATION of three independent, normalised (0-1) material
# mechanisms:
#   * compressibility       — plastic deformation of the surface
#   * compression_damping   — viscous (rate-dependent) losses
#   * 1 - restitution       — inelastic rebound losses
# These weights are an engineering heuristic (not a measured physical
# constant): they encode the relative importance attributed to each mechanism
# and MUST sum to 1.0 so that the absorption factor stays bounded in [0, 1]
# whenever each mechanism input is in [0, 1]. The 0.5 emphasis on
# compressibility reflects that permanent deformation dominates soft-turf
# energy loss (cf. soil-mechanics impact-absorption models); damping and
# inelastic rebound split the remainder 0.3 / 0.2.
ENERGY_ABSORPTION_COMPRESSIBILITY_WEIGHT = 0.5
ENERGY_ABSORPTION_DAMPING_WEIGHT = 0.3
ENERGY_ABSORPTION_RESTITUTION_WEIGHT = 0.2

# Grass-blade resistance coefficient (#7055).
#
# Additional resistive force from grass blades is modelled as
#   F_grass = k_grass * turf_density * grass_height * compression.
# ``k_grass`` (units N per [density * m * m_compression]) is an empirical
# tuning constant chosen so that typical fairway/rough turf adds a small
# fraction of the spring force; it is a model heuristic, pinned by value tests
# rather than a measured constant.
GRASS_RESISTANCE_COEFFICIENT = 0.1


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
        x, y, z = _validate_xyz(x, y, z)
        radius = _validate_nonnegative_scalar("radius", radius)
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
        x, y, z = _validate_xyz(x, y, z)
        radius = _validate_nonnegative_scalar("radius", radius)
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
        x, y, z = _validate_xyz(x, y, z)
        radius = _validate_nonnegative_scalar("radius", radius)
        velocity_vector = (
            _validate_vector3("velocity", velocity) if velocity is not None else None
        )
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
        if velocity_vector is not None:
            # Velocity component in normal direction
            v_normal = np.dot(velocity_vector, normal)
            # Only damp if moving into surface
            if v_normal < 0:
                damping_force = -damping * v_normal

        # Total normal force magnitude
        force_magnitude = spring_force + damping_force

        # Force acts in normal direction
        return _ensure_finite_vector3("contact_force", force_magnitude * normal)

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
        x, y, z = _validate_xyz(x, y, z)
        radius = _validate_nonnegative_scalar("radius", radius)
        velocity = _validate_vector3("velocity", velocity)

        # Get normal force if not provided
        if normal_force is None:
            normal_force = self.compute_contact_force(x, y, z, radius, velocity)
        else:
            normal_force = _validate_vector3("normal_force", normal_force)

        normal_force_magnitude = _magnitude(normal_force)
        if normal_force_magnitude < 1e-6:
            return np.zeros(3)

        # Get terrain normal
        normal = self.terrain.get_normal(x, y)

        # Get tangential velocity (perpendicular to normal)
        v_normal_component = np.dot(velocity, normal) * normal
        v_tangent = velocity - v_normal_component

        v_tangent_magnitude = _magnitude(v_tangent)
        if v_tangent_magnitude < 1e-6:
            return np.zeros(3)

        # Get friction coefficient
        mu = self.terrain.get_material(x, y).friction_coefficient

        # Coulomb friction (kinetic)
        friction_magnitude = mu * normal_force_magnitude

        # Direction opposes motion
        friction_direction = -v_tangent / v_tangent_magnitude

        return _ensure_finite_vector3(
            "friction_force", friction_magnitude * friction_direction
        )


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
        x, y, z = _validate_xyz(x, y, z)
        radius = _validate_nonnegative_scalar("radius", radius)
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
        x, y, z = _validate_xyz(x, y, z)
        radius = _validate_nonnegative_scalar("radius", radius)
        velocity_vector = (
            _validate_vector3("velocity", velocity) if velocity is not None else None
        )
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
        if velocity_vector is not None:
            v_normal = np.dot(velocity_vector, normal)
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

        # Grass resistance (additional resistance from grass blades).
        # See GRASS_RESISTANCE_COEFFICIENT for provenance (#7055).
        if material.grass_height_m > 0 and material.turf_density > 0:
            grass_resistance = (
                GRASS_RESISTANCE_COEFFICIENT
                * material.turf_density
                * material.grass_height_m
                * compression
            )
            force_magnitude += grass_resistance

        return _ensure_finite_vector3("contact_force", force_magnitude * normal)

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
        x, y = _validate_xy(x, y)
        ball_radius = _validate_nonnegative_scalar("ball_radius", ball_radius)
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
        x, y = _validate_xy(x, y)
        impact_velocity = _validate_vector3("impact_velocity", impact_velocity)
        mass = _validate_positive_scalar("mass", mass)
        _validate_nonnegative_scalar("radius", radius)
        material = self.terrain.get_material(x, y)
        normal = self.terrain.get_normal(x, y)

        # Kinetic energy
        speed = _magnitude(impact_velocity)
        kinetic_energy = 0.5 * mass * speed**2

        # Normal velocity component
        v_normal = abs(np.dot(impact_velocity, normal))

        # Energy absorbed depends on compressibility and damping.
        # Higher compressibility = more energy absorption. The weights form a
        # normalised convex combination (sum == 1); see the module-level
        # ENERGY_ABSORPTION_*_WEIGHT constants for provenance (#7055).
        absorption_factor = (
            material.compressibility * ENERGY_ABSORPTION_COMPRESSIBILITY_WEIGHT
            + material.compression_damping * ENERGY_ABSORPTION_DAMPING_WEIGHT
            + (1.0 - material.restitution) * ENERGY_ABSORPTION_RESTITUTION_WEIGHT
        )

        # Normal component energy
        normal_energy = 0.5 * mass * v_normal**2

        # Absorbed energy (mostly from normal component)
        absorbed_energy = normal_energy * absorption_factor

        remaining_energy = kinetic_energy - absorbed_energy

        return _ensure_finite_energy_payload(
            {
                "kinetic_energy": float(kinetic_energy),
                "absorbed_energy": float(absorbed_energy),
                "remaining_energy": float(max(0.0, remaining_energy)),
                "energy_absorption_ratio": float(
                    absorbed_energy / kinetic_energy if kinetic_energy > 0 else 0.0
                ),
            }
        )
