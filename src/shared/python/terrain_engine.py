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

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.shared.python.logging_config import get_logger
from src.shared.python.terrain import (
    MATERIALS,
    TERRAIN_MATERIAL_MAP,
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
        self.terrain: Terrain | None = terrain
        self.default_stiffness = stiffness
        self.default_damping = damping

    def set_terrain(self, terrain: Terrain) -> None:
        """Set the terrain configuration.

        Args:
            terrain: Terrain configuration
        """
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
        if self.terrain is None:
            return 0.0

        try:
            return self.terrain.elevation.get_elevation(x, y)
        except ValueError:
            # Out of bounds - return edge value
            return 0.0

    def get_contact_normal(self, x: float, y: float) -> np.ndarray:
        """Get terrain contact normal at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Unit normal vector (3,)
        """
        if self.terrain is None:
            return np.array([0.0, 0.0, 1.0])

        try:
            return self.terrain.elevation.get_normal(x, y)
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
        ground_height = self.terrain.elevation.get_elevation(x, y)
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
        ground_height = self.terrain.elevation.get_elevation(x, y)
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
        penetration = self.compute_penetration(x, y, z, radius)

        if penetration <= 0:
            return np.zeros(3)

        # Get terrain normal
        normal = self.terrain.elevation.get_normal(x, y)

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
        if normal_force is None:
            normal_force = self.compute_contact_force(x, y, z, radius, velocity)

        normal_force_magnitude = np.linalg.norm(normal_force)
        if normal_force_magnitude < 1e-6:
            return np.zeros(3)

        # Get terrain normal
        normal = self.terrain.elevation.get_normal(x, y)

        # Get tangential velocity (perpendicular to normal)
        v_normal_component = np.dot(velocity, normal) * normal
        v_tangent = velocity - v_normal_component

        v_tangent_magnitude = np.linalg.norm(v_tangent)
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
        material = self.terrain.get_material(x, y)
        ground_height = self.terrain.elevation.get_elevation(x, y)

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
        material = self.terrain.get_material(x, y)
        state = self.get_compression_state(x, y, z, radius)

        compression = state["compression_depth"]
        if compression <= 0:
            return np.zeros(3)

        # Get terrain normal
        normal = self.terrain.elevation.get_normal(x, y)

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
        material = self.terrain.get_material(x, y)
        terrain_type = self.terrain.get_terrain_type(x, y)

        # Ball weight creates compression
        ball_weight = 0.04593 * 9.81  # Golf ball weight in N

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
        material = self.terrain.get_material(x, y)
        normal = self.terrain.elevation.get_normal(x, y)

        # Kinetic energy
        speed = np.linalg.norm(impact_velocity)
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
            "kinetic_energy": kinetic_energy,
            "absorbed_energy": absorbed_energy,
            "remaining_energy": max(0.0, remaining_energy),
            "energy_absorption_ratio": (
                absorbed_energy / kinetic_energy if kinetic_energy > 0 else 0.0
            ),
        }


class TerrainGeometryGenerator:
    """Generate terrain geometry for physics engines.

    Creates meshes and heightfield data from terrain configuration
    for use in various physics engines.
    """

    def __init__(self, terrain: Terrain) -> None:
        """Initialize generator.

        Args:
            terrain: Terrain configuration
        """
        self.terrain = terrain

    def generate_mesh(self) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """Generate triangle mesh from terrain.

        Returns:
            Tuple of (vertices, triangles)
            - vertices: (N, 3) array of vertex positions
            - triangles: List of (i, j, k) vertex index tuples
        """
        elev = self.terrain.elevation
        n_rows, n_cols = elev.data.shape

        # Generate vertices
        vertices = []
        for j in range(n_rows):
            for i in range(n_cols):
                x = elev.origin_x + i * elev.resolution
                y = elev.origin_y + j * elev.resolution
                z = elev.data[j, i]
                vertices.append([x, y, z])

        vertices = np.array(vertices)

        # Generate triangles (2 per grid cell)
        triangles = []
        for j in range(n_rows - 1):
            for i in range(n_cols - 1):
                # Vertex indices for this cell
                v00 = j * n_cols + i
                v10 = j * n_cols + (i + 1)
                v01 = (j + 1) * n_cols + i
                v11 = (j + 1) * n_cols + (i + 1)

                # Two triangles per cell
                triangles.append((v00, v10, v11))
                triangles.append((v00, v11, v01))

        return vertices, triangles

    def generate_mujoco_hfield(self) -> tuple[np.ndarray, tuple[float, float]]:
        """Generate MuJoCo heightfield data.

        Returns:
            Tuple of (data, size)
            - data: 2D array of normalized heights [0, 1]
            - size: (width, length) in meters
        """
        elev = self.terrain.elevation

        # Normalize heights to [0, 1] range
        data = elev.data.copy()
        h_min = data.min()
        h_max = data.max()
        h_range = h_max - h_min if h_max > h_min else 1.0

        normalized = (data - h_min) / h_range

        size = (elev.width, elev.length)

        return normalized, size

    def generate_mujoco_xml(self, name: str = "terrain") -> str:
        """Generate MuJoCo XML snippet for terrain.

        Args:
            name: Name for the terrain geom

        Returns:
            XML string for inclusion in MuJoCo model
        """
        elev = self.terrain.elevation
        n_rows, n_cols = elev.data.shape

        # Calculate height range
        h_min = float(elev.data.min())
        h_max = float(elev.data.max())
        h_range = h_max - h_min if h_max > h_min else 0.1

        # Get average friction
        material = self.terrain.get_material(elev.width / 2, elev.length / 2)
        friction = material.friction_coefficient

        # Generate heightfield data as space-separated values
        # MuJoCo expects row-major order with values in [0, 1]
        normalized = (elev.data - h_min) / h_range if h_range > 0 else elev.data * 0

        # Create XML
        xml_parts = [
            "<asset>",
            f'  <hfield name="{name}_hfield" nrow="{n_rows}" ncol="{n_cols}" size="{elev.width/2} {elev.length/2} {h_range} 0.1"/>',
            "</asset>",
            "<worldbody>",
            f'  <geom name="{name}" type="hfield" hfield="{name}_hfield" pos="{elev.width/2} {elev.length/2} {h_min}" friction="{friction} 0.005 0.0001"/>',
            "</worldbody>",
        ]

        return "\n".join(xml_parts)

    def generate_urdf_collision(self, name: str = "terrain") -> str:
        """Generate URDF collision geometry for terrain.

        For simplicity, generates a box approximation. Full mesh
        support would require external mesh file.

        Args:
            name: Name for the collision geometry

        Returns:
            URDF XML snippet
        """
        elev = self.terrain.elevation
        h_max = float(elev.data.max())
        h_min = float(elev.data.min())

        xml = f"""<link name="{name}">
  <collision>
    <origin xyz="{elev.width/2} {elev.length/2} {(h_max+h_min)/2}" rpy="0 0 0"/>
    <geometry>
      <box size="{elev.width} {elev.length} {h_max - h_min + 0.1}"/>
    </geometry>
  </collision>
</link>"""

        return xml


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
    height = terrain.elevation.get_elevation(x, y)
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


def validate_terrain(
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


# Register terrain parameters with physics registry
def register_terrain_parameters() -> None:
    """Register terrain-related parameters with the physics registry."""
    from src.shared.python.physics_parameters import (
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


# Auto-register on import
try:
    register_terrain_parameters()
except Exception as e:
    logger.debug(f"Could not register terrain parameters: {e}")
