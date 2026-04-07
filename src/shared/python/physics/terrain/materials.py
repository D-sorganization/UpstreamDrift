from dataclasses import dataclass
from enum import Enum, auto


class TerrainType(Enum):
    """Golf course terrain types."""

    FAIRWAY = auto()
    ROUGH = auto()
    GREEN = auto()
    BUNKER = auto()
    TEE = auto()
    FRINGE = auto()
    WATER = auto()
    CART_PATH = auto()
    OUT_OF_BOUNDS = auto()


@dataclass
class SurfaceMaterial:
    """Physical properties of a surface material."""

    name: str
    friction_coefficient: float = 0.5
    rolling_resistance: float = 0.1
    restitution: float = 0.6
    hardness: float = 0.7
    grass_height_m: float = 0.0
    compressibility: float = 0.0
    compression_damping: float = 0.3
    turf_density: float = 0.0
    moisture_content: float = 0.3

    def __post_init__(self) -> None:
        """Validate material properties."""
        if self.friction_coefficient < 0:
            raise ValueError("friction_coefficient must be non-negative")
        if self.rolling_resistance < 0:
            raise ValueError("rolling_resistance must be non-negative")
        if not 0 <= self.restitution <= 1:
            raise ValueError("restitution must be between 0 and 1")
        if not 0 <= self.hardness <= 1:
            raise ValueError("hardness must be between 0 and 1")
        if self.grass_height_m < 0:
            raise ValueError("grass_height_m must be non-negative")
        if not 0 <= self.compressibility <= 1:
            raise ValueError("compressibility must be between 0 and 1")
        if not 0 <= self.compression_damping <= 1:
            raise ValueError("compression_damping must be between 0 and 1")
        if not 0 <= self.moisture_content <= 1:
            raise ValueError("moisture_content must be between 0 and 1")

    @property
    def is_compressible(self) -> bool:
        """Check if this material is compressible."""
        return self.compressibility > 0.01

    def get_effective_stiffness(self, base_stiffness: float = 1e5) -> float:
        """Get effective stiffness considering compressibility."""
        return base_stiffness * (1.0 - 0.9 * self.compressibility)

    def get_max_compression_depth(self) -> float:
        """Get maximum compression depth in meters."""
        base_depth = self.grass_height_m * 0.8 * self.compressibility
        moisture_factor = 1.0 + 0.5 * self.moisture_content
        return base_depth * moisture_factor


MATERIALS: dict[str, SurfaceMaterial] = {
    "fairway": SurfaceMaterial(
        name="fairway",
        friction_coefficient=0.45,
        rolling_resistance=0.08,
        restitution=0.65,
        hardness=0.75,
        grass_height_m=0.015,
        compressibility=0.15,
        compression_damping=0.25,
        turf_density=120.0,
        moisture_content=0.3,
    ),
    "rough": SurfaceMaterial(
        name="rough",
        friction_coefficient=0.55,
        rolling_resistance=0.20,
        restitution=0.45,
        hardness=0.65,
        grass_height_m=0.050,
        compressibility=0.35,
        compression_damping=0.40,
        turf_density=80.0,
        moisture_content=0.35,
    ),
    "green": SurfaceMaterial(
        name="green",
        friction_coefficient=0.35,
        rolling_resistance=0.05,
        restitution=0.70,
        hardness=0.80,
        grass_height_m=0.004,
        compressibility=0.05,
        compression_damping=0.15,
        turf_density=200.0,
        moisture_content=0.25,
    ),
    "bunker": SurfaceMaterial(
        name="bunker",
        friction_coefficient=0.80,
        rolling_resistance=0.40,
        restitution=0.30,
        hardness=0.30,
        grass_height_m=0.0,
        compressibility=0.70,
        compression_damping=0.60,
        turf_density=1500.0,
        moisture_content=0.10,
    ),
    "tee": SurfaceMaterial(
        name="tee",
        friction_coefficient=0.45,
        rolling_resistance=0.08,
        restitution=0.65,
        hardness=0.80,
        grass_height_m=0.010,
        compressibility=0.10,
        compression_damping=0.20,
        turf_density=150.0,
        moisture_content=0.25,
    ),
    "fringe": SurfaceMaterial(
        name="fringe",
        friction_coefficient=0.42,
        rolling_resistance=0.10,
        restitution=0.60,
        hardness=0.75,
        grass_height_m=0.012,
        compressibility=0.12,
        compression_damping=0.22,
        turf_density=140.0,
        moisture_content=0.28,
    ),
    "cart_path": SurfaceMaterial(
        name="cart_path",
        friction_coefficient=0.70,
        rolling_resistance=0.02,
        restitution=0.80,
        hardness=0.95,
        grass_height_m=0.0,
        compressibility=0.0,
        compression_damping=0.0,
        turf_density=0.0,
        moisture_content=0.0,
    ),
    "water": SurfaceMaterial(
        name="water",
        friction_coefficient=0.01,
        rolling_resistance=0.90,
        restitution=0.10,
        hardness=0.0,
        grass_height_m=0.0,
        compressibility=1.0,
        compression_damping=0.90,
        turf_density=1000.0,
        moisture_content=1.0,
    ),
    "soft_turf": SurfaceMaterial(
        name="soft_turf",
        friction_coefficient=0.50,
        rolling_resistance=0.15,
        restitution=0.50,
        hardness=0.50,
        grass_height_m=0.025,
        compressibility=0.45,
        compression_damping=0.45,
        turf_density=100.0,
        moisture_content=0.45,
    ),
    "wet_fairway": SurfaceMaterial(
        name="wet_fairway",
        friction_coefficient=0.35,
        rolling_resistance=0.12,
        restitution=0.55,
        hardness=0.60,
        grass_height_m=0.015,
        compressibility=0.30,
        compression_damping=0.35,
        turf_density=120.0,
        moisture_content=0.70,
    ),
    "divot": SurfaceMaterial(
        name="divot",
        friction_coefficient=0.60,
        rolling_resistance=0.25,
        restitution=0.40,
        hardness=0.40,
        grass_height_m=0.005,
        compressibility=0.50,
        compression_damping=0.50,
        turf_density=90.0,
        moisture_content=0.35,
    ),
}

TERRAIN_MATERIAL_MAP: dict[TerrainType, str] = {
    TerrainType.FAIRWAY: "fairway",
    TerrainType.ROUGH: "rough",
    TerrainType.GREEN: "green",
    TerrainType.BUNKER: "bunker",
    TerrainType.TEE: "tee",
    TerrainType.FRINGE: "fringe",
    TerrainType.CART_PATH: "cart_path",
    TerrainType.WATER: "water",
    TerrainType.OUT_OF_BOUNDS: "rough",
}
