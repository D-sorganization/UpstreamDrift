from dataclasses import dataclass
from typing import Any

from .materials import MATERIALS, TERRAIN_MATERIAL_MAP, SurfaceMaterial, TerrainType


@dataclass
class TerrainPatch:
    """A rectangular region with uniform terrain type."""

    terrain_type: TerrainType
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    material: SurfaceMaterial | None = None

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within this patch."""
        if x is None:
            raise ValueError("x must be provided")
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def get_material(self) -> SurfaceMaterial:
        """Get the material for this patch."""
        if self.material is not None:
            return self.material
        material_name = TERRAIN_MATERIAL_MAP.get(self.terrain_type, "rough")
        return MATERIALS[material_name]

    def to_dict(self) -> dict[str, Any]:
        """Serialize patch to dictionary."""
        result = {
            "terrain_type": self.terrain_type.name.lower(),
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }
        if self.material is not None:
            result["material"] = {
                "name": self.material.name,
                "friction_coefficient": self.material.friction_coefficient,
                "rolling_resistance": self.material.rolling_resistance,
                "restitution": self.material.restitution,
                "hardness": self.material.hardness,
                "grass_height_m": self.material.grass_height_m,
            }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerrainPatch":
        """Create patch from dictionary."""
        if data is None:
            raise ValueError("data must be provided")
        terrain_type = TerrainType[data["terrain_type"].upper()]
        material = None
        if "material" in data:
            material = SurfaceMaterial(**data["material"])
        return cls(
            terrain_type=terrain_type,
            x_min=data["x_min"],
            x_max=data["x_max"],
            y_min=data["y_min"],
            y_max=data["y_max"],
            material=material,
        )


@dataclass
class TerrainRegion:
    """A terrain region with complex shape (circle, polygon, etc.)."""

    terrain_type: TerrainType
    shape_type: str
    shape_data: dict[str, Any]
    material: SurfaceMaterial | None = None

    @classmethod
    def circle(
        cls,
        terrain_type: TerrainType,
        center_x: float,
        center_y: float,
        radius: float,
        material: SurfaceMaterial | None = None,
    ) -> "TerrainRegion":
        """Create a circular terrain region."""
        return cls(
            terrain_type=terrain_type,
            shape_type="circle",
            shape_data={"center_x": center_x, "center_y": center_y, "radius": radius},
            material=material,
        )

    @classmethod
    def polygon(
        cls,
        terrain_type: TerrainType,
        vertices: list[tuple[float, float]],
        material: SurfaceMaterial | None = None,
    ) -> "TerrainRegion":
        """Create a polygon terrain region."""
        return cls(
            terrain_type=terrain_type,
            shape_type="polygon",
            shape_data={"vertices": vertices},
            material=material,
        )

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within this region."""
        if x is None:
            raise ValueError("x must be provided")
        if self.shape_type == "circle":
            cx = self.shape_data["center_x"]
            cy = self.shape_data["center_y"]
            r = self.shape_data["radius"]
            return (x - cx) ** 2 + (y - cy) ** 2 <= r**2

        if self.shape_type == "polygon":
            vertices = self.shape_data["vertices"]
            return self._point_in_polygon(x, y, vertices)

        return False

    @staticmethod
    def _point_in_polygon(
        x: float, y: float, vertices: list[tuple[float, float]]
    ) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        if x is None:
            raise ValueError("x must be provided")
        n = len(vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi
            ):
                inside = not inside
            j = i

        return inside

    def get_material(self) -> SurfaceMaterial:
        """Get the material for this region."""
        if self.material is not None:
            return self.material
        material_name = TERRAIN_MATERIAL_MAP.get(self.terrain_type, "rough")
        return MATERIALS[material_name]

    def to_dict(self) -> dict[str, Any]:
        """Serialize region to dictionary."""
        result: dict[str, Any] = {
            "terrain_type": self.terrain_type.name.lower(),
            "shape_type": self.shape_type,
            "shape_data": self.shape_data,
        }
        if self.material is not None:
            result["material"] = {
                "name": self.material.name,
                "friction": self.material.friction_coefficient,
                "restitution": self.material.restitution,
            }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerrainRegion":
        """Deserialize region from dictionary."""
        if data is None:
            raise ValueError("data must be provided")
        material = None
        if "material" in data:
            mat = data["material"]
            material = SurfaceMaterial(
                name=mat["name"],
                friction_coefficient=mat["friction"],
                restitution=mat["restitution"],
            )
        return cls(
            terrain_type=TerrainType[data["terrain_type"].upper()],
            shape_type=data["shape_type"],
            shape_data=data["shape_data"],
            material=material,
        )
