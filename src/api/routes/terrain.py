"""Terrain and environment API routes for Golf Modeling Suite.

Provides engine-agnostic terrain queries, preset environment loading,
and surface property inspection.

Fixes #1145
Fixes #1142
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.shared.python.physics.terrain import (
    MATERIALS,
    TERRAIN_MATERIAL_MAP,
    ElevationMap,
    Terrain,
    TerrainPatch,
    TerrainRegion,
    TerrainType,
    create_flat_terrain,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/terrain", tags=["terrain"])


# ──────────────────────────────────────────────────────────────
#  Pydantic Models
# ──────────────────────────────────────────────────────────────


class TerrainQueryRequest(BaseModel):
    """Request for querying terrain properties at a point."""

    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")


class TerrainQueryResponse(BaseModel):
    """Response with terrain properties at a point."""

    x: float
    y: float
    elevation: float
    slope_angle_deg: float
    terrain_type: str
    friction: float
    restitution: float
    rolling_resistance: float


class EnvironmentPreset(BaseModel):
    """An available environment preset."""

    name: str = Field(..., description="Preset identifier")
    description: str = Field(..., description="Human-readable description")
    terrain_types: list[str] = Field(
        ..., description="Terrain types in this environment"
    )
    width_m: float = Field(..., description="Width in meters")
    length_m: float = Field(..., description="Length in meters")


class CreateEnvironmentRequest(BaseModel):
    """Request to create a terrain environment."""

    preset: str = Field(
        ..., description="Preset name (putting_green, fairway, driving_range, etc.)"
    )
    width: float | None = Field(None, description="Override width (meters)")
    length: float | None = Field(None, description="Override length (meters)")
    slope_angle_deg: float = Field(0.0, description="Slope angle (degrees)")
    slope_direction_deg: float = Field(0.0, description="Slope direction (degrees)")


class SurfaceMaterialResponse(BaseModel):
    """Surface material properties."""

    name: str
    friction_coefficient: float
    rolling_resistance: float
    restitution: float
    hardness: float
    grass_height_m: float
    compressibility: float


# ──────────────────────────────────────────────────────────────
#  In-memory terrain state
# ──────────────────────────────────────────────────────────────

_active_terrain: Terrain | None = None


def _get_active_terrain() -> Terrain:
    """Get the active terrain, creating a default if none exists."""
    global _active_terrain  # noqa: PLW0603
    if _active_terrain is None:
        _active_terrain = create_flat_terrain(
            name="default_fairway",
            width=100.0,
            length=200.0,
            terrain_type=TerrainType.FAIRWAY,
        )
    return _active_terrain


# ──────────────────────────────────────────────────────────────
#  Environment Presets
# ──────────────────────────────────────────────────────────────

ENVIRONMENT_PRESETS: dict[str, dict[str, Any]] = {
    "putting_green": {
        "description": "Close-range putting green with detailed surface (1–30 ft)",
        "width": 10.0,
        "length": 15.0,
        "terrain_types": ["green", "fringe"],
        "builder": "_build_putting_green",
    },
    "fairway": {
        "description": "Medium-range fairway with gentle slopes (50–200 yards)",
        "width": 50.0,
        "length": 200.0,
        "terrain_types": ["fairway", "rough", "bunker"],
        "builder": "_build_fairway",
    },
    "driving_range": {
        "description": "Long-range practice environment (100–300+ yards)",
        "width": 80.0,
        "length": 300.0,
        "terrain_types": ["tee", "fairway", "rough"],
        "builder": "_build_driving_range",
    },
    "bunker": {
        "description": "Sand bunker practice area with varied lip heights",
        "width": 20.0,
        "length": 20.0,
        "terrain_types": ["bunker", "green", "fringe"],
        "builder": "_build_bunker",
    },
    "rough": {
        "description": "Thick rough practice area with high grass",
        "width": 30.0,
        "length": 40.0,
        "terrain_types": ["rough", "fairway"],
        "builder": "_build_rough",
    },
    "full_hole": {
        "description": "Complete golf hole from tee to green (par 4, ~370 yards)",
        "width": 60.0,
        "length": 340.0,
        "terrain_types": ["tee", "fairway", "rough", "bunker", "fringe", "green"],
        "builder": "_build_full_hole",
    },
}


def _build_putting_green(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
    """Build a putting green environment."""
    elevation = ElevationMap.sloped(
        width=width,
        length=length,
        resolution=0.1,
        slope_angle_deg=slope if slope != 0 else 1.5,
        slope_direction_deg=direction,
    )
    patches = [
        TerrainPatch(TerrainType.GREEN, 0, width, 0, length),
        TerrainPatch(TerrainType.FRINGE, 0, width, 0, 1.0),
        TerrainPatch(TerrainType.FRINGE, 0, width, length - 1.0, length),
    ]
    return Terrain(name="putting_green", elevation=elevation, patches=patches)


def _build_fairway(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
    """Build a fairway environment."""
    elevation = ElevationMap.sloped(
        width=width,
        length=length,
        resolution=1.0,
        slope_angle_deg=slope if slope != 0 else 0.5,
        slope_direction_deg=direction,
    )
    patches = [
        TerrainPatch(TerrainType.FAIRWAY, 5, width - 5, 0, length),
        TerrainPatch(TerrainType.ROUGH, 0, 5, 0, length),
        TerrainPatch(TerrainType.ROUGH, width - 5, width, 0, length),
    ]
    regions = [
        TerrainRegion.circle(TerrainType.BUNKER, width / 2 + 8, length * 0.6, 5.0),
        TerrainRegion.circle(TerrainType.BUNKER, width / 2 - 10, length * 0.75, 4.0),
    ]
    return Terrain(
        name="fairway",
        elevation=elevation,
        patches=patches,
        regions=regions,
    )


def _build_driving_range(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
    """Build a driving range environment."""
    elevation = ElevationMap.flat(width=width, length=length, resolution=2.0)
    patches = [
        TerrainPatch(TerrainType.TEE, 0, width, 0, 5.0),
        TerrainPatch(TerrainType.FAIRWAY, 0, width, 5, length),
        TerrainPatch(TerrainType.ROUGH, 0, 5, 5, length),
        TerrainPatch(TerrainType.ROUGH, width - 5, width, 5, length),
    ]
    return Terrain(name="driving_range", elevation=elevation, patches=patches)


def _build_bunker(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
    """Build a bunker practice environment."""
    elevation = ElevationMap.flat(width=width, length=length, resolution=0.5)
    patches = [
        TerrainPatch(TerrainType.GREEN, 0, width, length / 2, length),
        TerrainPatch(TerrainType.FRINGE, 0, width, length / 2 - 2, length / 2),
    ]
    regions = [
        TerrainRegion.circle(TerrainType.BUNKER, width / 2, length / 4, 6.0),
    ]
    return Terrain(
        name="bunker",
        elevation=elevation,
        patches=patches,
        regions=regions,
        default_type=TerrainType.ROUGH,
    )


def _build_rough(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
    """Build a rough practice environment."""
    elevation = ElevationMap.sloped(
        width=width,
        length=length,
        resolution=0.5,
        slope_angle_deg=slope if slope != 0 else 2.0,
        slope_direction_deg=direction,
    )
    patches = [
        TerrainPatch(TerrainType.ROUGH, 0, width, 0, length * 0.7),
        TerrainPatch(TerrainType.FAIRWAY, 5, width - 5, length * 0.7, length),
    ]
    return Terrain(
        name="rough",
        elevation=elevation,
        patches=patches,
        default_type=TerrainType.ROUGH,
    )


def _build_full_hole(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
    """Build a complete golf hole (par 4)."""
    elevation = ElevationMap.sloped(
        width=width,
        length=length,
        resolution=2.0,
        slope_angle_deg=slope if slope != 0 else 0.3,
        slope_direction_deg=direction,
    )
    patches = [
        TerrainPatch(TerrainType.TEE, 20, 40, 0, 10),
        TerrainPatch(TerrainType.FAIRWAY, 10, 50, 10, length - 20),
        TerrainPatch(TerrainType.ROUGH, 0, 10, 10, length - 20),
        TerrainPatch(TerrainType.ROUGH, 50, width, 10, length - 20),
    ]
    regions = [
        TerrainRegion.circle(TerrainType.GREEN, width / 2, length - 12, 8.0),
        TerrainRegion.circle(TerrainType.FRINGE, width / 2, length - 12, 10.0),
        TerrainRegion.circle(TerrainType.BUNKER, width / 2 + 12, length - 15, 4.0),
        TerrainRegion.circle(TerrainType.BUNKER, width / 2 - 8, length - 8, 3.0),
    ]
    return Terrain(
        name="full_hole",
        elevation=elevation,
        patches=patches,
        regions=regions,
        default_type=TerrainType.ROUGH,
    )


_BUILDERS = {
    "putting_green": _build_putting_green,
    "fairway": _build_fairway,
    "driving_range": _build_driving_range,
    "bunker": _build_bunker,
    "rough": _build_rough,
    "full_hole": _build_full_hole,
}


# ──────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────


@router.get("/presets", response_model=list[EnvironmentPreset])
async def list_presets() -> list[EnvironmentPreset]:
    """List available environment presets."""
    return [
        EnvironmentPreset(
            name=name,
            description=info["description"],
            terrain_types=info["terrain_types"],
            width_m=info["width"],
            length_m=info["length"],
        )
        for name, info in ENVIRONMENT_PRESETS.items()
    ]


@router.post("/load", response_model=dict[str, Any])
async def load_environment(request: CreateEnvironmentRequest) -> dict[str, Any]:
    """Load an environment preset as the active terrain."""
    global _active_terrain  # noqa: PLW0603

    preset_name = request.preset.lower().strip()
    if preset_name not in _BUILDERS:
        return {
            "success": False,
            "error": f"Unknown preset '{request.preset}'. "
            f"Available: {sorted(_BUILDERS.keys())}",
        }

    preset_info = ENVIRONMENT_PRESETS[preset_name]
    width = request.width or preset_info["width"]
    length = request.length or preset_info["length"]

    builder = _BUILDERS[preset_name]
    _active_terrain = builder(
        width, length, request.slope_angle_deg, request.slope_direction_deg
    )

    logger.info("Loaded environment preset: %s (%gx%g m)", preset_name, width, length)

    return {
        "success": True,
        "name": _active_terrain.name,
        "width_m": width,
        "length_m": length,
        "terrain_types": preset_info["terrain_types"],
    }


@router.post("/query", response_model=TerrainQueryResponse)
async def query_terrain(request: TerrainQueryRequest) -> TerrainQueryResponse:
    """Query terrain properties at a specific point."""
    terrain = _get_active_terrain()

    try:
        elevation = terrain.elevation.get_elevation(request.x, request.y)
        slope_angle = terrain.elevation.get_slope_angle(request.x, request.y)
        terrain_type = terrain.get_terrain_type(request.x, request.y)
        material = terrain.get_material(request.x, request.y)
    except ValueError as exc:
        # Coordinates out of bounds — clamp to edge
        logger.warning("Terrain query out of bounds: %s", exc)
        elevation = 0.0
        slope_angle = 0.0
        terrain_type = terrain.default_type
        material_name = TERRAIN_MATERIAL_MAP.get(terrain_type, "rough")
        material = MATERIALS[material_name]

    return TerrainQueryResponse(
        x=request.x,
        y=request.y,
        elevation=elevation,
        slope_angle_deg=slope_angle,
        terrain_type=terrain_type.name.lower(),
        friction=material.friction_coefficient,
        restitution=material.restitution,
        rolling_resistance=material.rolling_resistance,
    )


@router.get("/materials", response_model=list[SurfaceMaterialResponse])
async def list_materials() -> list[SurfaceMaterialResponse]:
    """List all available surface materials and their properties."""
    return [
        SurfaceMaterialResponse(
            name=mat.name,
            friction_coefficient=mat.friction_coefficient,
            rolling_resistance=mat.rolling_resistance,
            restitution=mat.restitution,
            hardness=mat.hardness,
            grass_height_m=mat.grass_height_m,
            compressibility=mat.compressibility,
        )
        for mat in MATERIALS.values()
    ]


@router.get("/types", response_model=list[str])
async def list_terrain_types() -> list[str]:
    """List all available terrain types."""
    return [t.name.lower() for t in TerrainType]


@router.get("/active", response_model=dict[str, Any])
async def get_active_terrain() -> dict[str, Any]:
    """Get information about the currently active terrain."""
    terrain = _get_active_terrain()
    patch_count = len(terrain.patches)
    region_count = len(terrain.regions)
    return {
        "name": terrain.name,
        "width_m": terrain.elevation.width,
        "length_m": terrain.elevation.length,
        "resolution_m": terrain.elevation.resolution,
        "default_type": terrain.default_type.name.lower(),
        "patch_count": patch_count,
        "region_count": region_count,
    }
