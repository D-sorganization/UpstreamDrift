# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

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

from src.api.middleware.error_handler import handle_api_errors
from src.shared.python.core.contracts import precondition
from src.shared.python.physics.terrain import (
    MATERIALS,
    TERRAIN_MATERIAL_MAP,
    Terrain,
    TerrainType,
    create_flat_terrain,
)
from src.shared.python.physics.terrain_presets import (
    ENVIRONMENT_PRESETS as SHARED_ENVIRONMENT_PRESETS,
    build_environment_preset,
    get_environment_preset_names,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terrain", tags=["terrain"])


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
#  In-memory terrain state (singleton holder avoids 'global')
# ──────────────────────────────────────────────────────────────

_terrain_state: dict[str, Terrain | None] = {"active": None}


def _get_active_terrain() -> Terrain:
    """Get the active terrain, creating a default if none exists."""
    if _terrain_state["active"] is None:
        _terrain_state["active"] = create_flat_terrain(
            name="default_fairway",
            width=100.0,
            length=200.0,
            terrain_type=TerrainType.FAIRWAY,
        )
    terrain = _terrain_state["active"]
    assert terrain is not None  # for mypy; guaranteed by preceding assignment
    return terrain


# ──────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────


@router.get("/presets", response_model=list[EnvironmentPreset])
@handle_api_errors
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
        for name, info in SHARED_ENVIRONMENT_PRESETS.items()
    ]


@router.post("/load", response_model=dict[str, Any])
@handle_api_errors
@precondition(
    lambda request: (
        request is not None
        and request.preset is not None
        and len(request.preset.strip()) > 0
    ),
    "Environment request must contain a non-empty preset name",
)
async def load_environment(request: CreateEnvironmentRequest) -> dict[str, Any]:
    """Load an environment preset as the active terrain."""
    preset_name = request.preset.lower().strip()
    if preset_name not in SHARED_ENVIRONMENT_PRESETS:
        return {
            "success": False,
            "error": f"Unknown preset '{request.preset}'. "
            f"Available: {get_environment_preset_names()}",
        }

    preset_info = SHARED_ENVIRONMENT_PRESETS[preset_name]
    width = request.width or preset_info["width"]
    length = request.length or preset_info["length"]

    _terrain_state["active"] = build_environment_preset(
        preset_name,
        width=width,
        length=length,
        slope=request.slope_angle_deg,
        direction=request.slope_direction_deg,
    )

    terrain = _terrain_state["active"]
    logger.info("Loaded environment preset: %s (%gx%g m)", preset_name, width, length)

    return {
        "success": True,
        "name": terrain.name,  # type: ignore[union-attr]
        "width_m": width,
        "length_m": length,
        "terrain_types": preset_info["terrain_types"],
    }


@router.post("/query", response_model=TerrainQueryResponse)
@handle_api_errors
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
@handle_api_errors
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
@handle_api_errors
async def list_terrain_types() -> list[str]:
    """List all available terrain types."""
    return [t.name.lower() for t in TerrainType]


@router.get("/active", response_model=dict[str, Any])
@handle_api_errors
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
