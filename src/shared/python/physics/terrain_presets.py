"""Shared procedural terrain presets for API routes and launcher tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.physics.terrain import (
    ElevationMap,
    Terrain,
    TerrainPatch,
    TerrainRegion,
    TerrainType,
)

PresetInfo = dict[str, Any]
TerrainBuilder = Callable[[float, float, float, float], Terrain]

ENVIRONMENT_PRESETS: dict[str, PresetInfo] = {
    "putting_green": {
        "description": "Close-range putting green with detailed surface (1-30 ft)",
        "width": 10.0,
        "length": 15.0,
        "terrain_types": ["green", "fringe"],
        "builder": "_build_putting_green",
    },
    "fairway": {
        "description": "Medium-range fairway with gentle slopes (50-200 yards)",
        "width": 50.0,
        "length": 200.0,
        "terrain_types": ["fairway", "rough", "bunker"],
        "builder": "_build_fairway",
    },
    "driving_range": {
        "description": "Long-range practice environment (100-300+ yards)",
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


@precondition(
    lambda width, length, slope, direction: width > 0 and length > 0,
    "Terrain width and length must be positive",
)
@postcondition(
    lambda result: result is not None and bool(result.name),
    "Built terrain must have a valid name",
)
def _build_putting_green(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
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


@precondition(
    lambda width, length, slope, direction: width > 0 and length > 0,
    "Terrain width and length must be positive",
)
def _build_fairway(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
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


@precondition(
    lambda width, length, slope, direction: width > 0 and length > 0,
    "Terrain width and length must be positive",
)
def _build_driving_range(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
    elevation = ElevationMap.flat(width=width, length=length, resolution=2.0)
    patches = [
        TerrainPatch(TerrainType.TEE, 0, width, 0, 5.0),
        TerrainPatch(TerrainType.FAIRWAY, 0, width, 5, length),
        TerrainPatch(TerrainType.ROUGH, 0, 5, 5, length),
        TerrainPatch(TerrainType.ROUGH, width - 5, width, 5, length),
    ]
    return Terrain(name="driving_range", elevation=elevation, patches=patches)


@precondition(
    lambda width, length, slope, direction: width > 0 and length > 0,
    "Terrain width and length must be positive",
)
def _build_bunker(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
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


@precondition(
    lambda width, length, slope, direction: width > 0 and length > 0,
    "Terrain width and length must be positive",
)
def _build_rough(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
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


@precondition(
    lambda width, length, slope, direction: width > 0 and length > 0,
    "Terrain width and length must be positive",
)
def _build_full_hole(
    width: float, length: float, slope: float, direction: float
) -> Terrain:
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


TERRAIN_PRESET_BUILDERS: Mapping[str, TerrainBuilder] = {
    "putting_green": _build_putting_green,
    "fairway": _build_fairway,
    "driving_range": _build_driving_range,
    "bunker": _build_bunker,
    "rough": _build_rough,
    "full_hole": _build_full_hole,
}


def get_environment_preset_names() -> list[str]:
    """Return preset names with every entry backed by a builder."""
    return sorted(TERRAIN_PRESET_BUILDERS)


@precondition(
    lambda preset, width=None, length=None, slope=0.0, direction=0.0: (
        isinstance(preset, str) and bool(preset.strip())
    ),
    "Terrain preset name must be a non-empty string",
)
@postcondition(
    lambda result: result is not None and bool(result.name),
    "Built terrain must have a valid name",
)
def build_environment_preset(
    preset: str,
    *,
    width: float | None = None,
    length: float | None = None,
    slope: float = 0.0,
    direction: float = 0.0,
) -> Terrain:
    """Build a terrain preset by name with optional size and slope overrides."""
    preset_name = preset.lower().strip()
    if preset_name not in TERRAIN_PRESET_BUILDERS:
        raise ValueError(
            f"Unknown terrain preset '{preset}'. "
            f"Available: {get_environment_preset_names()}"
        )

    info = ENVIRONMENT_PRESETS[preset_name]
    resolved_width = float(width if width is not None else info["width"])
    resolved_length = float(length if length is not None else info["length"])
    if resolved_width <= 0 or resolved_length <= 0:
        raise ValueError("terrain width and length must be positive")

    return TERRAIN_PRESET_BUILDERS[preset_name](
        resolved_width,
        resolved_length,
        slope,
        direction,
    )
