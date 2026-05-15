from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
)
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    GrassType,
    TurfProperties,
)

if TYPE_CHECKING:
    from src.engines.physics_engines.putting_green.python._sim_core import (
        PuttingGreenSimulator,
    )


def load_from_path(sim: PuttingGreenSimulator, path: str) -> None:
    """Load green configuration from file.

    Supports JSON configuration files.

    Args:
        sim: The simulator instance
        path: Path to configuration file
    """
    if not (path is not None):
        raise ValueError("path must be provided")
    if not (path is not None):
        raise ValueError("path must be provided")
    filepath = Path(path)

    with open(filepath) as f:
        data = json.load(f)

    load_from_data(sim, data)


def load_from_string(
    sim: PuttingGreenSimulator, content: str, extension: str | None = None
) -> None:
    """Load green configuration from string.

    Args:
        sim: The simulator instance
        content: Configuration content
        extension: Format hint (e.g., "json")
    """
    if not (content is not None):
        raise ValueError("content must be provided")
    if not (content is not None):
        raise ValueError("content must be provided")
    data = json.loads(content)
    load_from_data(sim, data)


def load_from_data(sim: PuttingGreenSimulator, data: dict[str, Any]) -> None:
    """Load configuration from dictionary."""
    if not (data is not None):
        raise ValueError("data must be provided")
    if not (data is not None):
        raise ValueError("data must be provided")
    if "green" in data:
        green_data = data["green"]

        turf_data = green_data.get("turf", {})
        if "stimp_rating" in turf_data:
            grass_type = GrassType(turf_data.get("grass_type", "bent_grass"))
            turf = TurfProperties(
                stimp_rating=turf_data["stimp_rating"],
                grass_type=grass_type,
            )
        else:
            turf = TurfProperties()

        sim.green = GreenSurface(
            width=green_data.get("width", 20.0),
            height=green_data.get("height", 20.0),
            turf=turf,
        )

        if "hole_position" in green_data:
            sim.green.set_hole_position(np.array(green_data["hole_position"]))

        if "slopes" in green_data:
            for s in green_data["slopes"]:
                sim.green.add_slope_region(
                    SlopeRegion(
                        center=np.array(s["center"]),
                        radius=s["radius"],
                        slope_direction=np.array(s["direction"]),
                        slope_magnitude=s["magnitude"],
                    )
                )

    sim._physics = BallRollPhysics(
        green=sim.green,
        integrator=sim.config.integrator,
    )


def load_topographical_data(
    sim: PuttingGreenSimulator,
    path: str,
    width: float | None = None,
    height: float | None = None,
) -> None:
    """Load topographical/elevation data.

    Args:
        sim: The simulator instance
        path: Path to topographical data file
        width: Physical width [m] (uses current if None)
        height: Physical height [m] (uses current if None)
    """
    if not (path is not None):
        raise ValueError("path must be provided")
    if not (path is not None):
        raise ValueError("path must be provided")
    filepath = Path(path)
    suffix = filepath.suffix.lower()

    if width is not None:
        sim.green.width = width
    if height is not None:
        sim.green.height = height

    if suffix == ".npy":
        heightmap = np.load(filepath, allow_pickle=False)
        sim.green.set_heightmap(heightmap)
    elif suffix == ".csv" or suffix in (".tif", ".tiff"):
        sim.green.load_from_file(filepath)
    else:
        sim.green.load_from_file(filepath)

    sim._physics = BallRollPhysics(
        green=sim.green,
        integrator=sim.config.integrator,
    )
