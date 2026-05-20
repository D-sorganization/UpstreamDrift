from __future__ import annotations

import numpy as np

from ._topography_data import TopographyData
from ._topography_types import TopographyBounds


def create_flat_terrain(
    width: float = 100.0,
    height: float = 100.0,
    elevation: float = 0.0,
) -> TopographyData:
    """Create flat terrain.

    Args:
        width: Width [m]
        height: Height [m]
        elevation: Constant elevation [m]

    Returns:
        TopographyData with flat surface
    """
    if width is None:
        raise ValueError("width must be provided")
    topo = TopographyData(
        bounds=TopographyBounds(min_x=0, max_x=width, min_y=0, max_y=height)
    )
    heightmap = np.full((10, 10), elevation)
    topo.set_heightmap(heightmap, smooth=False)
    return topo


def create_sloped_terrain(
    width: float = 100.0,
    height: float = 100.0,
    slope_direction: np.ndarray = np.array([1.0, 0.0]),
    slope_magnitude: float = 0.02,
    base_elevation: float = 0.0,
) -> TopographyData:
    """Create uniformly sloped terrain.

    Args:
        width: Width [m]
        height: Height [m]
        slope_direction: Direction of downhill slope
        slope_magnitude: Steepness (rise/run)
        base_elevation: Elevation at origin [m]

    Returns:
        TopographyData with sloped surface
    """
    if width is None:
        raise ValueError("width must be provided")
    topo = TopographyData(
        bounds=TopographyBounds(min_x=0, max_x=width, min_y=0, max_y=height)
    )

    slope_dir = slope_direction / np.linalg.norm(slope_direction)

    resolution = 50
    x = np.linspace(0, width, resolution)
    y = np.linspace(0, height, resolution)
    X, Y = np.meshgrid(x, y)

    heightmap = base_elevation - slope_magnitude * (X * slope_dir[0] + Y * slope_dir[1])

    topo.set_heightmap(heightmap, smooth=False)
    return topo


def create_undulating_terrain(
    width: float = 100.0,
    height: float = 100.0,
    amplitude: float = 1.0,
    wavelength: float = 20.0,
    base_elevation: float = 0.0,
) -> TopographyData:
    """Create undulating (sine wave) terrain.

    Args:
        width: Width [m]
        height: Height [m]
        amplitude: Wave amplitude [m]
        wavelength: Wave wavelength [m]
        base_elevation: Mean elevation [m]

    Returns:
        TopographyData with undulating surface
    """
    if width is None:
        raise ValueError("width must be provided")
    topo = TopographyData(
        bounds=TopographyBounds(min_x=0, max_x=width, min_y=0, max_y=height)
    )

    resolution = 100
    x = np.linspace(0, width, resolution)
    y = np.linspace(0, height, resolution)
    X, Y = np.meshgrid(x, y)

    k = 2 * np.pi / wavelength
    heightmap = base_elevation + amplitude * (
        np.sin(k * X) * np.cos(k * Y) + 0.5 * np.sin(2 * k * X)
    )

    topo.set_heightmap(heightmap, smooth=True)
    return topo
