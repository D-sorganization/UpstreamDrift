"""Shared Topographical Data Module.

This module provides reusable topographical/elevation data handling that can be
used across multiple physics models including:
- Putting green surfaces
- Golf course terrain
- General ground surfaces for ball flight

Supported formats:
- NumPy arrays (.npy)
- CSV files (x, y, elevation columns)
- GeoTIFF (.tif, .tiff) - requires rasterio
- JSON contour definitions
- Image-based heightmaps (.png, .jpg)

Design by Contract:
    - All elevations are in meters
    - All coordinates are in meters unless specified
    - Interpolation is provided for smooth surface queries
"""

from __future__ import annotations

from ._topography_data import TopographyData
from ._topography_factory import (
    create_flat_terrain,
    create_sloped_terrain,
    create_undulating_terrain,
)
from ._topography_types import ElevationPoint, TopographyBounds, TopographyProvider

__all__ = [
    "ElevationPoint",
    "TopographyBounds",
    "TopographyProvider",
    "TopographyData",
    "create_flat_terrain",
    "create_sloped_terrain",
    "create_undulating_terrain",
]
