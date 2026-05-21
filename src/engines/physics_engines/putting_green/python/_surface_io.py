from __future__ import annotations

from pathlib import Path

import numpy as np

from src.engines.physics_engines.putting_green.python._surface_data import ContourPoint


class SurfaceIOMixin:
    """Mixin providing file-loading methods for GreenSurface."""

    def load_from_file(self, filepath: str | Path) -> None:
        """Load topographical data from file.

        Supports:
            - .npy: NumPy array
            - .csv: CSV with x,y,elevation columns
            - .json: JSON configuration
            - .tif/.tiff: GeoTIFF (requires rasterio)

        Args:
            filepath: Path to data file
        """
        if filepath is None:
            raise ValueError("filepath must be provided")
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix == ".npy":
            heightmap = np.load(filepath, allow_pickle=False)
            self.set_heightmap(heightmap)  # type: ignore[attr-defined]

        elif suffix == ".csv":
            self._load_csv_topography(filepath)

        elif suffix == ".json":
            self._load_json_topography(filepath)

        elif suffix in (".tif", ".tiff"):
            self._load_geotiff_topography(filepath)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_csv_topography(self, filepath: Path) -> None:
        """Load topography from CSV file."""
        if filepath is None:
            raise ValueError("filepath must be provided")
        import csv

        points = []
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append(
                    ContourPoint(
                        x=float(row.get("x", row.get("X", 0))),
                        y=float(row.get("y", row.get("Y", 0))),
                        elevation=float(
                            row.get("elevation", row.get("z", row.get("Z", 0)))
                        ),
                    )
                )

        self.set_contour_points(points)  # type: ignore[attr-defined]

    def _load_json_topography(self, filepath: Path) -> None:
        """Load topography from JSON file."""
        if filepath is None:
            raise ValueError("filepath must be provided")
        import json

        with open(filepath) as f:
            data = json.load(f)

        # Load contour points if present
        if "contours" in data:
            points = [
                ContourPoint(x=p["x"], y=p["y"], elevation=p["elevation"])
                for p in data["contours"]
            ]
            self.set_contour_points(points)  # type: ignore[attr-defined]

        # Load slope regions if present
        if "slopes" in data:
            from src.engines.physics_engines.putting_green.python._surface_data import (
                SlopeRegion,
            )

            for s in data["slopes"]:
                self.add_slope_region(  # type: ignore[attr-defined]
                    SlopeRegion(
                        center=np.array(s["center"]),
                        radius=s["radius"],
                        slope_direction=np.array(s["direction"]),
                        slope_magnitude=s["magnitude"],
                    )
                )

        # Load hole position if present
        if "hole_position" in data:
            self.set_hole_position(np.array(data["hole_position"]))  # type: ignore[attr-defined]

    def _load_geotiff_topography(self, filepath: Path) -> None:
        """Load topography from GeoTIFF file."""
        if filepath is None:
            raise ValueError("filepath must be provided")
        try:
            import rasterio  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "rasterio required for GeoTIFF support. Install with: pip install rasterio"
            ) from err

        with rasterio.open(filepath) as src:
            heightmap = src.read(1)
            self.set_heightmap(heightmap)  # type: ignore[attr-defined]
