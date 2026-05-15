from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._topography_types import ElevationPoint, TopographyBounds


class _TopographyIOMixin:
    _bounds: TopographyBounds
    _heightmap: np.ndarray | None

    def set_heightmap(
        self, heightmap: np.ndarray, smooth: bool = True, smooth_sigma: float = 1.0
    ) -> None: ...

    def set_contour_points(self, points: list[ElevationPoint]) -> None: ...

    def to_heightmap(self, resolution: int = 100) -> np.ndarray: ...  # type: ignore[empty-body]

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        width: float | None = None,
        height: float | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> _TopographyIOMixin:
        """Load topography from file.

        Args:
            filepath: Path to data file
            width: Physical width [m] (auto-detected if None)
            height: Physical height [m] (auto-detected if None)
            origin: Origin point (min_x, min_y)

        Returns:
            TopographyData instance
        """
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        topo = cls()
        topo._bounds.min_x = origin[0]
        topo._bounds.min_y = origin[1]

        if suffix == ".npy":
            topo._load_numpy(filepath, width, height)
        elif suffix == ".csv":
            topo._load_csv(filepath, width, height)
        elif suffix == ".json":
            topo._load_json(filepath, width, height)
        elif suffix in (".tif", ".tiff"):
            topo._load_geotiff(filepath, width, height)
        elif suffix in (".png", ".jpg", ".jpeg"):
            topo._load_image(filepath, width, height)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return topo

    def _load_numpy(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from NumPy file."""
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        heightmap = np.load(filepath, allow_pickle=False)

        if width is not None:
            self._bounds.max_x = self._bounds.min_x + width
        else:
            self._bounds.max_x = self._bounds.min_x + heightmap.shape[1]

        if height is not None:
            self._bounds.max_y = self._bounds.min_y + height
        else:
            self._bounds.max_y = self._bounds.min_y + heightmap.shape[0]

        self.set_heightmap(heightmap, smooth=False)

    def _load_csv(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from CSV file with x, y, elevation columns."""
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        import csv

        points = []
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row.get("x", row.get("X", row.get("easting", 0))))
                y = float(row.get("y", row.get("Y", row.get("northing", 0))))
                z = float(
                    row.get(
                        "elevation", row.get("z", row.get("Z", row.get("height", 0)))
                    )
                )
                points.append(ElevationPoint(x=x, y=y, z=z))

        self.set_contour_points(points)

        if width is not None:
            self._bounds.max_x = self._bounds.min_x + width
        if height is not None:
            self._bounds.max_y = self._bounds.min_y + height

    def _load_json(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from JSON file."""
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        with open(filepath) as f:
            data = json.load(f)

        if "contours" in data:
            points = [
                ElevationPoint(x=p["x"], y=p["y"], z=p.get("z", p.get("elevation", 0)))
                for p in data["contours"]
            ]
            self.set_contour_points(points)
        elif "heightmap" in data:
            heightmap = np.array(data["heightmap"])
            w = data.get("width", width or heightmap.shape[1])
            h = data.get("height", height or heightmap.shape[0])
            self._bounds.max_x = self._bounds.min_x + w
            self._bounds.max_y = self._bounds.min_y + h
            self.set_heightmap(heightmap)

        if width is not None:
            self._bounds.max_x = self._bounds.min_x + width
        if height is not None:
            self._bounds.max_y = self._bounds.min_y + height

    def _load_geotiff(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from GeoTIFF file."""
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        try:
            import rasterio  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "rasterio required for GeoTIFF support. Install with: pip install rasterio"
            ) from err

        with rasterio.open(filepath) as src:
            heightmap = src.read(1)

            if width is None:
                self._bounds.min_x = src.bounds.left
                self._bounds.max_x = src.bounds.right
            else:
                self._bounds.max_x = self._bounds.min_x + width

            if height is None:
                self._bounds.min_y = src.bounds.bottom
                self._bounds.max_y = src.bounds.top
            else:
                self._bounds.max_y = self._bounds.min_y + height

        self.set_heightmap(heightmap, smooth=False)

    def _load_image(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from image file (grayscale as elevation)."""
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            import matplotlib.pyplot as plt

            img = plt.imread(str(filepath))
            heightmap = np.mean(img, axis=2) if len(img.shape) == 3 else img
        else:
            pil_img = Image.open(filepath).convert("L")
            heightmap = np.array(pil_img) / 255.0

        if width is not None:
            self._bounds.max_x = self._bounds.min_x + width
        else:
            self._bounds.max_x = self._bounds.min_x + heightmap.shape[1]

        if height is not None:
            self._bounds.max_y = self._bounds.min_y + height
        else:
            self._bounds.max_y = self._bounds.min_y + heightmap.shape[0]

        self.set_heightmap(heightmap)

    def save_to_file(self, filepath: str | Path, format: str | None = None) -> None:
        """Save topography to file.

        Args:
            filepath: Output file path
            format: Output format ("npy", "csv", "json") - auto-detected from suffix if None
        """
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        filepath = Path(filepath)
        fmt = format or filepath.suffix.lower().lstrip(".")

        if fmt == "npy":
            if self._heightmap is not None:
                np.save(filepath, self._heightmap)
            else:
                np.save(filepath, self.to_heightmap())
        elif fmt == "csv":
            self._save_csv(filepath)
        elif fmt == "json":
            self._save_json(filepath)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    def _save_csv(self, filepath: Path) -> None:
        """Save to CSV file."""
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        import csv

        heightmap = (
            self._heightmap if self._heightmap is not None else self.to_heightmap()
        )
        ny, nx = heightmap.shape

        x_coords = np.linspace(self._bounds.min_x, self._bounds.max_x, nx)
        y_coords = np.linspace(self._bounds.min_y, self._bounds.max_y, ny)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "elevation"])
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    writer.writerow([x, y, heightmap[i, j]])

    def _save_json(self, filepath: Path) -> None:
        """Save to JSON file."""
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        heightmap = (
            self._heightmap if self._heightmap is not None else self.to_heightmap()
        )

        data = {
            "bounds": {
                "min_x": self._bounds.min_x,
                "max_x": self._bounds.max_x,
                "min_y": self._bounds.min_y,
                "max_y": self._bounds.max_y,
                "min_z": self._bounds.min_z,
                "max_z": self._bounds.max_z,
            },
            "heightmap": heightmap.tolist(),
            "width": self._bounds.width,
            "height": self._bounds.height,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
