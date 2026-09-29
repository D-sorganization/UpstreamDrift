from __future__ import annotations

import math
from typing import Any

import numpy as np


class SurfaceAnalysisMixin:
    """Mixin providing putt analysis methods for GreenSurface."""

    def calculate_break(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_samples: int = 20,
    ) -> dict[str, Any]:
        """Calculate break (lateral deviation) for a putt line.

        Args:
            start: Starting position [m, m]
            end: Target position [m, m]
            num_samples: Number of sample points along line

        Returns:
            Dictionary with break analysis
        """
        # Sample points along intended line
        if start is None:
            raise ValueError("start must be provided")
        t_values = np.linspace(0, 1, num_samples)
        positions = [start + t * (end - start) for t in t_values]

        slopes = [self.get_slope_at(p) for p in positions]  # type: ignore[attr-defined]

        # Perpendicular to putt direction
        putt_dir = end - start
        putt_len = math.hypot(*putt_dir)
        if putt_len < 1e-10:
            return {
                "total_break": 0.0,
                "break_direction": np.zeros(2),
                "average_slope": np.zeros(2),
            }

        putt_dir_norm = putt_dir / putt_len
        perp_dir = np.array([-putt_dir_norm[1], putt_dir_norm[0]])

        # Integrate cross-slope component
        total_break = 0.0
        for slope in slopes:
            cross_slope = np.dot(slope, perp_dir)
            total_break += cross_slope * (putt_len / num_samples)

        # Convert to actual break distance (approximation)
        # Break ≈ cross_slope * distance^2 / (4 * initial_velocity^2) * g
        avg_slope = np.mean(slopes, axis=0)
        break_magnitude = abs(total_break) * putt_len * 0.25

        break_direction = perp_dir * np.sign(total_break)

        return {
            "total_break": break_magnitude,
            "break_direction": break_direction,
            "average_slope": avg_slope,
            "cross_slopes": [np.dot(s, perp_dir) for s in slopes],
        }

    def read_putt_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_samples: int = 20,
    ) -> dict[str, Any]:
        """Read the putt line for elevations and slopes.

        Args:
            start: Starting position [m, m]
            end: Target position [m, m]
            num_samples: Number of sample points

        Returns:
            Dictionary with positions, elevations, and slopes along line
        """
        if start is None:
            raise ValueError("start must be provided")
        t_values = np.linspace(0, 1, num_samples)
        positions = [start + t * (end - start) for t in t_values]

        diff_arr = np.asarray(end - start, dtype=float).reshape(-1)
        return {
            "positions": np.array(positions),
            "elevations": np.array([self.get_elevation_at(p) for p in positions]),  # type: ignore[attr-defined]
            "slopes": np.array([self.get_slope_at(p) for p in positions]),  # type: ignore[attr-defined]
            "distance": 0.0 if diff_arr.size == 0 else math.hypot(*diff_arr),
        }

    def to_heightmap(self, resolution: int = 100) -> np.ndarray:
        """Export surface as heightmap array.

        Args:
            resolution: Number of points in each dimension

        Returns:
            2D array of elevations [resolution x resolution]
        """
        if resolution is None:
            raise ValueError("resolution must be provided")
        x = np.linspace(0, self.width, resolution)  # type: ignore[attr-defined]
        y = np.linspace(0, self.height, resolution)  # type: ignore[attr-defined]

        heightmap = np.zeros((resolution, resolution))

        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                heightmap[i, j] = self.get_elevation_at(np.array([xi, yi]))  # type: ignore[attr-defined]

        return heightmap
