"""Swing plane analysis module.

Provides analysis of the golf swing plane, including:
- Fitting a plane to the club head trajectory.
- Calculating deviation from the plane.
- Computing plane orientation (steepness/inclination, direction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import ensure, require


@dataclass
class SwingPlaneMetrics:
    """Metrics related to the swing plane."""

    normal_vector: np.ndarray  # Normal vector of the plane (3,)
    point_on_plane: np.ndarray  # A point on the plane (centroid) (3,)
    steepness_deg: float  # Angle with horizontal (degrees)
    direction_deg: float  # Azimuth direction (degrees)
    rmse: float  # Root Mean Square Error of fit (on-plane score)
    max_deviation: float  # Maximum distance from plane


class SwingPlaneAnalyzer:
    """Analyzes the swing plane from 3D trajectory data."""

    def fit_plane(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit a plane to a set of 3D points using SVD.

        Design by Contract:
            Preconditions:
                - len(points) >= 3
            Postconditions:
                - normal vector is unit length (within tolerance)

        Args:
            points: Array of points (N, 3)

        Returns:
            Tuple of (centroid, normal)

        Raises:
            ValueError: If fewer than 3 points are provided.
        """
        require(len(points) >= 3, "at least 3 points required to fit a plane")

        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        # SVD
        # The normal vector is the last row of Vh (corresponding to smallest singular value)
        # OPTIMIZATION: Use full_matrices=False to avoid computing NxN U matrix (huge for large N)
        _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
        normal = vh[2, :]

        # Normalize (just in case)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm

        ensure(
            abs(np.linalg.norm(normal) - 1.0) < 1e-6,
            "normal vector must be unit length",
        )

        return centroid, normal

    def calculate_deviation(
        self,
        points: np.ndarray,
        centroid: np.ndarray,
        normal: np.ndarray,
    ) -> np.ndarray:
        """Calculate signed distance of points from the plane.

        Args:
            points: (N, 3)
            centroid: (3,)
            normal: (3,)

        Returns:
            deviations: (N,) signed distances
        """
        result = np.dot(points - centroid, normal)
        return np.asarray(result)

    def analyze(self, points: np.ndarray) -> SwingPlaneMetrics:
        """Perform full swing plane analysis on trajectory.

        Design by Contract:
            Postconditions:
                - rmse >= 0
                - max_deviation >= 0
                - steepness_deg in [0, 180]

        Args:
            points: (N, 3) club head trajectory (or similar)

        Returns:
            SwingPlaneMetrics object
        """
        centroid, normal = self.fit_plane(points)
        deviations = self.calculate_deviation(points, centroid, normal)

        rmse = np.sqrt(np.mean(deviations**2))
        max_dev = np.max(np.abs(deviations))

        # Steepness: Angle between normal and vertical (Z axis)
        # Ensure normal z is positive to measure from top
        if normal[2] < 0:
            normal = -normal

        nz = normal[2]
        angle_rad = np.arccos(np.clip(nz, -1.0, 1.0))
        steepness = np.degrees(angle_rad)

        # Direction (Azimuth): Direction the plane is facing (projected on XY)
        nx, ny = normal[0], normal[1]
        direction = np.degrees(np.arctan2(ny, nx))

        result = SwingPlaneMetrics(
            normal_vector=normal,
            point_on_plane=centroid,
            steepness_deg=float(steepness),
            direction_deg=float(direction),
            rmse=float(rmse),
            max_deviation=float(max_dev),
        )

        # Postconditions
        ensure(result.rmse >= 0, "RMSE must be non-negative", result.rmse)
        ensure(
            result.max_deviation >= 0,
            "max deviation must be non-negative",
            result.max_deviation,
        )
        ensure(
            0.0 <= result.steepness_deg <= 180.0,
            "steepness must be in [0, 180] degrees",
            result.steepness_deg,
        )

        return result
