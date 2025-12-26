"""Swing Plane Analysis Integration for Pinocchio Engine.

This module integrates the shared SwingPlaneAnalyzer with Pinocchio-specific
golf swing simulations, providing consistent swing plane analysis across engines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shared.python.swing_plane_analysis import SwingPlaneMetrics

try:
    from shared.python.core import setup_logging
    from shared.python.swing_plane_analysis import SwingPlaneAnalyzer
except ImportError as e:
    raise ImportError(
        "Failed to import shared modules. Ensure shared.python is in PYTHONPATH."
    ) from e

logger = setup_logging(__name__)


class PinocchioSwingPlaneAnalyzer:
    """Pinocchio-specific swing plane analysis integration."""

    def __init__(self):
        """Initialize the Pinocchio swing plane analyzer."""
        self.analyzer = SwingPlaneAnalyzer()
        self.logger = logger

    def analyze_trajectory(
        self, positions: np.ndarray, timestamps: np.ndarray | None = None
    ) -> SwingPlaneMetrics:
        """Analyze swing plane from Pinocchio trajectory data.

        Args:
            positions: Club head positions (N, 3) in world coordinates
            timestamps: Optional timestamps for each position

        Returns:
            SwingPlaneMetrics with plane analysis results
        """
        if positions.shape[1] != 3:
            raise ValueError("Positions must be (N, 3) array")

        if len(positions) < 3:
            raise ValueError("At least 3 positions required for plane analysis")

        self.logger.info(
            f"Analyzing swing plane from {len(positions)} trajectory points"
        )

        # Use shared analyzer
        metrics = self.analyzer.analyze(positions)

        self.logger.info(
            f"Swing plane analysis complete: "
            f"steepness={metrics.steepness_deg:.1f}°, "
            f"RMSE={metrics.rmse:.4f}"
        )

        return metrics

    def analyze_double_pendulum_swing(
        self,
        joint_angles: np.ndarray,
        link_lengths: tuple[float, float],
        plane_inclination_deg: float = 35.0,
    ) -> SwingPlaneMetrics:
        """Analyze swing plane for double pendulum model.

        Args:
            joint_angles: Joint angles (N, 2) - [shoulder, wrist]
            link_lengths: (upper_arm_length, forearm_length) in meters
            plane_inclination_deg: Inclination of swing plane from vertical

        Returns:
            SwingPlaneMetrics for the pendulum swing
        """
        # Convert joint angles to 3D club head positions
        positions = self._compute_club_head_positions(
            joint_angles, link_lengths, plane_inclination_deg
        )

        return self.analyze_trajectory(positions)

    def _compute_club_head_positions(
        self,
        joint_angles: np.ndarray,
        link_lengths: tuple[float, float],
        plane_inclination_deg: float,
    ) -> np.ndarray:
        """Convert joint angles to 3D club head positions.

        Args:
            joint_angles: Joint angles (N, 2) - [shoulder, wrist]
            link_lengths: (upper_arm_length, forearm_length)
            plane_inclination_deg: Plane inclination from vertical

        Returns:
            Club head positions (N, 3) in world coordinates
        """
        l1, l2 = link_lengths
        theta1, theta2 = joint_angles[:, 0], joint_angles[:, 1]

        # Convert plane inclination to radians
        plane_incline_rad = np.radians(plane_inclination_deg)

        # Compute positions in the swing plane coordinate system
        # Shoulder position (origin of upper arm) - not used in calculations
        # but kept for clarity of the coordinate system

        # Elbow position
        x_elbow = l1 * np.sin(theta1)
        y_elbow = -l1 * np.cos(theta1)

        # Club head position (end of forearm)
        x_club = x_elbow + l2 * np.sin(theta1 + theta2)
        y_club = y_elbow - l2 * np.cos(theta1 + theta2)

        # Transform from swing plane to world coordinates
        # Rotate around x-axis by plane inclination
        cos_incline = np.cos(plane_incline_rad)
        sin_incline = np.sin(plane_incline_rad)

        # In swing plane: (x_club, y_club, 0)
        # After rotation: (x, y*cos - z*sin, y*sin + z*cos)
        positions = np.column_stack(
            [
                x_club,  # x unchanged
                y_club * cos_incline,  # y rotated
                y_club * sin_incline,  # z from rotation
            ]
        )

        return positions

    def get_plane_visualization_data(
        self, metrics: SwingPlaneMetrics, extent: float = 2.0
    ) -> dict[str, np.ndarray]:
        """Get data for visualizing the swing plane.

        Args:
            metrics: Swing plane metrics from analysis
            extent: Size of plane visualization (meters)

        Returns:
            Dictionary with plane mesh data for visualization
        """
        # Create a mesh grid for the plane
        u = np.linspace(-extent, extent, 20)
        v = np.linspace(-extent, extent, 20)
        U, V = np.meshgrid(u, v)

        # Plane equation: normal · (point - point_on_plane) = 0
        # Solve for the third coordinate
        normal = metrics.normal_vector
        point = metrics.point_on_plane

        # Choose two orthogonal vectors in the plane
        if abs(normal[2]) > 0.1:
            # Normal not parallel to z-axis
            v1 = np.array([1, 0, -normal[0] / normal[2]])
        else:
            # Normal parallel to z-axis, use different approach
            v1 = np.array([0, 1, -normal[1] / normal[2]])

        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)

        # Generate plane points
        plane_points = (
            point[np.newaxis, np.newaxis, :]
            + U[:, :, np.newaxis] * v1[np.newaxis, np.newaxis, :]
            + V[:, :, np.newaxis] * v2[np.newaxis, np.newaxis, :]
        )

        return {
            "x": plane_points[:, :, 0],
            "y": plane_points[:, :, 1],
            "z": plane_points[:, :, 2],
            "normal": normal,
            "center": point,
        }
