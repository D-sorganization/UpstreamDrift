"""Swing Plane Visualization Module.

Guideline L: Visualization - Swing Plane Visualization (L1).

Provides 3D visualization of golf swing planes including:
- Instantaneous swing plane (real-time)
- Functional Swing Plane (FSP) from post-simulation analysis
- Plane mesh generation for 3D rendering
- Integration with reference frame transformations

Works with both meshcat for WebGL rendering and export to standard formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.io_utils import ensure_directory
from src.shared.python.logging_config import get_logger
from src.shared.python.reference_frames import (
    SwingPlaneFrame,
    fit_functional_swing_plane,
    fit_instantaneous_swing_plane,
)
from src.shared.python.swing_plane_analysis import SwingPlaneAnalyzer, SwingPlaneMetrics

if TYPE_CHECKING:
    pass

LOGGER = get_logger(__name__)

# Visualization defaults
DEFAULT_PLANE_SIZE = 1.5  # [m] half-width of visualized plane
DEFAULT_PLANE_COLOR = 0x4488FF  # Blue with transparency
DEFAULT_FSP_COLOR = 0xFF8844  # Orange for FSP
DEFAULT_TRAJECTORY_COLOR = 0x00FF00  # Green for trajectory


@dataclass
class SwingPlaneVisualization:
    """Visualization data for a swing plane.

    Attributes:
        plane_frame: The swing plane frame definition
        vertices: Plane mesh vertices (4, 3) for corners [m]
        normal_arrow_start: Start point of normal arrow [m]
        normal_arrow_end: End point of normal arrow [m]
        color: Visualization color (hex)
        label: Display label for plane
        is_fsp: True if this is a Functional Swing Plane
    """

    plane_frame: SwingPlaneFrame | None
    vertices: np.ndarray
    normal_arrow_start: np.ndarray
    normal_arrow_end: np.ndarray
    color: int
    label: str = "Swing Plane"
    is_fsp: bool = False


@dataclass
class TrajectoryVisualization:
    """Visualization data for clubhead trajectory.

    Attributes:
        points: Trajectory points [m] (N, 3)
        timestamps: Time at each point [s] (N,)
        color: Visualization color (hex)
        marker_size: Point marker size [m]
    """

    points: np.ndarray
    timestamps: np.ndarray
    color: int = DEFAULT_TRAJECTORY_COLOR
    marker_size: float = 0.01


@dataclass
class SwingPlaneScene:
    """Complete scene for swing plane visualization.

    Attributes:
        instantaneous_plane: Current instantaneous swing plane
        fsp: Functional Swing Plane (if computed)
        trajectory: Clubhead trajectory
        plane_metrics: Metrics from swing plane analysis
        deviation_heatmap: Optional deviation coloring for trajectory
    """

    instantaneous_plane: SwingPlaneVisualization | None = None
    fsp: SwingPlaneVisualization | None = None
    trajectory: TrajectoryVisualization | None = None
    plane_metrics: SwingPlaneMetrics | None = None
    deviation_heatmap: np.ndarray | None = None


def generate_plane_vertices(
    origin: np.ndarray,
    normal: np.ndarray,
    in_plane_x: np.ndarray,
    in_plane_y: np.ndarray,
    half_size: float = DEFAULT_PLANE_SIZE,
) -> np.ndarray:
    """Generate corner vertices for a rectangular plane.

    Args:
        origin: Center of the plane [m] (3,)
        normal: Plane normal vector [unitless] (3,)
        in_plane_x: In-plane X axis [unitless] (3,)
        in_plane_y: In-plane Y axis [unitless] (3,)
        half_size: Half-width of plane [m]

    Returns:
        vertices: Corner positions (4, 3) in CCW order
    """
    # Four corners in CCW order
    corners = np.array(
        [
            origin + half_size * in_plane_x + half_size * in_plane_y,
            origin - half_size * in_plane_x + half_size * in_plane_y,
            origin - half_size * in_plane_x - half_size * in_plane_y,
            origin + half_size * in_plane_x - half_size * in_plane_y,
        ]
    )
    return corners


def create_instantaneous_plane_visualization(
    clubhead_velocity: np.ndarray,
    grip_position: np.ndarray,
    clubhead_position: np.ndarray,
    half_size: float = DEFAULT_PLANE_SIZE,
    color: int = DEFAULT_PLANE_COLOR,
) -> SwingPlaneVisualization:
    """Create visualization for instantaneous swing plane.

    Args:
        clubhead_velocity: Clubhead velocity [m/s] (3,)
        grip_position: Grip center position [m] (3,)
        clubhead_position: Clubhead position [m] (3,)
        half_size: Half-width of visualized plane [m]
        color: Visualization color (hex)

    Returns:
        SwingPlaneVisualization for the current instant
    """
    # Fit the plane
    plane_frame = fit_instantaneous_swing_plane(
        clubhead_velocity, grip_position, clubhead_position
    )

    # Generate plane vertices
    vertices = generate_plane_vertices(
        origin=plane_frame.origin,
        normal=plane_frame.normal,
        in_plane_x=plane_frame.in_plane_x,
        in_plane_y=plane_frame.in_plane_y,
        half_size=half_size,
    )

    # Normal arrow (at plane center, pointing outward)
    arrow_length = half_size * 0.3  # 30% of plane size
    normal_start = plane_frame.origin
    normal_end = plane_frame.origin + arrow_length * plane_frame.normal

    return SwingPlaneVisualization(
        plane_frame=plane_frame,
        vertices=vertices,
        normal_arrow_start=normal_start,
        normal_arrow_end=normal_end,
        color=color,
        label="Instantaneous Plane",
        is_fsp=False,
    )


def create_fsp_visualization(
    clubhead_trajectory: np.ndarray,
    timestamps: np.ndarray,
    impact_time: float,
    window_ms: float = 100.0,
    half_size: float = DEFAULT_PLANE_SIZE,
    color: int = DEFAULT_FSP_COLOR,
) -> SwingPlaneVisualization:
    """Create visualization for Functional Swing Plane (FSP).

    Args:
        clubhead_trajectory: Clubhead positions over time [m] (N, 3)
        timestamps: Time values [s] (N,)
        impact_time: Time of impact [s]
        window_ms: Fitting window around impact [ms]
        half_size: Half-width of visualized plane [m]
        color: Visualization color (hex)

    Returns:
        SwingPlaneVisualization for the FSP
    """
    # Fit the FSP
    fsp_frame = fit_functional_swing_plane(
        clubhead_trajectory, timestamps, impact_time, window_ms
    )

    # Generate plane vertices
    vertices = generate_plane_vertices(
        origin=fsp_frame.origin,
        normal=fsp_frame.normal,
        in_plane_x=fsp_frame.in_plane_x,
        in_plane_y=fsp_frame.in_plane_y,
        half_size=half_size,
    )

    # Normal arrow
    arrow_length = half_size * 0.3
    normal_start = fsp_frame.origin
    normal_end = fsp_frame.origin + arrow_length * fsp_frame.normal

    return SwingPlaneVisualization(
        plane_frame=fsp_frame,
        vertices=vertices,
        normal_arrow_start=normal_start,
        normal_arrow_end=normal_end,
        color=color,
        label=f"FSP (RMSE: {fsp_frame.fitting_rmse:.3f}m)",
        is_fsp=True,
    )


def compute_trajectory_deviations(
    trajectory: np.ndarray,
    plane_frame: SwingPlaneFrame,
) -> np.ndarray:
    """Compute deviation of each trajectory point from the swing plane.

    Args:
        trajectory: Clubhead positions [m] (N, 3)
        plane_frame: Swing plane frame

    Returns:
        deviations: Signed distance from plane [m] (N,)
    """
    # Vector from plane origin to each point
    offsets = trajectory - plane_frame.origin

    # Deviation = dot product with normal
    deviations = np.dot(offsets, plane_frame.normal)

    return np.asarray(deviations)


def create_deviation_colormap(
    deviations: np.ndarray,
    max_deviation: float = 0.1,
) -> np.ndarray:
    """Create RGB colors based on deviation from plane.

    Negative deviation (below plane) → Blue
    Zero deviation (on plane) → Green
    Positive deviation (above plane) → Red

    Args:
        deviations: Deviation values [m] (N,)
        max_deviation: Max deviation for color scaling [m]

    Returns:
        colors: RGB colors (N, 3) normalized 0-1
    """
    # Normalize deviations to [-1, 1]
    normalized = np.clip(deviations / max_deviation, -1, 1)

    n = len(deviations)
    colors = np.zeros((n, 3))

    # Negative → Blue, Zero → Green, Positive → Red (vectorized)
    # Use boolean indexing for fast array operations
    negative_mask = normalized < 0

    # Blue to green for negative values
    colors[negative_mask, 1] = 1 + normalized[negative_mask]  # Green increases
    colors[negative_mask, 2] = -normalized[negative_mask]  # Blue decreases

    # Green to red for positive values
    colors[~negative_mask, 0] = normalized[~negative_mask]  # Red increases
    colors[~negative_mask, 1] = 1 - normalized[~negative_mask]  # Green decreases

    return colors


class SwingPlaneVisualizer:
    """High-level interface for swing plane visualization.

    Provides methods for creating and updating swing plane visualizations
    throughout a golf swing simulation.
    """

    def __init__(self) -> None:
        """Initialize the swing plane visualizer."""
        self.analyzer = SwingPlaneAnalyzer()
        self.current_scene = SwingPlaneScene()
        self.trajectory_history: list[np.ndarray] = []
        self.timestamp_history: list[float] = []

    def update_instantaneous_plane(
        self,
        clubhead_velocity: np.ndarray,
        grip_position: np.ndarray,
        clubhead_position: np.ndarray,
    ) -> SwingPlaneVisualization:
        """Update the instantaneous swing plane visualization.

        Args:
            clubhead_velocity: Current clubhead velocity [m/s] (3,)
            grip_position: Current grip position [m] (3,)
            clubhead_position: Current clubhead position [m] (3,)

        Returns:
            Updated SwingPlaneVisualization
        """
        vis = create_instantaneous_plane_visualization(
            clubhead_velocity, grip_position, clubhead_position
        )
        self.current_scene.instantaneous_plane = vis
        return vis

    def record_trajectory_point(self, position: np.ndarray, timestamp: float) -> None:
        """Record a trajectory point for FSP computation.

        Args:
            position: Clubhead position [m] (3,)
            timestamp: Current time [s]
        """
        self.trajectory_history.append(position.copy())
        self.timestamp_history.append(timestamp)

    def compute_fsp(
        self,
        impact_time: float,
        window_ms: float = 100.0,
    ) -> SwingPlaneVisualization | None:
        """Compute and visualize the Functional Swing Plane.

        Args:
            impact_time: Estimated time of impact [s]
            window_ms: Fitting window around impact [ms]

        Returns:
            FSP visualization, or None if insufficient data
        """
        if len(self.trajectory_history) < 3:
            LOGGER.warning("Insufficient trajectory data for FSP computation")
            return None

        trajectory = np.array(self.trajectory_history)
        timestamps = np.array(self.timestamp_history)

        fsp_vis = create_fsp_visualization(
            trajectory, timestamps, impact_time, window_ms
        )
        self.current_scene.fsp = fsp_vis

        # Also compute metrics using the analyzer
        metrics = self.analyzer.analyze(trajectory)
        self.current_scene.plane_metrics = metrics

        return fsp_vis

    def get_trajectory_visualization(self) -> TrajectoryVisualization | None:
        """Get trajectory visualization with deviation coloring.

        Returns:
            TrajectoryVisualization, or None if no trajectory
        """
        if not self.trajectory_history:
            return None

        trajectory = np.array(self.trajectory_history)
        timestamps = np.array(self.timestamp_history)

        vis = TrajectoryVisualization(
            points=trajectory,
            timestamps=timestamps,
        )
        self.current_scene.trajectory = vis

        # If FSP exists, compute deviation heatmap
        if self.current_scene.fsp and self.current_scene.fsp.plane_frame:
            deviations = compute_trajectory_deviations(
                trajectory, self.current_scene.fsp.plane_frame
            )
            self.current_scene.deviation_heatmap = create_deviation_colormap(deviations)

        return vis

    def export_scene_json(self, output_path: Path | str) -> None:
        """Export current scene to JSON for external visualization.

        Args:
            output_path: Path to output JSON file
        """
        from typing import Any

        output_path = Path(output_path)
        ensure_directory(output_path.parent)

        scene_data: dict[str, Any] = {
            "instantaneous_plane": None,
            "fsp": None,
            "trajectory": None,
            "metrics": None,
        }

        if self.current_scene.instantaneous_plane:
            plane = self.current_scene.instantaneous_plane
            scene_data["instantaneous_plane"] = {
                "vertices": plane.vertices.tolist(),
                "normal_start": plane.normal_arrow_start.tolist(),
                "normal_end": plane.normal_arrow_end.tolist(),
                "color": plane.color,
                "label": plane.label,
            }

        if self.current_scene.fsp:
            fsp = self.current_scene.fsp
            scene_data["fsp"] = {
                "vertices": fsp.vertices.tolist(),
                "normal_start": fsp.normal_arrow_start.tolist(),
                "normal_end": fsp.normal_arrow_end.tolist(),
                "color": fsp.color,
                "label": fsp.label,
                "rmse": (fsp.plane_frame.fitting_rmse if fsp.plane_frame else None),
            }

        if self.current_scene.trajectory:
            traj = self.current_scene.trajectory
            traj_data: dict[str, Any] = {
                "points": traj.points.tolist(),
                "timestamps": traj.timestamps.tolist(),
            }
            if self.current_scene.deviation_heatmap is not None:
                traj_data["colors"] = self.current_scene.deviation_heatmap.tolist()
            scene_data["trajectory"] = traj_data

        if self.current_scene.plane_metrics:
            m = self.current_scene.plane_metrics
            scene_data["metrics"] = {
                "steepness_deg": m.steepness_deg,
                "direction_deg": m.direction_deg,
                "rmse": m.rmse,
                "max_deviation": m.max_deviation,
            }

        with open(output_path, "w") as f:
            json.dump(scene_data, f, indent=2)

        LOGGER.info(f"Exported swing plane scene to {output_path}")

    def reset(self) -> None:
        """Reset the visualizer state."""
        self.trajectory_history.clear()
        self.timestamp_history.clear()
        self.current_scene = SwingPlaneScene()
