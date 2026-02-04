"""Visualization components for Unreal Engine integration.

This module provides renderers and data providers for visualizing
physics simulation data in Unreal Engine.

Components:
    - ForceVectorRenderer: Renders force/torque arrows
    - TrajectoryRenderer: Renders motion paths and ball flights
    - HUDDataProvider: Provides real-time metrics for UI overlays

Design by Contract:
    - Renderers validate input data before processing
    - All renderers support configurable visual properties
    - Output data is optimized for real-time rendering

Usage:
    from src.unreal_integration.visualization import (
        ForceVectorRenderer,
        TrajectoryRenderer,
        HUDDataProvider,
    )

    renderer = ForceVectorRenderer()
    render_data = renderer.render(forces)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from src.unreal_integration.data_models import (
    ForceVector,
    SwingMetrics,
    TrajectoryPoint,
    Vector3,
)

import logging
logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """Types of visualizations."""

    FORCE_ARROW = auto()
    TORQUE_RING = auto()
    TRAJECTORY_LINE = auto()
    TRAJECTORY_RIBBON = auto()
    SKELETON_OVERLAY = auto()
    HUD_PANEL = auto()


@dataclass
class VisualizationConfig:
    """Configuration for visualization rendering.

    Attributes:
        force_scale: Scale factor for force arrow length.
        torque_scale: Scale factor for torque indicators.
        trajectory_width: Width of trajectory lines.
        trajectory_segments: Number of segments in trajectory.
        force_color_map: Color mapping for force types.
        show_labels: Whether to show value labels.
        animation_speed: Animation speed multiplier.
    """

    force_scale: float = 0.1  # 10 cm per Newton
    torque_scale: float = 0.05  # 5 cm per Nm
    trajectory_width: float = 0.02  # 2 cm
    trajectory_segments: int = 100
    force_color_map: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "force": (0.0, 1.0, 0.0, 1.0),  # Green
            "torque": (1.0, 0.5, 0.0, 1.0),  # Orange
            "gravity": (0.5, 0.5, 0.5, 1.0),  # Gray
            "muscle": (1.0, 0.0, 0.0, 1.0),  # Red
            "ground_reaction": (0.0, 0.0, 1.0, 1.0),  # Blue
        }
    )
    show_labels: bool = True
    animation_speed: float = 1.0

    @classmethod
    def default(cls) -> VisualizationConfig:
        """Create default configuration."""
        return cls()

    @classmethod
    def for_vr(cls) -> VisualizationConfig:
        """Create VR-optimized configuration."""
        return cls(
            force_scale=0.15,  # Larger for VR visibility
            trajectory_width=0.03,
            show_labels=False,  # Labels hard to read in VR
        )


@dataclass
class RenderData:
    """Data ready for rendering.

    Attributes:
        visualization_type: Type of visualization.
        vertices: Vertex positions.
        colors: Vertex colors (RGBA).
        indices: Triangle indices (if applicable).
        metadata: Additional rendering metadata.
    """

    visualization_type: VisualizationType
    vertices: np.ndarray
    colors: np.ndarray | None = None
    indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.visualization_type.name.lower(),
            "vertices": self.vertices.tolist(),
            "colors": self.colors.tolist() if self.colors is not None else None,
            "indices": self.indices.tolist() if self.indices is not None else None,
            "metadata": self.metadata,
        }


class ForceVectorRenderer:
    """Renderer for force and torque vector visualization.

    Creates 3D arrow geometry for force vectors that can be
    rendered in Unreal Engine.

    Example:
        >>> renderer = ForceVectorRenderer()
        >>> forces = [ForceVector(...), ForceVector(...)]
        >>> render_data = renderer.render(forces)
    """

    def __init__(self, config: VisualizationConfig | None = None):
        """Initialize force vector renderer.

        Args:
            config: Visualization configuration.
        """
        self.config = config or VisualizationConfig.default()

    def render(self, forces: list[ForceVector]) -> list[RenderData]:
        """Render force vectors to arrow geometry.

        Args:
            forces: List of force vectors to render.

        Returns:
            List of RenderData for each force.
        """
        results: list[RenderData] = []

        for force in forces:
            if force.force_type == "torque":
                render_data = self._render_torque(force)
            else:
                render_data = self._render_arrow(force)
            results.append(render_data)

        return results

    def _render_arrow(self, force: ForceVector) -> RenderData:
        """Render force as arrow.

        Args:
            force: Force vector to render.

        Returns:
            RenderData with arrow geometry.
        """
        # Calculate arrow dimensions
        scale = self.config.force_scale * force.scale_factor
        arrow_length = force.magnitude * scale
        arrow_head_length = min(0.1, arrow_length * 0.3)
        shaft_radius = 0.01
        head_radius = 0.025

        # Get direction (normalize)
        direction = force.direction.to_numpy()
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([0, 0, 1])

        # Calculate endpoint
        origin = force.origin.to_numpy()
        endpoint = origin + direction * arrow_length

        # Create arrow geometry (simplified - line + cone)
        vertices = np.array([
            origin,
            endpoint - direction * arrow_head_length,  # Shaft end
            endpoint,  # Tip
        ])

        # Get color
        color = self.config.force_color_map.get(
            force.force_type,
            (0.0, 1.0, 0.0, 1.0)
        )
        if force.color is not None:
            color = force.color

        colors = np.tile(color, (len(vertices), 1))

        return RenderData(
            visualization_type=VisualizationType.FORCE_ARROW,
            vertices=vertices,
            colors=colors,
            metadata={
                "magnitude": force.magnitude,
                "force_type": force.force_type,
                "joint_name": force.joint_name,
                "shaft_radius": shaft_radius,
                "head_radius": head_radius,
            },
        )

    def _render_torque(self, force: ForceVector) -> RenderData:
        """Render torque as circular arc.

        Args:
            force: Torque vector to render.

        Returns:
            RenderData with arc geometry.
        """
        scale = self.config.torque_scale * force.scale_factor
        arc_radius = 0.1  # 10 cm base radius
        arc_segments = 16

        # Get axis (direction)
        axis = force.direction.to_numpy()
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
        else:
            axis = np.array([0, 0, 1])

        origin = force.origin.to_numpy()

        # Create perpendicular vectors for arc
        if abs(axis[2]) < 0.9:
            perp1 = np.cross(axis, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(axis, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)

        # Generate arc vertices
        arc_angle = min(2 * np.pi, force.magnitude * scale)
        vertices = []
        for i in range(arc_segments + 1):
            angle = arc_angle * i / arc_segments
            point = origin + arc_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(point)

        vertices = np.array(vertices)

        # Get color
        color = self.config.force_color_map.get("torque", (1.0, 0.5, 0.0, 1.0))
        if force.color is not None:
            color = force.color

        colors = np.tile(color, (len(vertices), 1))

        return RenderData(
            visualization_type=VisualizationType.TORQUE_RING,
            vertices=vertices,
            colors=colors,
            metadata={
                "magnitude": force.magnitude,
                "force_type": force.force_type,
                "joint_name": force.joint_name,
                "arc_radius": arc_radius,
            },
        )


class TrajectoryRenderer:
    """Renderer for motion trajectories and ball flights.

    Creates spline/ribbon geometry for trajectory visualization.

    Example:
        >>> renderer = TrajectoryRenderer()
        >>> points = [TrajectoryPoint(...), ...]
        >>> render_data = renderer.render(points)
    """

    def __init__(self, config: VisualizationConfig | None = None):
        """Initialize trajectory renderer.

        Args:
            config: Visualization configuration.
        """
        self.config = config or VisualizationConfig.default()

    def render(
        self,
        points: list[TrajectoryPoint],
        as_ribbon: bool = False,
    ) -> RenderData:
        """Render trajectory points to line/ribbon geometry.

        Args:
            points: List of trajectory points.
            as_ribbon: If True, render as ribbon instead of line.

        Returns:
            RenderData with trajectory geometry.
        """
        if not points:
            return RenderData(
                visualization_type=VisualizationType.TRAJECTORY_LINE,
                vertices=np.array([]),
            )

        # Extract positions
        vertices = np.array([p.position.to_numpy() for p in points])

        # Extract or generate colors
        colors = []
        for p in points:
            if p.color is not None:
                colors.append(p.color)
            else:
                # Default: gradient from blue to red based on velocity
                if p.velocity is not None:
                    speed = p.velocity.magnitude
                    t = min(1.0, speed / 50.0)  # Normalize to ~50 m/s
                    colors.append((t, 0.0, 1.0 - t, 1.0))
                else:
                    colors.append((0.0, 0.5, 1.0, 1.0))

        colors = np.array(colors)

        viz_type = VisualizationType.TRAJECTORY_RIBBON if as_ribbon else VisualizationType.TRAJECTORY_LINE

        return RenderData(
            visualization_type=viz_type,
            vertices=vertices,
            colors=colors,
            metadata={
                "point_count": len(points),
                "line_width": self.config.trajectory_width,
                "total_time": points[-1].time - points[0].time if len(points) > 1 else 0,
            },
        )

    def render_ball_flight(
        self,
        points: list[TrajectoryPoint],
        landing_marker: bool = True,
    ) -> list[RenderData]:
        """Render golf ball flight trajectory.

        Args:
            points: Trajectory points.
            landing_marker: Whether to add landing position marker.

        Returns:
            List of RenderData for trajectory and markers.
        """
        results: list[RenderData] = []

        # Main trajectory
        trajectory = self.render(points, as_ribbon=False)
        trajectory.metadata["trajectory_type"] = "ball_flight"
        results.append(trajectory)

        # Landing marker
        if landing_marker and points:
            landing_point = points[-1].position.to_numpy()
            marker_vertices = np.array([landing_point])
            results.append(RenderData(
                visualization_type=VisualizationType.TRAJECTORY_LINE,
                vertices=marker_vertices,
                colors=np.array([[1.0, 1.0, 0.0, 1.0]]),  # Yellow
                metadata={"marker_type": "landing", "radius": 0.2},
            ))

        return results


class HUDDataProvider:
    """Provider for HUD (Head-Up Display) data.

    Formats swing metrics and real-time data for UI overlays.

    Example:
        >>> provider = HUDDataProvider()
        >>> hud_data = provider.get_hud_data(metrics, frame_data)
    """

    def __init__(self, units: str = "metric"):
        """Initialize HUD data provider.

        Args:
            units: Unit system ("metric" or "imperial").
        """
        self.units = units
        self._conversion_factors = {
            "metric": {"speed": 1.0, "distance": 1.0, "angle": 1.0},
            "imperial": {"speed": 2.237, "distance": 1.094, "angle": 1.0},  # m/s to mph, m to yards
        }

    def get_hud_data(
        self,
        metrics: SwingMetrics | None = None,
        timestamp: float = 0.0,
        frame_number: int = 0,
    ) -> dict[str, Any]:
        """Get formatted HUD data.

        Args:
            metrics: Swing metrics.
            timestamp: Current timestamp.
            frame_number: Current frame number.

        Returns:
            Dictionary of HUD display data.
        """
        conv = self._conversion_factors[self.units]

        hud: dict[str, Any] = {
            "timestamp": timestamp,
            "frame": frame_number,
            "units": self.units,
            "panels": {},
        }

        if metrics:
            # Speed panel
            if metrics.club_head_speed is not None:
                hud["panels"]["club_head_speed"] = {
                    "label": "Club Head Speed",
                    "value": metrics.club_head_speed * conv["speed"],
                    "unit": "mph" if self.units == "imperial" else "m/s",
                    "format": "{:.1f}",
                }

            # Ball speed
            if metrics.estimated_ball_speed is not None:
                hud["panels"]["ball_speed"] = {
                    "label": "Ball Speed",
                    "value": metrics.estimated_ball_speed * conv["speed"],
                    "unit": "mph" if self.units == "imperial" else "m/s",
                    "format": "{:.1f}",
                }

            # Smash factor
            if metrics.smash_factor is not None:
                hud["panels"]["smash_factor"] = {
                    "label": "Smash Factor",
                    "value": metrics.smash_factor,
                    "unit": "",
                    "format": "{:.2f}",
                }

            # X-Factor
            if metrics.x_factor is not None:
                hud["panels"]["x_factor"] = {
                    "label": "X-Factor",
                    "value": metrics.x_factor,
                    "unit": "\u00b0",  # Degree symbol
                    "format": "{:.1f}",
                }

            # Attack angle
            if metrics.attack_angle is not None:
                hud["panels"]["attack_angle"] = {
                    "label": "Attack Angle",
                    "value": metrics.attack_angle,
                    "unit": "\u00b0",
                    "format": "{:+.1f}",
                }

            # Swing path
            if metrics.swing_path is not None:
                hud["panels"]["swing_path"] = {
                    "label": "Swing Path",
                    "value": metrics.swing_path,
                    "unit": "\u00b0",
                    "format": "{:+.1f}",
                }

            # Face to path
            if metrics.face_to_path is not None:
                hud["panels"]["face_to_path"] = {
                    "label": "Face to Path",
                    "value": metrics.face_to_path,
                    "unit": "\u00b0",
                    "format": "{:+.1f}",
                }

            # Kinetic energy
            if metrics.kinetic_energy is not None:
                hud["panels"]["kinetic_energy"] = {
                    "label": "Kinetic Energy",
                    "value": metrics.kinetic_energy,
                    "unit": "J",
                    "format": "{:.0f}",
                }

        return hud

    def format_value(self, panel_data: dict[str, Any]) -> str:
        """Format a panel value for display.

        Args:
            panel_data: Panel data dictionary.

        Returns:
            Formatted string.
        """
        fmt = panel_data.get("format", "{}")
        value = panel_data.get("value", 0)
        unit = panel_data.get("unit", "")
        return f"{fmt.format(value)} {unit}".strip()

    def get_compact_hud(self, metrics: SwingMetrics | None) -> dict[str, str]:
        """Get compact HUD with formatted strings.

        Args:
            metrics: Swing metrics.

        Returns:
            Dictionary of label to formatted value strings.
        """
        full_hud = self.get_hud_data(metrics)
        compact: dict[str, str] = {}

        for key, panel in full_hud.get("panels", {}).items():
            compact[panel["label"]] = self.format_value(panel)

        return compact
