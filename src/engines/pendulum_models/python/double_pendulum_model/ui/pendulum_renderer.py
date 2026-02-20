"""
3D rendering mixin for the Double Pendulum GUI.

Extracted from DoublePendulumApp to respect SRP:
3D visualization drawing is independent of physics simulation and UI controls.
"""

from __future__ import annotations

import math
import typing
from typing import Protocol

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    from ..core.dynamics import DoublePendulumDynamics
    from ..core.state import DoublePendulumState

    class RendererProtocol(Protocol):
        """Protocol for renderer host class."""

        ax: Axes
        canvas: FigureCanvasTkAgg
        state: DoublePendulumState | None
        dynamics: DoublePendulumDynamics | None


from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class PendulumRendererMixin:
    """3D visualization rendering for the double pendulum.

    Requires host class to provide:
        ax: Axes3D
        canvas: FigureCanvasTkAgg
        state: DoublePendulumState | None
        dynamics: DoublePendulumDynamics | None
    """

    def _draw_pendulum_3d(self) -> None:
        """Draw pendulum in 3D space using helper methods."""
        import numpy as np

        host = typing.cast("RendererProtocol", self)
        if host.state is None or host.dynamics is None:
            logger.debug("DEBUG: state=%s, dynamics=%s", host.state, host.dynamics)
            return

        try:
            host.ax.clear()
        except (RuntimeError, ValueError, OSError):
            logger.exception("Error clearing axes")
            return

        # Prepare
        pivot = np.array([0.0, 0.0, 0.0])
        upper = host.dynamics.parameters.upper_segment
        lower = host.dynamics.parameters.lower_segment
        max_range = (upper.length_m + lower.length_m) * 1.3

        # Calculate Positions
        pivot, elbow, wrist = self._calculate_3d_positions(pivot)

        # Draw Elements
        self._draw_reference_lines(pivot, max_range, host.state.theta1)
        self._draw_segments(pivot, elbow, wrist)
        self._draw_plane(upper.length_m + lower.length_m)

        # Finalize Plot
        host.ax.set_xlim([-max_range, max_range])
        host.ax.set_ylim([-max_range, max_range])
        host.ax.set_zlim([-max_range * 0.5, max_range * 0.5])
        host.ax.set_xlabel("X (m)", fontsize=10)
        host.ax.set_ylabel("Y (m)", fontsize=10)
        host.ax.set_zlabel("Z (m)", fontsize=10)
        host.ax.set_title(
            "Double Pendulum 3D View\nPivot at origin, t1=0 deg is vertical down",
            fontsize=11,
            fontweight="bold",
        )
        host.ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        host.canvas.draw()

    def _calculate_3d_positions(
        self, pivot: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Calculate the 3D positions of pendulum joints."""
        import numpy as np

        host = typing.cast("RendererProtocol", self)
        if host.state is None or host.dynamics is None:
            return pivot, pivot, pivot

        upper = host.dynamics.parameters.upper_segment
        lower = host.dynamics.parameters.lower_segment
        theta1 = host.state.theta1
        theta2 = host.state.theta2

        # Upper segment vector
        upper_vec = np.array(
            [
                math.sin(theta1) * upper.length_m,
                0.0,
                -math.cos(theta1) * upper.length_m,
            ]
        )
        elbow = pivot + upper_vec

        # Lower segment vector
        lower_abs_angle = theta1 + theta2
        lower_vec = np.array(
            [
                math.sin(lower_abs_angle) * lower.length_m,
                0.0,
                -math.cos(lower_abs_angle) * lower.length_m,
            ]
        )
        wrist = elbow + lower_vec

        # Apply rotations
        if not host.dynamics.parameters.constrained_to_plane:
            # Out of plane rotation
            phi = host.state.phi if hasattr(host.state, "phi") else 0.0
            elbow = self._rotate_out_of_plane(elbow, phi)
            wrist = self._rotate_out_of_plane(wrist, phi)

        if host.dynamics.parameters.constrained_to_plane:
            # Plane inclination rotation
            plane_angle = host.dynamics.parameters.plane_inclination_rad
            pivot = self._rotate_plane(pivot, plane_angle)
            elbow = self._rotate_plane(elbow, plane_angle)
            wrist = self._rotate_plane(wrist, plane_angle)

        return pivot, elbow, wrist

    @staticmethod
    def _rotate_out_of_plane(
        point: npt.NDArray[np.float64], phi: float
    ) -> npt.NDArray[np.float64]:
        """Rotate point around Z axis by phi."""
        import numpy as np

        x, y, z = point[0], point[1], point[2]
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        new_x = x * cos_phi - y * sin_phi
        new_y = x * sin_phi + y * cos_phi
        return np.array([new_x, new_y, z])

    @staticmethod
    def _rotate_plane(
        point: npt.NDArray[np.float64], angle: float
    ) -> npt.NDArray[np.float64]:
        """Rotate point around X axis by angle."""
        import numpy as np

        y, z = point[1], point[2]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        new_y = y * cos_a - z * sin_a
        new_z = y * sin_a + z * cos_a
        return np.array([point[0], new_y, new_z])

    def _draw_reference_lines(
        self,
        pivot: npt.NDArray[np.float64],
        max_range: float,
        theta1: float,
    ) -> None:
        """Draw reference lines and gravity."""
        import numpy as np

        host = typing.cast("RendererProtocol", self)
        # Vertical reference
        host.ax.plot(
            [pivot[0], pivot[0]],
            [pivot[1], pivot[1]],
            [pivot[2], pivot[2] - max_range * 0.3],
            "k--",
            linewidth=1,
            alpha=0.3,
            label="Plane Vertical (t1=0 deg)",
        )
        # Horizontal reference
        host.ax.plot(
            [pivot[0] - max_range * 0.3, pivot[0] + max_range * 0.3],
            [pivot[1], pivot[1]],
            [pivot[2], pivot[2]],
            "k--",
            linewidth=1,
            alpha=0.3,
            label="Plane Horizontal",
        )
        # Theta1 arc
        arc_theta = np.linspace(0, theta1, 20)
        arc_radius = max_range * 0.2
        arc_x = [pivot[0] + arc_radius * math.sin(t) for t in arc_theta]
        arc_z = [pivot[2] - arc_radius * math.cos(t) for t in arc_theta]
        if len(arc_x) > 1:
            host.ax.plot(arc_x, [pivot[1]] * 20, arc_z, "b-", linewidth=2, alpha=0.5)

        # Gravity
        self._draw_gravity_arrow(pivot, max_range)

    def _draw_gravity_arrow(
        self, pivot: npt.NDArray[np.float64], max_range: float
    ) -> None:
        """Draw the gravity vector and label."""
        import numpy as np

        host = typing.cast("RendererProtocol", self)
        gravity_len = max_range * 0.35
        g_start = pivot + np.array([max_range * 0.6, max_range * 0.2, max_range * 0.3])
        g_vec = np.array([0, 0, -gravity_len])
        host.ax.quiver(
            g_start[0],
            g_start[1],
            g_start[2],
            g_vec[0],
            g_vec[1],
            g_vec[2],
            color="#00AA00",
            arrow_length_ratio=0.3,
            linewidth=5,
            label="Gravity (g) - Always Vertical Down",
            alpha=0.95,
        )
        # Gravity label
        g_label_pos = g_start + g_vec * 0.5
        host.ax.text(
            g_label_pos[0] + max_range * 0.1,
            g_label_pos[1],
            g_label_pos[2],
            "g",
            fontsize=16,
            color="#00AA00",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#E8F5E9",
                "alpha": 0.95,
                "edgecolor": "#00AA00",
                "linewidth": 3,
            },
        )

    def _draw_segments(
        self,
        pivot: npt.NDArray[np.float64],
        elbow: npt.NDArray[np.float64],
        wrist: npt.NDArray[np.float64],
    ) -> None:
        """Draw the pendulum segments and joints."""
        self._draw_upper_segment(pivot, elbow)
        self._draw_lower_segment(elbow, wrist)
        self._draw_joint_markers(pivot, elbow, wrist)
        self._draw_segment_labels(pivot, elbow, wrist)

    def _draw_upper_segment(
        self,
        pivot: npt.NDArray[np.float64],
        elbow: npt.NDArray[np.float64],
    ) -> None:
        """Draw the upper pendulum segment."""
        host = typing.cast("RendererProtocol", self)
        host.ax.plot(
            [pivot[0], elbow[0]],
            [pivot[1], elbow[1]],
            [pivot[2], elbow[2]],
            color="#2E86AB",
            linewidth=7,
            label="Upper Segment (Shoulder)",
            alpha=0.9,
            zorder=5,
        )

    def _draw_lower_segment(
        self,
        elbow: npt.NDArray[np.float64],
        wrist: npt.NDArray[np.float64],
    ) -> None:
        """Draw the lower pendulum segment."""
        host = typing.cast("RendererProtocol", self)
        host.ax.plot(
            [elbow[0], wrist[0]],
            [elbow[1], wrist[1]],
            [elbow[2], wrist[2]],
            color="#A23B72",
            linewidth=8,
            label="Lower Segment (Wrist)",
            alpha=0.9,
            zorder=5,
        )

    def _draw_joint_markers(
        self,
        pivot: npt.NDArray[np.float64],
        elbow: npt.NDArray[np.float64],
        wrist: npt.NDArray[np.float64],
    ) -> None:
        """Draw the joint markers at pivot, elbow, and wrist."""
        host = typing.cast("RendererProtocol", self)
        host.ax.scatter(
            *pivot,
            color="black",
            s=250,
            marker="o",
            label="Pivot (Hub)",
            edgecolors="white",
            linewidths=3,
            zorder=10,
        )
        host.ax.scatter(
            *elbow,
            color="#2E86AB",
            s=100,
            marker="o",
            edgecolors="white",
            linewidths=2,
            zorder=9,
        )
        host.ax.scatter(
            *wrist,
            color="#A23B72",
            s=180,
            marker="o",
            label="End Point (Clubhead)",
            edgecolors="white",
            linewidths=3,
            zorder=9,
        )

    def _draw_segment_labels(
        self,
        pivot: npt.NDArray[np.float64],
        elbow: npt.NDArray[np.float64],
        wrist: npt.NDArray[np.float64],
    ) -> None:
        """Draw labels at midpoints of segments."""
        host = typing.cast("RendererProtocol", self)
        upper_mid = (pivot + elbow) / 2
        host.ax.text(
            upper_mid[0],
            upper_mid[1],
            upper_mid[2],
            "UPPER",
            fontsize=9,
            color="#2E86AB",
            weight="bold",
            ha="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "#2E86AB",
            },
        )

        lower_mid = (elbow + wrist) / 2
        host.ax.text(
            lower_mid[0],
            lower_mid[1],
            lower_mid[2],
            "LOWER",
            fontsize=9,
            color="#A23B72",
            weight="bold",
            ha="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "#A23B72",
            },
        )

    def _draw_plane(self, size: float) -> None:
        """Draw the inclined plane surface."""
        import numpy as np

        host = typing.cast("RendererProtocol", self)
        if host.dynamics is None or not host.dynamics.parameters.constrained_to_plane:
            return

        plane_size = size * 1.2
        x_plane = np.linspace(-plane_size, plane_size, 15)
        y_plane = np.linspace(-plane_size, plane_size, 15)
        x_grid, y_grid = np.meshgrid(x_plane, y_plane)

        angle = host.dynamics.parameters.plane_inclination_rad
        z_plane = y_grid * math.sin(angle)
        y_rot = y_grid * math.cos(angle)

        host.ax.plot_surface(
            x_grid, y_rot, z_plane, alpha=0.15, color="gray", edgecolor="none"
        )
