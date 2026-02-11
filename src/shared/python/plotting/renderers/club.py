"""Club plotting renderer."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from src.shared.python.biomechanics.swing_plane_analysis import SwingPlaneAnalyzer
from src.shared.python.plotting.renderers.base import BaseRenderer


class ClubRenderer(BaseRenderer):
    """Renderer for club-related plots."""

    def plot_club_head_speed(self, fig: Figure) -> None:
        """Plot club head speed over time."""
        times, speeds = self.data.get_series("club_head_speed")

        if len(times) == 0 or len(speeds) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No club head data", ha="center", va="center")
            return

        if not isinstance(speeds, np.ndarray):
            speeds = np.array(speeds)

        ax = fig.add_subplot(111)
        speeds_mph = speeds * 2.23694

        ax.plot(times, speeds_mph, linewidth=3, color=self.colors["primary"])
        ax.fill_between(times, 0, speeds_mph, alpha=0.3, color=self.colors["primary"])

        max_idx = np.argmax(speeds_mph)
        max_speed = speeds_mph[max_idx]
        max_time = times[max_idx]
        ax.plot(
            max_time,
            max_speed,
            "r*",
            markersize=20,
            label=f"Peak: {max_speed:.1f} mph",
        )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Club Head Speed (mph)", fontsize=12, fontweight="bold")
        ax.set_title("Club Head Speed vs Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_club_head_trajectory(self, fig: Figure) -> None:
        """Plot 3D club head trajectory."""
        times, positions = self.data.get_series("club_head_position")

        if len(times) == 0 or len(positions) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No club head data", ha="center", va="center")
            return

        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)

        ax = fig.add_subplot(111, projection="3d")

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        sc = ax.scatter(x, y, z, c=times, cmap="viridis", s=20)
        ax.plot(x, y, z, alpha=0.3, color="gray", linewidth=1)

        ax.scatter(
            [x[0]],
            [y[0]],
            [z[0]],
            color="green",
            s=100,
            marker="o",
            label="Start",
        )
        ax.scatter(
            [x[-1]],
            [y[-1]],
            [z[-1]],
            color="red",
            s=100,
            marker="s",
            label="End",
        )

        ax.set_xlabel("X (m)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Y (m)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Z (m)", fontsize=10, fontweight="bold")
        ax.set_title("Club Head 3D Trajectory", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)
        fig.tight_layout()

    def plot_swing_plane(self, fig: Figure) -> None:
        """Plot fitted swing plane and trajectory deviation."""
        times, positions = self.data.get_series("club_head_position")

        if len(times) < 3 or len(positions) < 3:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Insufficient data for plane fitting",
                ha="center",
                va="center",
            )
            return

        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)

        analyzer = SwingPlaneAnalyzer()
        try:
            metrics = analyzer.analyze(positions)
        except ValueError as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            return

        ax = fig.add_subplot(111, projection="3d")

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        centroid = metrics.point_on_plane
        normal = metrics.normal_vector
        deviations = analyzer.calculate_deviation(positions, centroid, normal)

        sc = ax.scatter(
            x,
            y,
            zs=z,
            c=np.abs(deviations),
            cmap="coolwarm",
            s=20,
            label="Trajectory",
        )

        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        margin = 0.5
        xx, yy = np.meshgrid(
            np.linspace(min_x - margin, max_x + margin, 10),
            np.linspace(min_y - margin, max_y + margin, 10),
        )

        if abs(float(normal[2])) > 1e-6:
            zz = (
                centroid[2]
                - (normal[0] * (xx - centroid[0]) + normal[1] * (yy - centroid[1]))
                / normal[2]
            )
            ax.plot_surface(xx, yy, zz, alpha=0.2, color="cyan")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(
            f"Swing Plane Analysis\nSteepness: {metrics.steepness_deg:.1f}°, "
            f"RMSE: {metrics.rmse * 100:.1f} cm",
            fontsize=12,
            fontweight="bold",
        )

        fig.colorbar(sc, ax=ax, label="Deviation from Plane (m)", shrink=0.6)
        fig.tight_layout()

    def plot_club_induced_acceleration(
        self,
        fig: Figure,
        breakdown_mode: bool = True,
    ) -> None:
        """Plot club head task-space induced accelerations."""
        ax = fig.add_subplot(111)

        components = ["gravity", "velocity", "control", "constraint", "total"]
        labels = [
            "Gravity",
            "Velocity (Kinematic)",
            "Control (Muscle)",
            "Constraint",
            "Total",
        ]
        colors = [
            self.colors["secondary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["quinary"],
            "black",
        ]
        styles = ["--", "-.", ":", "--", "-"]

        has_data = False

        for comp, label, color, style in zip(
            components, labels, colors, styles, strict=False
        ):
            times, acc_vec = self.data.get_club_induced_acceleration_series(comp)

            if len(times) > 0 and acc_vec.size > 0:
                mag = np.sqrt(np.sum(acc_vec**2, axis=1))

                if np.max(mag) > 1e-4 or comp == "total":
                    ax.plot(
                        times,
                        mag,
                        label=label,
                        color=color,
                        linestyle=style,
                        linewidth=2,
                    )
                    has_data = True

        if not has_data:
            ax.text(
                0.5, 0.5, "No Club Induced Acceleration Data", ha="center", va="center"
            )
            return

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Acceleration Magnitude (m/s²)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Club Head Acceleration Contributors", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
