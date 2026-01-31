"""Dashboard plotting renderer."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class DashboardRenderer(BaseRenderer):
    """Renderer for dashboard and summary plots."""

    def plot_summary_dashboard(self, fig: Figure) -> None:
        """Create a comprehensive dashboard with multiple subplots."""
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Club head speed (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        times, speeds = self.data.get_series("club_head_speed")
        speeds = np.asarray(speeds)
        if len(times) > 0 and len(speeds) > 0:
            speeds_mph = speeds * 2.23694
            ax1.plot(times, speeds_mph, linewidth=2, color=self.colors["primary"])
            ax1.fill_between(
                times, 0, speeds_mph, alpha=0.3, color=self.colors["primary"]
            )
            max_speed = np.max(speeds_mph)
            ax1.set_title(
                f"Club Speed (Peak: {max_speed:.1f} mph)",
                fontsize=11,
                fontweight="bold",
            )
            ax1.set_xlabel("Time (s)", fontsize=9)
            ax1.set_ylabel("Speed (mph)", fontsize=9)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No club head data", ha="center", va="center")

        # 2. Energy (Top Center)
        ax2 = fig.add_subplot(gs[0, 1])
        times_ke, ke = self.data.get_series("kinetic_energy")
        times_pe, pe = self.data.get_series("potential_energy")
        if len(times_ke) > 0:
            ax2.plot(
                times_ke, ke, label="KE", linewidth=2, color=self.colors["primary"]
            )
            ax2.plot(
                times_pe, pe, label="PE", linewidth=2, color=self.colors["secondary"]
            )
            ax2.set_title("Energy", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Time (s)", fontsize=9)
            ax2.set_ylabel("Energy (J)", fontsize=9)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No energy data", ha="center", va="center")

        # 3. Angular Momentum (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        times_am, am = self.data.get_series("angular_momentum")
        am = np.asarray(am)
        if len(times_am) > 0 and am.size > 0:
            am_mag = np.sqrt(np.sum(am**2, axis=1))
            ax3.plot(
                times_am,
                am_mag,
                label="Mag",
                linewidth=2,
                color=self.colors["quaternary"],
            )
            ax3.set_title("Angular Momentum", fontsize=11, fontweight="bold")
            ax3.set_xlabel("Time (s)", fontsize=9)
            ax3.set_ylabel("L (kg m²/s)", fontsize=9)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No AM data", ha="center", va="center")

        # 4. Joint Angles (Bottom Left)
        ax4 = fig.add_subplot(gs[1, 0])
        times, positions = self.data.get_series("joint_positions")
        positions = np.asarray(positions)
        if len(times) > 0 and len(positions) > 0 and positions.ndim >= 2:
            for idx in range(min(3, positions.shape[1])):  # Plot first 3 joints
                ax4.plot(
                    times,
                    np.rad2deg(positions[:, idx]),
                    label=self.data.get_joint_name(idx),
                    linewidth=2,
                )
            ax4.set_title("Joint Angles", fontsize=11, fontweight="bold")
            ax4.set_xlabel("Time (s)", fontsize=9)
            ax4.set_ylabel("Angle (deg)", fontsize=9)
            ax4.legend(fontsize=7, loc="best")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No position data", ha="center", va="center")

        # 5. CoP (Bottom Center)
        ax5 = fig.add_subplot(gs[1, 1])
        times_cop, cop = self.data.get_series("cop_position")
        cop = np.asarray(cop)
        if len(times_cop) > 0 and cop.size > 0:
            ax5.scatter(cop[:, 0], cop[:, 1], c=times_cop, cmap="viridis", s=10)
            ax5.set_title("CoP Trajectory", fontsize=11, fontweight="bold")
            ax5.set_xlabel("X (m)", fontsize=9)
            ax5.set_ylabel("Y (m)", fontsize=9)
            ax5.axis("equal")
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No CoP data", ha="center", va="center")

        # 6. Torques (Bottom Right)
        ax6 = fig.add_subplot(gs[1, 2])
        times, torques = self.data.get_series("joint_torques")
        torques = np.asarray(torques)
        if len(times) > 0 and len(torques) > 0 and torques.ndim >= 2:
            for idx in range(min(3, torques.shape[1])):
                ax6.plot(
                    times,
                    torques[:, idx],
                    label=self.data.get_joint_name(idx),
                    linewidth=2,
                )
            ax6.set_title("Joint Torques", fontsize=11, fontweight="bold")
            ax6.set_xlabel("Time (s)", fontsize=9)
            ax6.set_ylabel("Torque (Nm)", fontsize=9)
            ax6.legend(fontsize=7, loc="best")
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, "No torque data", ha="center", va="center")

        fig.suptitle(
            "Golf Swing Analysis Dashboard",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

    def plot_radar_chart(
        self,
        fig: Figure,
        metrics: dict[str, float],
        title: str = "Swing Profile",
        ax: Axes | None = None,
    ) -> None:
        """Plot a radar chart of swing metrics."""
        labels = list(metrics.keys())
        values = list(metrics.values())
        num_vars = len(labels)

        if ax is None:
            ax = fig.add_subplot(111, polar=True)

        if num_vars < 3:
            ax.text(
                0.5,
                0.5,
                "Need at least 3 metrics for radar chart",
                ha="center",
                va="center",
            )
            return

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        values += values[:1]
        angles += angles[:1]
        labels += labels[:1]

        ax.plot(angles, values, color=self.colors["primary"], linewidth=2)
        ax.fill(angles, values, color=self.colors["primary"], alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels[:-1])

        ax.grid(True, alpha=0.3)

        ax.set_title(title, size=15, color=self.colors["primary"], y=1.1)
        fig.tight_layout()
