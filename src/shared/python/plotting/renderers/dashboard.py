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

        self._dash_club_speed(fig.add_subplot(gs[0, 0]))
        self._dash_energy(fig.add_subplot(gs[0, 1]))
        self._dash_angular_momentum(fig.add_subplot(gs[0, 2]))
        self._dash_joint_angles(fig.add_subplot(gs[1, 0]))
        self._dash_cop(fig.add_subplot(gs[1, 1]))
        self._dash_torques(fig.add_subplot(gs[1, 2]))

        fig.suptitle(
            "Golf Swing Analysis Dashboard",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

    def _dash_club_speed(self, ax: Axes) -> None:
        """Dashboard panel: club head speed."""
        times, speeds = self.data.get_series("club_head_speed")
        speeds = np.asarray(speeds)
        if len(times) > 0 and len(speeds) > 0:
            speeds_mph = speeds * 2.23694
            ax.plot(times, speeds_mph, linewidth=2, color=self.colors["primary"])
            ax.fill_between(
                times, 0, speeds_mph, alpha=0.3, color=self.colors["primary"]
            )
            ax.set_title(
                f"Club Speed (Peak: {np.max(speeds_mph):.1f} mph)",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Speed (mph)", fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No club head data", ha="center", va="center")

    def _dash_energy(self, ax: Axes) -> None:
        """Dashboard panel: kinetic and potential energy."""
        times_ke, ke = self.data.get_series("kinetic_energy")
        times_pe, pe = self.data.get_series("potential_energy")
        if len(times_ke) > 0:
            ax.plot(times_ke, ke, label="KE", linewidth=2, color=self.colors["primary"])
            ax.plot(
                times_pe, pe, label="PE", linewidth=2, color=self.colors["secondary"]
            )
            ax.set_title("Energy", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Energy (J)", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No energy data", ha="center", va="center")

    def _dash_angular_momentum(self, ax: Axes) -> None:
        """Dashboard panel: angular momentum magnitude."""
        times_am, am = self.data.get_series("angular_momentum")
        am = np.asarray(am)
        if len(times_am) > 0 and am.size > 0:
            am_mag = np.sqrt(np.sum(am**2, axis=1))
            ax.plot(
                times_am,
                am_mag,
                label="Mag",
                linewidth=2,
                color=self.colors["quaternary"],
            )
            ax.set_title("Angular Momentum", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("L (kg m²/s)", fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No AM data", ha="center", va="center")

    def _dash_joint_angles(self, ax: Axes) -> None:
        """Dashboard panel: joint angles (first 3 joints)."""
        times, positions = self.data.get_series("joint_positions")
        positions = np.asarray(positions)
        if len(times) > 0 and len(positions) > 0 and positions.ndim >= 2:
            for idx in range(min(3, positions.shape[1])):
                ax.plot(
                    times,
                    np.rad2deg(positions[:, idx]),
                    label=self.data.get_joint_name(idx),
                    linewidth=2,
                )
            ax.set_title("Joint Angles", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Angle (deg)", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No position data", ha="center", va="center")

    def _dash_cop(self, ax: Axes) -> None:
        """Dashboard panel: center of pressure trajectory."""
        times_cop, cop = self.data.get_series("cop_position")
        cop = np.asarray(cop)
        if len(times_cop) > 0 and cop.size > 0:
            ax.scatter(cop[:, 0], cop[:, 1], c=times_cop, cmap="viridis", s=10)
            ax.set_title("CoP Trajectory", fontsize=11, fontweight="bold")
            ax.set_xlabel("X (m)", fontsize=9)
            ax.set_ylabel("Y (m)", fontsize=9)
            ax.axis("equal")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No CoP data", ha="center", va="center")

    def _dash_torques(self, ax: Axes) -> None:
        """Dashboard panel: joint torques (first 3 joints)."""
        times, torques = self.data.get_series("joint_torques")
        torques = np.asarray(torques)
        if len(times) > 0 and len(torques) > 0 and torques.ndim >= 2:
            for idx in range(min(3, torques.shape[1])):
                ax.plot(
                    times,
                    torques[:, idx],
                    label=self.data.get_joint_name(idx),
                    linewidth=2,
                )
            ax.set_title("Joint Torques", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Torque (Nm)", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No torque data", ha="center", va="center")

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
