from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.plotting.renderers.base import BaseRenderer

logger = get_logger(__name__)


class CoordinationPhaseMixin(BaseRenderer):
    def plot_coupling_angle(
        self,
        fig: Figure,
        coupling_angles: np.ndarray,
        title: str | None = None,
        ax: Axes | None = None,
    ) -> None:
        """Plot Coupling Angle time series (Vector Coding)."""
        if fig is None:
            raise ValueError("fig must be provided")
        times, _ = self.data.get_series("joint_positions")

        if ax is None:
            ax = fig.add_subplot(111)

        if len(times) == 0 or len(coupling_angles) == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        if len(coupling_angles) != len(times):
            logger.warning(
                f"Coupling angle length ({len(coupling_angles)}) does not match "
                f"time series length ({len(times)}). Truncating times."
            )
            plot_times = times[: len(coupling_angles)]
        else:
            plot_times = times

        ax.plot(
            plot_times,
            coupling_angles,
            color=self.colors["primary"],
            linewidth=2,
            label="Coupling Angle",
        )

        for angle in [0, 90, 180, 270, 360]:
            ax.axhline(y=angle, color="gray", linestyle="--", alpha=0.3)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Coupling Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_title(
            title or "Coordination Variability", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    def plot_coordination_patterns(
        self,
        fig: Figure,
        coupling_angles: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Plot coordination patterns as a color-coded strip over time."""
        if fig is None:
            raise ValueError("fig must be provided")
        times, _ = self.data.get_series("joint_positions")

        if len(times) == 0 or len(coupling_angles) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        if len(coupling_angles) != len(times):
            times = times[: len(coupling_angles)]

        binned = np.floor((coupling_angles + 22.5) / 45.0) % 8

        classes = np.zeros_like(binned)
        classes[(binned == 0) | (binned == 4)] = 0  # Proximal
        classes[(binned == 1) | (binned == 5)] = 1  # In-Phase
        classes[(binned == 2) | (binned == 6)] = 2  # Distal
        classes[(binned == 3) | (binned == 7)] = 3  # Anti-Phase

        cmap_colors = [
            self.colors["primary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["secondary"],
        ]

        cmap = ListedColormap(cmap_colors)

        ax = fig.add_subplot(111)

        if len(times) > 1:
            dt = times[1] - times[0]
            time_edges = np.concatenate(
                (
                    [times[0] - dt / 2],
                    times[:-1] + np.diff(times) / 2,
                    [times[-1] + dt / 2],
                )
            )
        else:
            time_edges = np.array([times[0] - 0.5, times[0] + 0.5])

        y_edges = np.array([0, 1])
        X, Y = np.meshgrid(time_edges, y_edges)
        C = classes.reshape(1, -1)

        ax.pcolormesh(X, Y, C, cmap=cmap, vmin=0, vmax=3, shading="flat")

        legend_patches = [
            Rectangle((0, 0), 1, 1, color=cmap_colors[0], label="Proximal Leading"),
            Rectangle((0, 0), 1, 1, color=cmap_colors[1], label="In-Phase"),
            Rectangle((0, 0), 1, 1, color=cmap_colors[2], label="Distal Leading"),
            Rectangle((0, 0), 1, 1, color=cmap_colors[3], label="Anti-Phase"),
        ]

        ax.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=4,
        )

        ax.set_yticks([])
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or "Coordination Pattern Dynamics",
            fontsize=14,
            fontweight="bold",
            y=1.2,
        )

        fig.tight_layout()

    def plot_continuous_relative_phase(
        self,
        fig: Figure,
        crp_data: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Plot Continuous Relative Phase (CRP) time series."""
        if fig is None:
            raise ValueError("fig must be provided")
        times, _ = self.data.get_series("joint_positions")

        if len(times) == 0 or len(crp_data) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        plot_times = times[: len(crp_data)] if len(crp_data) != len(times) else times

        ax = fig.add_subplot(111)

        ax.plot(
            plot_times,
            crp_data,
            color=self.colors["primary"],
            linewidth=2,
            label="CRP",
        )

        ax.axhline(y=0, color="green", linestyle="--", alpha=0.3, label="In-Phase")
        ax.axhline(y=180, color="red", linestyle="--", alpha=0.3, label="Anti-Phase")
        ax.axhline(y=-180, color="red", linestyle="--", alpha=0.3)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Relative Phase (deg)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or "Continuous Relative Phase", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
