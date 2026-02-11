"""Coordination and sequencing plotting renderer."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.plotting.renderers.base import BaseRenderer

logger = get_logger(__name__)


class CoordinationRenderer(BaseRenderer):
    """Renderer for coordination, sequencing, and variability plots."""

    def plot_coupling_angle(
        self,
        fig: Figure,
        coupling_angles: np.ndarray,
        title: str | None = None,
        ax: Axes | None = None,
    ) -> None:
        """Plot Coupling Angle time series (Vector Coding)."""
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
        times, _ = self.data.get_series("joint_positions")

        if len(times) == 0 or len(crp_data) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        if len(crp_data) != len(times):
            plot_times = times[: len(crp_data)]
        else:
            plot_times = times

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

    def plot_dtw_alignment(
        self,
        fig: Figure,
        times1: np.ndarray,
        data1: np.ndarray,
        times2: np.ndarray,
        data2: np.ndarray,
        path: list[tuple[int, int]],
        title: str = "Sequence Alignment",
    ) -> None:
        """Plot alignment between two sequences (DTW)."""
        ax = fig.add_subplot(111)

        offset = np.max(data1) - np.min(data2) + 1.0

        ax.plot(
            times1,
            data1 + offset,
            label="Reference",
            color=self.colors["primary"],
            linewidth=2,
        )
        ax.plot(
            times2, data2, label="Student", color=self.colors["secondary"], linewidth=2
        )

        step = max(1, len(path) // 50)
        for idx in range(0, len(path), step):
            i, j = path[idx]
            ax.plot(
                [times1[i], times2[j]],
                [data1[i] + offset, data2[j]],
                color="gray",
                alpha=0.3,
                linewidth=1,
                linestyle="--",
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_yticks([])
        ax.legend()
        fig.tight_layout()

    def plot_cross_recurrence_plot(
        self,
        fig: Figure,
        recurrence_matrix: np.ndarray,
        title: str = "Cross Recurrence Plot",
    ) -> None:
        """Plot Cross Recurrence Plot."""
        if recurrence_matrix.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No CRP Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        ax.imshow(recurrence_matrix, cmap="Greys", origin="lower", aspect="auto")

        ax.set_xlabel("Time Step (System 2)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Time Step (System 1)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")

        if recurrence_matrix.shape[0] == recurrence_matrix.shape[1]:
            ax.plot(
                [0, recurrence_matrix.shape[1] - 1],
                [0, recurrence_matrix.shape[0] - 1],
                color="red",
                linestyle="--",
                alpha=0.5,
            )

        fig.tight_layout()

    def plot_recurrence_plot(
        self,
        fig: Figure,
        recurrence_matrix: np.ndarray,
        title: str = "Recurrence Plot",
    ) -> None:
        """Plot Recurrence Plot (binary matrix)."""
        if recurrence_matrix.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Recurrence Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        ax.imshow(
            recurrence_matrix,
            cmap="Greys",
            origin="lower",
            interpolation="none",
        )

        ax.set_xlabel("Time Step (j)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Time Step (i)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

    def plot_correlation_sum(
        self,
        fig: Figure,
        radii: np.ndarray,
        counts: np.ndarray,
        slope_region: slice | None = None,
        slope_val: float | None = None,
    ) -> None:
        """Plot Correlation Sum C(r) vs r on log-log scale."""
        ax = fig.add_subplot(111)

        log_r = np.log(radii)
        log_c = np.log(counts)

        ax.plot(log_r, log_c, "o-", color=self.colors["primary"], markersize=4)

        if slope_region and slope_val is not None:
            reg_r = log_r[slope_region]
            reg_c = log_c[slope_region]

            c = np.mean(reg_c) - slope_val * np.mean(reg_r)
            fit_line = slope_val * reg_r + c

            ax.plot(
                reg_r,
                fit_line,
                color="red",
                linewidth=2,
                linestyle="--",
                label=f"Slope (D2) = {slope_val:.2f}",
            )
            ax.legend()

        ax.set_xlabel("log(r)", fontsize=12, fontweight="bold")
        ax.set_ylabel("log(C(r))", fontsize=12, fontweight="bold")
        ax.set_title(
            "Correlation Sum (Grassberger-Procaccia)", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, which="both", linestyle="--")
        fig.tight_layout()

    def plot_lag_matrix(
        self,
        fig: Figure,
        data_type: str = "velocity",
        max_lag: float = 0.5,
    ) -> None:
        """Plot time lag matrix between joints."""
        try:
            from src.shared.python.validation_pkg.statistical_analysis import (
                StatisticalAnalyzer,
            )
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Analysis module missing", ha="center", va="center")
            return

        times, positions = self.data.get_series("joint_positions")
        _, velocities = self.data.get_series("joint_velocities")
        _, torques = self.data.get_series("joint_torques")

        if len(times) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        analyzer = StatisticalAnalyzer(
            times=np.asarray(times),
            joint_positions=np.asarray(positions),
            joint_velocities=np.asarray(velocities),
            joint_torques=np.asarray(torques),
        )

        lag_matrix, labels = analyzer.compute_lag_matrix(data_type, max_lag)

        if lag_matrix.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Could not compute lag matrix", ha="center", va="center")
            return

        ax = fig.add_subplot(111)
        im = ax.imshow(lag_matrix, cmap="RdBu_r", vmin=-max_lag, vmax=max_lag)

        if len(labels) <= 12:
            real_labels = [
                self.data.get_joint_name(int(lbl[1:])) if lbl.startswith("J") else lbl
                for lbl in labels
            ]
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(real_labels, rotation=45, ha="right")
            ax.set_yticklabels(real_labels)
        else:
            ax.set_xlabel("Joint Index (Lagging)")
            ax.set_ylabel("Joint Index (Leading)")

        ax.set_title(f"Time Lag Matrix ({data_type})", fontsize=14, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Time Lag (s)\n(Pos: Row leads Col)", rotation=270, labelpad=20)
        fig.tight_layout()

    def plot_kinematic_sequence(
        self,
        fig: Figure,
        segment_indices: dict[str, int],
        analyzer_result: Any | None = None,
    ) -> None:
        """Plot kinematic sequence (normalized velocities)."""
        times, velocities = self.data.get_series("joint_velocities")
        velocities = np.asarray(velocities)

        if len(times) == 0 or len(velocities) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        ax = fig.add_subplot(111)
        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["quinary"],
        ]

        for i, (name, idx) in enumerate(segment_indices.items()):
            if idx < velocities.shape[1]:
                vel = np.abs(velocities[:, idx])
                max_vel = float(np.max(vel))
                if max_vel > 0:
                    vel_norm = vel / max_vel
                else:
                    vel_norm = vel

                color = colors[i % len(colors)]
                ax.plot(times, vel_norm, label=name, color=color, linewidth=2)

                if analyzer_result:
                    peak_info = next(
                        (p for p in analyzer_result.peaks if p.name == name), None
                    )
                    if peak_info:
                        ax.plot(
                            peak_info.time,
                            peak_info.normalized_velocity,
                            "o",
                            color=color,
                            markersize=8,
                        )
                        order_idx = analyzer_result.sequence_order.index(name) + 1
                        ax.text(
                            peak_info.time,
                            peak_info.normalized_velocity + 0.05,
                            f"{order_idx}",
                            color=color,
                            fontsize=10,
                            fontweight="bold",
                            ha="center",
                        )
                else:
                    max_t_idx = np.argmax(vel)
                    ax.plot(
                        times[max_t_idx],
                        vel_norm[max_t_idx],
                        "o",
                        color=color,
                        markersize=8,
                    )

        title = "Kinematic Sequence (Normalized)"
        if analyzer_result:
            score = analyzer_result.efficiency_score * 100
            title += f"\nEfficiency Score: {score:.1f}%"
            if not analyzer_result.is_valid_sequence:
                title += " (Out of Order)"

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Normalized Velocity", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_kinematic_sequence_bars(
        self,
        fig: Figure,
        segment_indices: dict[str, int],
        impact_time: float | None = None,
    ) -> None:
        """Plot kinematic sequence as a Gantt-style bar chart of peak times."""
        times, velocities = self.data.get_series("joint_velocities")
        velocities = np.asarray(velocities)

        if len(times) == 0 or velocities.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        peaks = []
        names = []
        for name, idx in segment_indices.items():
            if idx < velocities.shape[1]:
                vel_abs = np.abs(velocities[:, idx])
                peak_idx = np.argmax(vel_abs)
                peaks.append(times[peak_idx])
                names.append(name)

        if not peaks:
            ax.text(0.5, 0.5, "No valid segments", ha="center", va="center")
            return

        ref_time = impact_time if impact_time is not None else peaks[-1]
        rel_times = np.array(peaks) - ref_time
        y_pos = np.arange(len(names))

        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["quinary"],
        ][: len(names)]

        ax.hlines(
            y=y_pos,
            xmin=min(0, np.min(rel_times) - 0.05),
            xmax=rel_times,
            color="gray",
            alpha=0.5,
        )
        ax.scatter(rel_times, y_pos, color=colors, s=100, zorder=3)

        for i, t in enumerate(rel_times):
            ax.text(
                t,
                i + 0.15,
                f"{t * 1000:.0f} ms",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=colors[i],
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontweight="bold", fontsize=11)
        ax.set_xlabel("Time relative to Impact (s)", fontsize=12, fontweight="bold")
        ax.set_title("Kinematic Sequence Timing", fontsize=14, fontweight="bold")
        ax.axvline(0, color="black", linestyle="--", alpha=0.8, label="Impact")

        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend()
        fig.tight_layout()

    def plot_x_factor_cycle(
        self,
        fig: Figure,
        shoulder_idx: int,
        hip_idx: int,
    ) -> None:
        """Plot X-Factor Cycle (Stretch-Shortening Cycle)."""
        times, positions = self.data.get_series("joint_positions")
        positions = np.asarray(positions)

        if (
            len(times) < 2
            or positions.ndim < 2
            or shoulder_idx >= positions.shape[1]
            or hip_idx >= positions.shape[1]
        ):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return

        shoulder_rot = np.rad2deg(positions[:, shoulder_idx])
        hip_rot = np.rad2deg(positions[:, hip_idx])
        x_factor = shoulder_rot - hip_rot

        dt = float(np.mean(np.diff(times)))
        if dt <= 0:
            dt = 0.01
        x_factor_vel = np.gradient(x_factor, dt)

        ax = fig.add_subplot(111)

        sc = ax.scatter(x_factor, x_factor_vel, c=times, cmap="magma", s=30, alpha=0.6)
        ax.plot(x_factor, x_factor_vel, alpha=0.3, color="gray", linewidth=1)

        max_idx = np.argmax(x_factor)
        ax.scatter(
            x_factor[max_idx],
            x_factor_vel[max_idx],
            c="blue",
            s=150,
            marker="*",
            label=f"Peak Stretch: {x_factor[max_idx]:.1f}°",
            zorder=10,
        )

        ax.set_xlabel("X-Factor (degrees)", fontsize=12, fontweight="bold")
        ax.set_ylabel("X-Factor Velocity (deg/s)", fontsize=12, fontweight="bold")
        ax.set_title(
            "X-Factor Stretch-Shortening Cycle", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="k", linestyle="-", alpha=0.3)
        ax.legend(loc="best")

        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_muscle_synergies(
        self,
        fig: Figure,
        synergy_result: Any,
    ) -> None:
        """Plot extracted muscle synergies (Weights and Activations)."""
        if not hasattr(synergy_result, "weights") or not hasattr(
            synergy_result, "activations"
        ):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Invalid SynergyResult object", ha="center", va="center")
            return

        n_synergies = synergy_result.n_synergies
        n_muscles = synergy_result.weights.shape[0]

        gs = fig.add_gridspec(
            n_synergies, 2, width_ratios=[1, 2], hspace=0.4, wspace=0.3
        )

        times, _ = self.data.get_series("joint_positions")
        if len(times) != synergy_result.activations.shape[1]:
            times = np.linspace(
                times[0], times[-1], synergy_result.activations.shape[1]
            )

        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["quinary"],
            self.colors["senary"],
        ]

        muscle_names = synergy_result.muscle_names or [
            f"M{i}" for i in range(n_muscles)
        ]

        for i in range(n_synergies):
            color = colors[i % len(colors)]

            ax_w = fig.add_subplot(gs[i, 0])
            weights = synergy_result.weights[:, i]

            y_pos = np.arange(n_muscles)
            ax_w.barh(y_pos, weights, color=color, alpha=0.8)
            ax_w.set_yticks(y_pos)

            if i == n_synergies - 1:
                ax_w.set_xlabel("Weight", fontsize=9)

            ax_w.set_yticklabels(muscle_names, fontsize=8)
            ax_w.invert_yaxis()
            ax_w.set_title(f"Synergy {i + 1} Weights", fontsize=10, fontweight="bold")
            ax_w.grid(True, axis="x", alpha=0.3)

            ax_h = fig.add_subplot(gs[i, 1])
            activation = synergy_result.activations[i, :]

            ax_h.plot(times, activation, color=color, linewidth=2)
            ax_h.fill_between(times, 0, activation, color=color, alpha=0.2)

            if i == n_synergies - 1:
                ax_h.set_xlabel("Time (s)", fontsize=10)

            ax_h.set_title(
                f"Synergy {i + 1} Activation", fontsize=10, fontweight="bold"
            )
            ax_h.grid(True, alpha=0.3)

        fig.suptitle(
            f"Muscle Synergies (VAF: {synergy_result.vaf * 100:.1f}%)",
            fontsize=14,
            fontweight="bold",
        )

    def plot_correlation_matrix(
        self,
        fig: Figure,
        data_type: str = "velocity",
    ) -> None:
        """Plot correlation matrix between joints."""
        if data_type == "position":
            _, data = self.data.get_series("joint_positions")
            title = "Joint Position Correlation"
        elif data_type == "torque":
            _, data = self.data.get_series("joint_torques")
            title = "Joint Torque Correlation"
        else:
            _, data = self.data.get_series("joint_velocities")
            title = "Joint Velocity Correlation"

        data = np.asarray(data)
        if len(data) == 0 or data.ndim < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        corr_matrix = np.corrcoef(data.T)

        ax = fig.add_subplot(111)
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

        if data.shape[1] <= 10:
            labels = [self.data.get_joint_name(i) for i in range(data.shape[1])]
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
        else:
            ax.set_xlabel("Joint Index")
            ax.set_ylabel("Joint Index")

        if data.shape[1] <= 8:
            n = data.shape[1]
            i_coords, j_coords = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            i_flat = i_coords.ravel()
            j_flat = j_coords.ravel()
            values_flat = corr_matrix.ravel()
            colors = np.where(np.abs(values_flat) < 0.5, "k", "w")

            for idx in range(len(i_flat)):
                ax.text(
                    j_flat[idx],
                    i_flat[idx],
                    f"{values_flat[idx]:.2f}",
                    ha="center",
                    va="center",
                    color=colors[idx],
                    fontsize=8,
                )

        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Correlation Coefficient")
        fig.tight_layout()

    def plot_dynamic_correlation(
        self,
        fig: Figure,
        joint_idx_1: int,
        joint_idx_2: int,
        window_size: int = 20,
    ) -> None:
        """Plot Rolling Correlation between two joint velocities."""
        try:
            from src.shared.python.validation_pkg.statistical_analysis import (
                StatisticalAnalyzer,
            )
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Analysis module missing", ha="center", va="center")
            return

        times, positions = self.data.get_series("joint_positions")
        _, velocities = self.data.get_series("joint_velocities")
        _, torques = self.data.get_series("joint_torques")

        if len(times) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        analyzer = StatisticalAnalyzer(
            times=np.asarray(times),
            joint_positions=np.asarray(positions),
            joint_velocities=np.asarray(velocities),
            joint_torques=np.asarray(torques),
        )

        try:
            w_times, corrs = analyzer.compute_rolling_correlation(
                joint_idx_1, joint_idx_2, window_size, data_type="velocity"
            )
        except AttributeError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Method not available", ha="center", va="center")
            return

        if len(w_times) == 0:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "Insufficient data for correlation", ha="center", va="center"
            )
            return

        ax = fig.add_subplot(111)
        ax.plot(w_times, corrs, color=self.colors["primary"], linewidth=2)

        name1 = self.data.get_joint_name(joint_idx_1)
        name2 = self.data.get_joint_name(joint_idx_2)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Correlation Coefficient", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Dynamic Correlation: {name1} vs {name2}\n(Window={window_size})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="k", linestyle="-", alpha=0.3)
        fig.tight_layout()

    def plot_synergy_trajectory(
        self,
        fig: Figure,
        synergy_result: Any,
        dim1: int = 0,
        dim2: int = 1,
    ) -> None:
        """Plot trajectory in synergy space (Activation 1 vs Activation 2)."""
        if not hasattr(synergy_result, "activations"):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Invalid SynergyResult", ha="center", va="center")
            return

        activations = synergy_result.activations
        if activations.shape[0] <= max(dim1, dim2):
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "Not enough synergies extracted", ha="center", va="center"
            )
            return

        times, _ = self.data.get_series("joint_positions")
        n_samples = min(len(times), activations.shape[1])
        act1 = activations[dim1, :n_samples]
        act2 = activations[dim2, :n_samples]
        plot_times = times[:n_samples]

        ax = fig.add_subplot(111)
        sc = ax.scatter(act1, act2, c=plot_times, cmap="viridis", s=30, alpha=0.8)
        ax.plot(act1, act2, color="gray", alpha=0.3, linewidth=1)

        ax.scatter(act1[0], act2[0], color="green", s=100, label="Start")
        ax.scatter(act1[-1], act2[-1], color="red", s=100, marker="s", label="End")

        ax.set_xlabel(f"Synergy {dim1 + 1} Activation", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Synergy {dim2 + 1} Activation", fontsize=12, fontweight="bold")
        ax.set_title("Synergy Space Trajectory", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_principal_component_analysis(
        self,
        fig: Figure,
        pca_result: Any,
        modes_to_plot: int = 3,
    ) -> None:
        """Plot PCA/Principal Movements analysis results."""
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)

        ax1 = fig.add_subplot(gs[0])

        cum_var = np.cumsum(pca_result.explained_variance_ratio) * 100
        n_comps = len(cum_var)
        x_indices = np.arange(1, n_comps + 1)

        ax1.bar(
            x_indices,
            pca_result.explained_variance_ratio * 100,
            alpha=0.6,
            label="Individual",
        )
        ax1.plot(x_indices, cum_var, "r-o", linewidth=2, label="Cumulative")

        ax1.set_ylabel("Explained Variance (%)", fontweight="bold")
        ax1.set_xlabel("Principal Component", fontweight="bold")
        ax1.set_title("PCA Scree Plot", fontsize=12, fontweight="bold")
        ax1.set_xticks(x_indices)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        ax2 = fig.add_subplot(gs[1])

        times, _ = self.data.get_series("joint_positions")
        scores = pca_result.projected_data
        if len(times) != scores.shape[0]:
            if len(times) > scores.shape[0]:
                times = times[: scores.shape[0]]
            else:
                scores = scores[: len(times)]

        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["quinary"],
        ]

        for i in range(min(modes_to_plot, scores.shape[1])):
            color = colors[i % len(colors)]
            ax2.plot(times, scores[:, i], label=f"PC {i + 1}", linewidth=2, color=color)

        ax2.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Score (Projection)", fontsize=12, fontweight="bold")
        ax2.set_title("Principal Movement Scores", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            "Principal Component Analysis (Principal Movements)",
            fontsize=14,
            fontweight="bold",
        )
