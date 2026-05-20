from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class CoordinationSequenceMixin(BaseRenderer):
    def plot_lag_matrix(
        self,
        fig: Figure,
        data_type: str = "velocity",
        max_lag: float = 0.5,
    ) -> None:
        """Plot time lag matrix between joints."""
        if fig is None:
            raise ValueError("fig must be provided")
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
        if fig is None:
            raise ValueError("fig must be provided")
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
                vel_norm = vel / max_vel if max_vel > 0 else vel

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
        if fig is None:
            raise ValueError("fig must be provided")
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
        if fig is None:
            raise ValueError("fig must be provided")
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
