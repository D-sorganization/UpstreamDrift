"""Plotting module for comparative swing analysis.

Visualizes the differences between two swings using overlays and difference plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.shared.python.comparative_analysis import ComparativeSwingAnalyzer


class ComparativePlotter:
    """Generates comparison plots for two swings."""

    def __init__(self, analyzer: ComparativeSwingAnalyzer) -> None:
        """Initialize with an analyzer instance.

        Args:
            analyzer: ComparativeSwingAnalyzer containing the two swings
        """
        self.analyzer = analyzer
        self.colors = {
            "a": "#1f77b4",  # Blue
            "b": "#ff7f0e",  # Orange
            "diff": "#d62728",  # Red
            "grid": "#cccccc",
        }

    def plot_comparison(
        self,
        fig: Figure | Any,
        field_name: str,
        joint_idx: int | None = None,
        title: str | None = None,
        ylabel: str = "Value",
    ) -> None:
        """Plot overlay of two signals and their difference.

        Args:
            fig: Matplotlib figure
            field_name: Data field to compare
            joint_idx: Joint index (optional)
            title: Plot title
            ylabel: Y-axis label
        """
        aligned = self.analyzer.align_signals(field_name, joint_idx=joint_idx)

        if aligned is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center")
            return

        # Use GridSpec for main plot and difference plot
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

        # Main overlay plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(
            aligned.times * 100,
            aligned.signal_a,
            color=self.colors["a"],
            label=self.analyzer.name_a,
            linewidth=2,
        )
        ax1.plot(
            aligned.times * 100,
            aligned.signal_b,
            color=self.colors["b"],
            label=self.analyzer.name_b,
            linewidth=2,
            linestyle="--",
        )

        ax1.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax1.set_title(
            title or f"Comparison: {field_name}", fontsize=12, fontweight="bold"
        )
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)

        # Difference plot
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(
            aligned.times * 100,
            aligned.error_curve,
            color=self.colors["diff"],
            linewidth=1.5,
        )
        ax2.fill_between(
            aligned.times * 100,
            0,
            aligned.error_curve,
            color=self.colors["diff"],
            alpha=0.2,
        )

        ax2.set_xlabel("Swing Progress (%)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Diff (A-B)", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 100)
        ax2.axhline(0, color="k", alpha=0.3, linestyle="-")

        # Add correlation stats
        stats_text = (
            f"Correlation: {aligned.correlation:.3f}\nRMS Diff: {aligned.rms_error:.3f}"
        )
        ax1.text(
            0.02,
            0.95,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        # Only use tight_layout if we created the subplots directly on the figure (not passed axis)
        # Here we did create subplots via GridSpec on figure, so it's safeish, but usually handled by caller.
        # fig.tight_layout()

    def plot_phase_comparison(
        self,
        fig: Figure | None = None,
        joint_idx: int = 0,
        joint_name: str = "Joint",
        ax: Axes | None = None,
    ) -> None:
        """Overlay phase diagrams (Angle vs Velocity).

        Args:
            fig: Matplotlib figure (optional if ax provided)
            joint_idx: Joint index
            joint_name: Name of joint for labels
            ax: Matplotlib Axes to plot on (optional)
        """
        if ax is None:
            if fig is None:
                raise ValueError("Must provide either fig or ax")
            ax = fig.add_subplot(111)

        # Get aligned data for position and velocity
        pos_aligned = self.analyzer.align_signals(
            "joint_positions", joint_idx=joint_idx
        )
        vel_aligned = self.analyzer.align_signals(
            "joint_velocities", joint_idx=joint_idx
        )

        if pos_aligned is None or vel_aligned is None:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center")
            return

        # Plot A
        ax.plot(
            np.rad2deg(pos_aligned.signal_a),
            np.rad2deg(vel_aligned.signal_a),
            color=self.colors["a"],
            label=self.analyzer.name_a,
            linewidth=2,
        )

        # Plot B
        ax.plot(
            np.rad2deg(pos_aligned.signal_b),
            np.rad2deg(vel_aligned.signal_b),
            color=self.colors["b"],
            label=self.analyzer.name_b,
            linewidth=2,
            linestyle="--",
        )

        # Mark start points
        ax.plot(
            np.rad2deg(pos_aligned.signal_a[0]),
            np.rad2deg(vel_aligned.signal_a[0]),
            "o",
            color=self.colors["a"],
        )
        ax.plot(
            np.rad2deg(pos_aligned.signal_b[0]),
            np.rad2deg(vel_aligned.signal_b[0]),
            "o",
            color=self.colors["b"],
        )

        ax.set_xlabel(f"{joint_name} Angle (deg)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{joint_name} Velocity (deg/s)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Phase Diagram Comparison: {joint_name}", fontsize=12, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    def plot_coordination_comparison(
        self,
        fig: Figure,
        joint_idx_1: int,
        joint_idx_2: int,
        title: str = "Coordination Comparison",
    ) -> None:
        """Compare coordination patterns (Phase-Phase) of two swings.

        Note: Ideally compares Continuous Relative Phase (CRP) if available,
        but typically we plot Angle-Angle overlays or calculate CRP on fly.
        Here we'll overlay Angle-Angle diagrams first as it's the most standard.

        Args:
            fig: Matplotlib figure
            joint_idx_1: Joint 1 index (X axis)
            joint_idx_2: Joint 2 index (Y axis)
            title: Title
        """
        # Align both joints
        pos1_aligned = self.analyzer.align_signals(
            "joint_positions", joint_idx=joint_idx_1
        )
        pos2_aligned = self.analyzer.align_signals(
            "joint_positions", joint_idx=joint_idx_2
        )

        if pos1_aligned is None or pos2_aligned is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        # Plot Swing A (Angle 1 vs Angle 2)
        ax.plot(
            np.rad2deg(pos1_aligned.signal_a),
            np.rad2deg(pos2_aligned.signal_a),
            color=self.colors["a"],
            label=self.analyzer.name_a,
            linewidth=2,
        )

        # Plot Swing B
        ax.plot(
            np.rad2deg(pos1_aligned.signal_b),
            np.rad2deg(pos2_aligned.signal_b),
            color=self.colors["b"],
            label=self.analyzer.name_b,
            linewidth=2,
            linestyle="--",
        )

        # Mark Start
        ax.scatter(
            np.rad2deg(pos1_aligned.signal_a[0]),
            np.rad2deg(pos2_aligned.signal_a[0]),
            color=self.colors["a"],
            marker="o",
            s=50,
        )
        ax.scatter(
            np.rad2deg(pos1_aligned.signal_b[0]),
            np.rad2deg(pos2_aligned.signal_b[0]),
            color=self.colors["b"],
            marker="o",
            s=50,
        )

        ax.set_xlabel(
            f"Joint {joint_idx_1} Angle (deg)", fontsize=11, fontweight="bold"
        )
        ax.set_ylabel(
            f"Joint {joint_idx_2} Angle (deg)", fontsize=11, fontweight="bold"
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")  # Preserve aspect ratio for phase portraits

    def plot_3d_trajectory_comparison(
        self,
        fig: Figure,
        title: str = "Trajectory Comparison",
    ) -> None:
        """Plot 3D trajectory comparison (e.g., Club Head Path).

        Args:
            fig: Matplotlib figure
            title: Plot title
        """
        # Align club head positions
        # Club head position is (N, 3), so we treat each dimension as a signal

        # This assumes align_signals handles multi-dim or we loop.
        # But align_signals is for 1D. We need to align manually or use underlying method.
        # Actually, if we use time warping, the time mapping should be same for all dimensions.
        # The current ComparativeSwingAnalyzer aligns based on a reference signal (usually club speed)
        # and applies warping indices.

        # For simplicity in this implementation, we will assume pre-aligned or align each dim separately
        # (which might distort geometry) OR better: rely on time normalization (0-100%).

        # Since RecorderInterface doesn't support 'get_time_series' with warping directly,
        # we rely on the analyzer to give us signals.

        # Let's extract raw signals from analyzer's recorders and assume simple time scaling
        # if the analyzer doesn't expose 3D alignment.
        # Actually, ComparativeSwingAnalyzer aligns 1D signals.

        # We will retrieve the raw data and just plot them in their original space,
        # maybe normalizing time for color.

        rec_a = self.analyzer.recorder_a
        rec_b = self.analyzer.recorder_b

        t_a, pos_a = rec_a.get_time_series("club_head_position")
        t_b, pos_b = rec_b.get_time_series("club_head_position")

        pos_a = np.asarray(pos_a)
        pos_b = np.asarray(pos_b)

        if len(t_a) == 0 or len(t_b) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center")
            return

        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            pos_a[:, 0],
            pos_a[:, 1],
            pos_a[:, 2],
            label=self.analyzer.name_a,
            color=self.colors["a"],
        )
        ax.plot(
            pos_b[:, 0],
            pos_b[:, 1],
            pos_b[:, 2],
            label=self.analyzer.name_b,
            color=self.colors["b"],
            linestyle="--",
        )

        # Start/End markers
        ax.scatter(
            [pos_a[0, 0]],
            [pos_a[0, 1]],
            [pos_a[0, 2]],
            color=self.colors["a"],
            marker="o",
        )
        ax.scatter(
            [pos_b[0, 0]],
            [pos_b[0, 1]],
            [pos_b[0, 2]],
            color=self.colors["b"],
            marker="o",
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")  # type: ignore
        ax.legend()
        fig.tight_layout()

    def plot_dashboard(self, fig: Figure) -> None:
        """Create a summary dashboard for the comparison.

        Args:
            fig: Matplotlib figure
        """
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Club Head Speed Comparison
        subfig1 = fig.add_subfigure(gs[0, 0])
        self.plot_comparison(
            subfig1,
            "club_head_speed",
            title="Club Head Speed Comparison",
            ylabel="Speed (m/s)",
        )

        # 2. Kinetic Energy Comparison
        subfig2 = fig.add_subfigure(gs[0, 1])
        self.plot_comparison(
            subfig2,
            "kinetic_energy",
            title="Kinetic Energy Comparison",
            ylabel="Energy (J)",
        )

        # 3. Phase Diagram (Joint 0 - usually hips/pelvis)
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_phase_comparison(ax=ax3, joint_idx=0, joint_name="Joint 0")

        # 4. Metric Difference Bar Chart
        ax4 = fig.add_subplot(gs[1, 1])
        report = self.analyzer.generate_comparison_report()
        metrics = report["metrics"]

        if metrics:
            names = [m.name for m in metrics]
            diffs = [m.percent_diff for m in metrics]

            y_pos = np.arange(len(names))
            ax4.barh(
                y_pos,
                diffs,
                align="center",
                color=[self.colors["a"] if x > 0 else self.colors["b"] for x in diffs],
            )
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(names)
            ax4.invert_yaxis()
            ax4.set_xlabel("% Difference (A - B)", fontsize=10)
            ax4.set_title("Relative Differences", fontsize=10, fontweight="bold")
            ax4.grid(True, axis="x", alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No metrics available", ha="center", va="center")

        fig.suptitle(
            f"Comparison: {self.analyzer.name_a} vs {self.analyzer.name_b}",
            fontsize=14,
            fontweight="bold",
        )

    def plot_dtw_alignment(
        self,
        fig: Figure,
        field_name: str,
        joint_idx: int | None = None,
        radius: int = 10,
    ) -> None:
        """Plot Dynamic Time Warping alignment path.

        Visualizes the optimal warping path between the two signals.

        Args:
            fig: Matplotlib figure
            field_name: Data field name
            joint_idx: Joint index (optional)
            radius: Sakoe-Chiba radius used for calculation
        """
        dist, path = self.analyzer.compute_dtw_distance(field_name, joint_idx, radius)

        if not path:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "DTW calculation failed", ha="center", va="center")
            return

        path_arr = np.array(path)

        ax = fig.add_subplot(111)
        ax.plot(
            path_arr[:, 0],
            path_arr[:, 1],
            color="purple",
            linewidth=2,
            label="Optimal Path",
        )

        # Plot diagonal (perfect alignment)
        max_idx = max(path_arr[:, 0].max(), path_arr[:, 1].max())
        ax.plot([0, max_idx], [0, max_idx], "k--", alpha=0.3, label="Linear Match")

        ax.set_xlabel(f"{self.analyzer.name_a} Index", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{self.analyzer.name_b} Index", fontsize=10, fontweight="bold")
        ax.set_title(
            f"DTW Alignment Path: {field_name}\nDistance: {dist:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        fig.tight_layout()

    def plot_bland_altman(
        self,
        fig: Figure,
        field_name: str,
        joint_idx: int | None = None,
        title: str | None = None,
    ) -> None:
        """Plot Bland-Altman diagram (Difference vs Mean).

        Visualizes agreement between two measurement methods.

        Args:
            fig: Matplotlib figure
            field_name: Data field name
            joint_idx: Optional joint index
            title: Optional title
        """
        aligned = self.analyzer.align_signals(field_name, joint_idx=joint_idx)

        if aligned is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center")
            return

        means = (aligned.signal_a + aligned.signal_b) / 2.0
        diffs = aligned.signal_a - aligned.signal_b
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        ax = fig.add_subplot(111)

        ax.scatter(means, diffs, c=self.colors["a"], alpha=0.5, s=20)

        # Draw mean and Limits of Agreement (LoA = mean +/- 1.96*SD)
        ax.axhline(mean_diff, color="black", linestyle="-", label="Mean Diff")
        ax.axhline(
            mean_diff + 1.96 * std_diff,
            color="red",
            linestyle="--",
            label="Upper LoA (+1.96SD)",
        )
        ax.axhline(
            mean_diff - 1.96 * std_diff,
            color="red",
            linestyle="--",
            label="Lower LoA (-1.96SD)",
        )

        # Annotate values
        ax.text(
            max(means),
            mean_diff + 1.96 * std_diff,
            f"{mean_diff + 1.96 * std_diff:.2f}",
            va="bottom",
            ha="right",
            fontsize=9,
        )
        ax.text(
            max(means),
            mean_diff - 1.96 * std_diff,
            f"{mean_diff - 1.96 * std_diff:.2f}",
            va="top",
            ha="right",
            fontsize=9,
        )
        ax.text(
            max(means),
            mean_diff,
            f"{mean_diff:.2f}",
            va="bottom",
            ha="right",
            fontsize=9,
        )

        ax.set_xlabel("Mean of two measures", fontsize=12, fontweight="bold")
        ax.set_ylabel("Difference (A - B)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or f"Bland-Altman: {field_name}", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
