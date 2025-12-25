"""Plotting module for comparative swing analysis.

Visualizes the differences between two swings using overlays and difference plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from shared.python.comparative_analysis import ComparativeSwingAnalyzer


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
