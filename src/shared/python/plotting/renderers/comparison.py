"""Comparison plotting renderer."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class ComparisonRenderer(BaseRenderer):
    """Renderer for comparative plots (counterfactuals, etc.)."""

    def plot_counterfactual_comparison(
        self, fig: Figure, cf_name: str, metric_idx: int = 0
    ) -> None:
        """Plot counterfactual data against actual data."""
        if cf_name == "dual":
            self._plot_counterfactual_dual(fig, metric_idx)
            return

        times_actual, actual_data = self.data.get_series("joint_positions")
        actual = np.asarray(actual_data)

        try:
            times_cf, cf_data_raw = self.data.get_counterfactual_series(cf_name)
            cf_data = np.asarray(cf_data_raw)
        except (AttributeError, KeyError):
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"No counterfactual data for {cf_name}",
                ha="center",
                va="center",
            )
            return

        if len(times_actual) == 0 or len(times_cf) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        if actual.ndim > 1 and metric_idx < actual.shape[1]:
            ax.plot(
                times_actual,
                np.rad2deg(actual[:, metric_idx]),
                label="Actual",
                linewidth=2,
                color="black",
            )

        if cf_data.ndim > 1 and metric_idx < cf_data.shape[1]:
            ax.plot(
                times_cf,
                np.rad2deg(cf_data[:, metric_idx]),
                label=cf_name.upper(),
                linewidth=2,
                linestyle="--",
                color=self.colors["primary"],
            )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Counterfactual Analysis: {cf_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def _plot_counterfactual_dual(self, fig: Figure, joint_idx: int) -> None:
        """Helper to plot ZTCF (Accel) and ZVCF (Torque) on dual axes."""
        try:
            times_z, ztcf = self.data.get_counterfactual_series("ztcf_accel")
            times_v, zvcf = self.data.get_counterfactual_series("zvcf_torque")
        except (AttributeError, KeyError):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Counterfactual data missing", ha="center", va="center")
            return

        if len(times_z) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No CF data", ha="center", va="center")
            return

        ax1 = fig.add_subplot(111)

        if joint_idx < ztcf.shape[1]:
            line1 = ax1.plot(
                times_z,
                ztcf[:, joint_idx],
                color=self.colors["primary"],
                label="ZTCF Accel (Zero Torque)",
            )
        else:
            return

        ax1.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Acceleration (rad/s²)", color=self.colors["primary"])
        ax1.tick_params(axis="y", labelcolor=self.colors["primary"])

        if len(times_v) > 0 and joint_idx < zvcf.shape[1]:
            ax2 = ax1.twinx()
            line2 = ax2.plot(
                times_v,
                zvcf[:, joint_idx],
                color=self.colors["quaternary"],
                linestyle="--",
                label="ZVCF Torque (Zero Velocity)",
            )
            ax2.set_ylabel("Torque (Nm)", color=self.colors["quaternary"])
            ax2.tick_params(axis="y", labelcolor=self.colors["quaternary"])

            lns = line1 + line2
            labs = [str(line.get_label()) for line in lns]
            ax1.legend(lns, labs, loc="upper left")
        else:
            ax1.legend(loc="best")

        joint_name = self.data.get_joint_name(joint_idx)
        ax1.set_title(
            f"Counterfactuals (ZTCF vs ZVCF): {joint_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
