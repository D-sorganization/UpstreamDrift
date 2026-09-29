from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class CoordinationAlignmentMixin(BaseRenderer):
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
        if fig is None:
            raise ValueError("fig must be provided")
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
        if fig is None:
            raise ValueError("fig must be provided")
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
        if fig is None:
            raise ValueError("fig must be provided")
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
        if fig is None:
            raise ValueError("fig must be provided")
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
