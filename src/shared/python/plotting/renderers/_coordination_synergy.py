from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class CoordinationSynergyMixin(BaseRenderer):
    def plot_muscle_synergies(
        self,
        fig: Figure,
        synergy_result: Any,
    ) -> None:
        """Plot extracted muscle synergies (Weights and Activations)."""
        if fig is None:
            raise ValueError("fig must be provided")
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
        if fig is None:
            raise ValueError("fig must be provided")
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
        if fig is None:
            raise ValueError("fig must be provided")
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
        if fig is None:
            raise ValueError("fig must be provided")
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
