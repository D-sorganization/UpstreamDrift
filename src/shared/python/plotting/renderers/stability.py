"""Stability plotting renderer."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class StabilityRenderer(BaseRenderer):
    """Renderer for stability and balance plots."""

    def plot_stability_metrics(self, fig: Figure) -> None:
        """Plot stability metrics (CoM-CoP distance and Inclination Angle)."""
        try:
            times_cop, cop = self.data.get_series("cop_position")
            times_com, com = self.data.get_series("com_position")
        except (AttributeError, KeyError):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Stability data missing", ha="center", va="center")
            return

        cop = np.asarray(cop)
        com = np.asarray(com)

        if len(times_cop) == 0 or len(times_com) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No stability data", ha="center", va="center")
            return

        cop_xy = cop[:, :2]
        com_xy = com[:, :2]
        diff = cop_xy - com_xy
        dist = np.hypot(diff[:, 0], diff[:, 1])

        cop_z = np.zeros(len(cop)) if cop.shape[1] == 2 else cop[:, 2]

        vec_temp = com - np.column_stack((cop_xy, cop_z))
        vec = vec_temp
        vec_norm = np.sqrt(np.sum(vec**2, axis=1))
        vec_norm[vec_norm < 1e-6] = 1.0

        cos_theta = vec[:, 2] / vec_norm
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles_deg = np.rad2deg(np.arccos(cos_theta))

        ax1 = fig.add_subplot(111)

        line1 = ax1.plot(
            times_cop,
            dist,
            color=self.colors["primary"],
            linewidth=2,
            label="CoM-CoP Dist (m)",
        )
        ax1.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax1.set_ylabel(
            "Distance (m)", fontsize=12, fontweight="bold", color=self.colors["primary"]
        )
        ax1.tick_params(axis="y", labelcolor=self.colors["primary"])

        ax2 = ax1.twinx()
        line2 = ax2.plot(
            times_cop,
            angles_deg,
            color=self.colors["quaternary"],
            linewidth=2,
            linestyle="--",
            label="Inclination (deg)",
        )
        ax2.set_ylabel(
            "Inclination Angle (deg)",
            fontsize=12,
            fontweight="bold",
            color=self.colors["quaternary"],
        )
        ax2.tick_params(axis="y", labelcolor=self.colors["quaternary"])

        lns = line1 + line2
        labs = [str(ln.get_label()) for ln in lns]
        ax1.legend(lns, labs, loc="best")

        ax1.set_title("Postural Stability Metrics", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        fig.tight_layout()

    def plot_cop_trajectory(self, fig: Figure) -> None:
        """Plot Center of Pressure trajectory (top-down view)."""
        times, cop_data = self.data.get_series("cop_position")
        cop_data = np.asarray(cop_data)

        if len(times) == 0 or cop_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No CoP Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        x = cop_data[:, 0]
        y = cop_data[:, 1]

        sc = ax.scatter(x, y, c=times, cmap="viridis", s=30, zorder=2)
        ax.plot(x, y, color="gray", alpha=0.4, zorder=1)

        ax.scatter(x[0], y[0], c="green", s=100, label="Start", zorder=3)
        ax.scatter(x[-1], y[-1], c="red", s=100, marker="s", label="End", zorder=3)

        ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
        ax.set_title("Center of Pressure Trajectory", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axis("equal")

        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_cop_vector_field(self, fig: Figure, skip_steps: int = 5) -> None:
        """Plot CoP velocity vector field."""
        times, cop_data = self.data.get_series("cop_position")
        cop_data = np.asarray(cop_data)

        if len(times) == 0 or cop_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No CoP Data", ha="center", va="center")
            return

        dt = np.mean(np.diff(times)) if len(times) > 1 else 1.0
        vel = np.gradient(cop_data, dt, axis=0)

        ax = fig.add_subplot(111)

        x = cop_data[::skip_steps, 0]
        y = cop_data[::skip_steps, 1]
        u = vel[::skip_steps, 0]
        v = vel[::skip_steps, 1]
        t = times[::skip_steps]

        q = ax.quiver(x, y, u, v, t, cmap="viridis", scale_units="xy", angles="xy")

        ax.plot(cop_data[:, 0], cop_data[:, 1], "k-", alpha=0.2)

        ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Center of Pressure Velocity Field", fontsize=14, fontweight="bold"
        )
        ax.axis("equal")
        fig.colorbar(q, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_grf_butterfly_diagram(
        self,
        fig: Figure,
        skip_steps: int = 5,
        scale: float = 0.001,
    ) -> None:
        """Plot Ground Reaction Force 'Butterfly Diagram'."""
        try:
            times, cop_data = self.data.get_series("cop_position")
            _, grf_data = self.data.get_series("ground_forces")
        except (AttributeError, KeyError):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "GRF/CoP Data unavailable", ha="center", va="center")
            return

        cop_data = np.asarray(cop_data)
        grf_data = np.asarray(grf_data)

        if len(times) == 0 or cop_data.size == 0 or grf_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No GRF Data", ha="center", va="center")
            return

        if grf_data.shape[1] >= 3:
            fx = grf_data[:, 0]
            fy = grf_data[:, 1]
            fz = grf_data[:, 2]
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Invalid GRF dimensions", ha="center", va="center")
            return

        ax = fig.add_subplot(111, projection="3d")

        cx = cop_data[:, 0]
        cy = cop_data[:, 1]
        cz = cop_data[:, 2] if cop_data.shape[1] > 2 else np.zeros_like(cx)

        ax.plot(cx, cy, cz, color="black", linewidth=2, label="CoP Path")

        indices = range(0, len(times), skip_steps)
        for i in indices:
            ox, oy, oz = cx[i], cy[i], cz[i]
            vx, vy, vz = fx[i], fy[i], fz[i]

            ax.plot(
                [ox, ox + vx * scale],
                [oy, oy + vy * scale],
                [oz, oz + vz * scale],
                color=self.colors["secondary"],
                alpha=0.6,
                linewidth=1,
            )

        ax.set_xlabel("X (m)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Y (m)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Force (scaled)", fontsize=10, fontweight="bold")
        ax.set_title("GRF Butterfly Diagram", fontsize=14, fontweight="bold")

        all_x = np.concatenate([cx, cx + fx * scale])
        all_y = np.concatenate([cy, cy + fy * scale])
        all_z = np.concatenate([cz, cz + fz * scale])

        ax.set_xlim(np.min(all_x), np.max(all_x))
        ax.set_ylim(np.min(all_y), np.max(all_y))
        ax.set_zlim(np.min(all_z), np.max(all_z))

        fig.tight_layout()

    def plot_3d_vector_field(
        self,
        fig: Figure,
        vector_name: str,
        position_name: str,
        skip_steps: int = 5,
        scale: float = 0.1,
    ) -> None:
        """Plot 3D vector field along a trajectory."""
        try:
            times, vectors = self.data.get_series(vector_name)
            _, positions = self.data.get_series(position_name)
        except (AttributeError, KeyError):
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Data missing: {vector_name}/{position_name}",
                ha="center",
                va="center",
            )
            return

        vectors = np.asarray(vectors)
        positions = np.asarray(positions)

        if len(times) == 0 or vectors.shape[1] < 3 or positions.shape[1] < 3:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "Invalid dimensions or empty data", ha="center", va="center"
            )
            return

        ax = fig.add_subplot(111, projection="3d")

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        ax.plot(x, y, z, color="k", alpha=0.3, linewidth=1, label=f"{position_name}")

        indices = range(0, len(times), skip_steps)
        for i in indices:
            u, v, w = vectors[i, 0], vectors[i, 1], vectors[i, 2]
            px, py, pz = x[i], y[i], z[i]

            ax.plot(
                [px, px + u * scale],
                [py, py + v * scale],
                [pz, pz + w * scale],
                color=self.colors["secondary"],
                linewidth=1.5,
                alpha=0.6,
            )

        ax.set_title(f"3D Vector Field: {vector_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("X", fontweight="bold")
        ax.set_ylabel("Y", fontweight="bold")
        ax.set_zlabel("Z", fontweight="bold")
        fig.tight_layout()

    def plot_local_stability(
        self,
        fig: Figure,
        joint_idx: int = 0,
        embedding_dim: int = 3,
        tau: int = 5,
    ) -> None:
        """Plot local divergence rate (Local Stability) over time."""
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

        if len(times) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        analyzer = StatisticalAnalyzer(
            times=np.asarray(times),
            joint_positions=np.asarray(positions),
            joint_velocities=np.asarray(velocities),
            joint_torques=np.zeros_like(positions),
        )

        try:
            ld_times, ld_rates = analyzer.compute_local_divergence_rate(
                joint_idx=joint_idx,
                tau=tau,
                dim=embedding_dim,
                window=tau * 2,
                data_type="velocity",
            )
        except AttributeError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Method not implemented", ha="center", va="center")
            return

        if len(ld_times) == 0:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Insufficient data for stability analysis",
                ha="center",
                va="center",
            )
            return

        ax = fig.add_subplot(111)
        ax.plot(ld_times, ld_rates, color=self.colors["quaternary"], linewidth=2)

        ax.fill_between(
            ld_times,
            0,
            ld_rates,
            where=(ld_rates > 0),  # type: ignore[arg-type]
            alpha=0.2,
            color="red",
            label="Unstable",
        )
        ax.fill_between(
            ld_times,
            0,
            ld_rates,
            where=(ld_rates <= 0),  # type: ignore[arg-type]
            alpha=0.2,
            color="green",
            label="Stable",
        )

        name = self.data.get_joint_name(joint_idx)
        ax.set_title(
            f"Local Stability (Divergence Rate): {name}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Divergence Rate (1/s)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

    def plot_stability_diagram(self, fig: Figure) -> None:
        """Plot Stability Diagram (CoM vs CoP on Ground Plane)."""
        try:
            times, cop_data = self.data.get_series("cop_position")
            _, com_data = self.data.get_series("com_position")
        except (AttributeError, KeyError):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Stability Data unavailable", ha="center", va="center")
            return

        cop_data = np.asarray(cop_data)
        com_data = np.asarray(com_data)

        if len(times) == 0 or cop_data.size == 0 or com_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Stability Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        ax.plot(
            cop_data[:, 0],
            cop_data[:, 1],
            color=self.colors["secondary"],
            linewidth=2,
            label="CoP",
        )

        ax.plot(
            com_data[:, 0],
            com_data[:, 1],
            color=self.colors["primary"],
            linewidth=2,
            linestyle="--",
            label="CoM (Proj)",
        )

        indices = range(0, len(times), len(times) // 10 if len(times) > 10 else 1)
        for i in indices:
            ax.plot(
                [cop_data[i, 0], com_data[i, 0]],
                [cop_data[i, 1], com_data[i, 1]],
                color="gray",
                alpha=0.3,
                linewidth=1,
            )

        ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
        ax.set_title("Stability Diagram (CoM vs CoP)", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axis("equal")

        fig.tight_layout()
