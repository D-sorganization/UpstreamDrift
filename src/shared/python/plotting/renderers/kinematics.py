"""Kinematics plotting renderer."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class KinematicsRenderer(BaseRenderer):
    """Renderer for kinematic plots (angles, velocities, phase)."""

    def plot_joint_angles(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
    ) -> None:
        """Plot joint angles over time.

        Args:
            fig: Matplotlib figure to plot on
            joint_indices: List of joint indices to plot (None = all)
        """
        times, positions = self.data.get_series("joint_positions")

        if len(times) == 0 or len(positions) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        # Ensure positions is a numpy array
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)

        if joint_indices is None:
            joint_indices = list(range(positions.shape[1]))

        for idx in joint_indices:
            if idx < positions.shape[1]:
                label = self.data.get_aligned_label(idx, positions.shape[1])
                ax.plot(times, np.rad2deg(positions[:, idx]), label=label, linewidth=2)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Joint Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_title("Joint Angles vs Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_joint_velocities(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
    ) -> None:
        """Plot joint velocities over time.

        Args:
            fig: Matplotlib figure to plot on
            joint_indices: List of joint indices to plot (None = all)
        """
        times, velocities = self.data.get_series("joint_velocities")

        if len(times) == 0 or len(velocities) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        # Ensure velocities is a numpy array
        if not isinstance(velocities, np.ndarray):
            velocities = np.array(velocities)

        ax = fig.add_subplot(111)

        if joint_indices is None:
            joint_indices = list(range(velocities.shape[1]))

        for idx in joint_indices:
            if idx < velocities.shape[1]:
                label = self.data.get_aligned_label(idx, velocities.shape[1])
                ax.plot(times, np.rad2deg(velocities[:, idx]), label=label, linewidth=2)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Angular Velocity (deg/s)", fontsize=12, fontweight="bold")
        ax.set_title("Joint Velocities vs Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_angle_angle_diagram(
        self,
        fig: Figure,
        joint_idx_1: int,
        joint_idx_2: int,
        title: str | None = None,
        ax: Axes | None = None,
    ) -> None:
        """Plot Angle-Angle diagram (Cyclogram) for two joints."""
        times, positions = self.data.get_series("joint_positions")
        positions = np.asarray(positions)

        if ax is None:
            ax = fig.add_subplot(111)

        if (
            len(times) == 0
            or positions.ndim < 2
            or joint_idx_1 >= positions.shape[1]
            or joint_idx_2 >= positions.shape[1]
        ):
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        theta1 = np.rad2deg(positions[:, joint_idx_1])
        theta2 = np.rad2deg(positions[:, joint_idx_2])

        # Scatter with time color
        sc = ax.scatter(theta1, theta2, c=times, cmap="viridis", s=30, alpha=0.7)
        ax.plot(theta1, theta2, color="gray", alpha=0.3, linewidth=1)

        # Mark Start/End
        ax.scatter(
            theta1[0],
            theta2[0],
            c="green",
            s=100,
            label="Start",
            edgecolor="black",
            zorder=5,
        )
        ax.scatter(
            theta1[-1],
            theta2[-1],
            c="red",
            s=100,
            marker="s",
            label="End",
            edgecolor="black",
            zorder=5,
        )

        name1 = self.data.get_joint_name(joint_idx_1)
        name2 = self.data.get_joint_name(joint_idx_2)

        ax.set_xlabel(f"{name1} Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{name2} Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or f"Coordination: {name1} vs {name2}", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_phase_diagram(self, fig: Figure, joint_idx: int = 0) -> None:
        """Plot phase diagram (angle vs angular velocity) for a joint."""
        times, positions = self.data.get_series("joint_positions")
        _, velocities = self.data.get_series("joint_velocities")

        # Convert to numpy arrays if needed
        positions = np.asarray(positions)
        velocities = np.asarray(velocities)

        if (
            len(times) == 0
            or positions.ndim < 2
            or joint_idx >= positions.shape[1]
            or velocities.ndim < 2
            or joint_idx >= velocities.shape[1]
        ):
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No data available or index out of bounds",
                ha="center",
                va="center",
            )
            return

        ax = fig.add_subplot(111)

        angles = np.rad2deg(positions[:, joint_idx])
        ang_vels = np.rad2deg(velocities[:, joint_idx])

        # Color by time
        sc = ax.scatter(angles, ang_vels, c=times, cmap="viridis", s=30, alpha=0.6)
        ax.plot(angles, ang_vels, alpha=0.2, color="gray", linewidth=1)

        # Mark start
        ax.scatter(
            [angles[0]],
            [ang_vels[0]],
            color="green",
            s=150,
            marker="o",
            edgecolor="black",
            linewidth=2,
            label="Start",
            zorder=5,
        )

        joint_name = self.data.get_joint_name(joint_idx)
        ax.set_xlabel(f"{joint_name} Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{joint_name} Velocity (deg/s)", fontsize=12, fontweight="bold")
        ax.set_title(f"Phase Diagram: {joint_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_3d_phase_space(self, fig: Figure, joint_idx: int = 0) -> None:
        """Plot 3D phase space (Position vs Velocity vs Acceleration)."""
        times, positions = self.data.get_series("joint_positions")
        _, velocities = self.data.get_series("joint_velocities")
        _, accelerations = self.data.get_series("joint_accelerations")

        positions = np.asarray(positions)
        velocities = np.asarray(velocities)
        accelerations = np.asarray(accelerations)

        if (
            len(times) == 0
            or positions.ndim < 2
            or joint_idx >= positions.shape[1]
            or accelerations.ndim < 2
        ):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111, projection="3d")

        pos = np.rad2deg(positions[:, joint_idx])
        vel = np.rad2deg(velocities[:, joint_idx])
        acc = np.rad2deg(accelerations[:, joint_idx])

        # Color by time
        sc = ax.scatter(pos, vel, acc, c=times, cmap="viridis", s=20)
        ax.plot(pos, vel, acc, alpha=0.3, color="gray", linewidth=1)

        # Mark start
        ax.scatter(
            [pos[0]],
            [vel[0]],
            [acc[0]],
            color="green",
            s=100,
            marker="o",
            label="Start",
        )

        joint_name = self.data.get_joint_name(joint_idx)
        ax.set_title(f"3D Phase Space: {joint_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Position (deg)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Velocity (deg/s)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Acceleration (deg/s²)", fontsize=10, fontweight="bold")
        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)
        fig.tight_layout()

    def plot_poincare_map_3d(
        self,
        fig: Figure,
        dimensions: list[tuple[str, int]],
        section_condition: tuple[str, int, float] = ("velocity", 0, 0.0),
        direction: str = "both",
        title: str | None = None,
    ) -> None:
        """Plot 3D Poincaré Map (Poincaré Section)."""
        if len(dimensions) != 3:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "Must specify exactly 3 dimensions", ha="center", va="center"
            )
            return

        cond_type, cond_idx, cond_val = section_condition
        cond_data = self._get_poincare_data(cond_type, cond_idx)
        times, _ = self.data.get_series("joint_positions")

        if cond_data is None or len(cond_data) < 2:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Condition data {cond_type}[{cond_idx}] unavailable",
                ha="center",
                va="center",
            )
            return

        crossings = self._find_section_crossings(cond_data, cond_val, direction)

        if not crossings:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No section crossings found", ha="center", va="center")
            return

        points_arr, point_times = self._interpolate_crossings(
            crossings, cond_data, cond_val, times, dimensions
        )

        self._render_poincare_3d(
            fig,
            points_arr,
            point_times,
            dimensions,
            cond_type,
            cond_idx,
            cond_val,
            title,
        )

    def _get_poincare_data(self, dtype: str, idx: int) -> np.ndarray | None:
        """Retrieve a single data column for Poincare section computation."""
        series_map = {
            "position": "joint_positions",
            "velocity": "joint_velocities",
            "acceleration": "joint_accelerations",
            "torque": "joint_torques",
        }
        key = series_map.get(dtype)
        if key is None:
            return None
        _, d = self.data.get_series(key)
        d = np.asarray(d)
        if d.ndim > 1 and idx < d.shape[1]:
            return d[:, idx]
        return None

    @staticmethod
    def _find_section_crossings(
        cond_data: np.ndarray, cond_val: float, direction: str
    ) -> list[int]:
        """Find zero-crossing indices in (cond_data - cond_val)."""
        diff = cond_data - cond_val
        crossings = []
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] <= 0:
                if diff[i] < diff[i + 1] and direction in ["positive", "both"]:
                    crossings.append(i)
                elif diff[i] > diff[i + 1] and direction in ["negative", "both"]:
                    crossings.append(i)
        return crossings

    def _interpolate_crossings(
        self,
        crossings: list[int],
        cond_data: np.ndarray,
        cond_val: float,
        times: np.ndarray,
        dimensions: list[tuple[str, int]],
    ) -> tuple[np.ndarray, list[float]]:
        """Interpolate crossing points in the requested dimensions."""
        diff = cond_data - cond_val
        points = []
        point_times: list[float] = []

        for i in crossings:
            denom = diff[i + 1] - diff[i]
            alpha = 0.5 if abs(denom) < 1e-9 else -diff[i] / denom

            t_cross = times[i] + alpha * (times[i + 1] - times[i])
            point_times.append(t_cross)

            pt_coords = []
            for dim_type, dim_idx in dimensions:
                data = self._get_poincare_data(dim_type, dim_idx)
                if data is None:
                    pt_coords.append(0.0)
                else:
                    val = data[i] + alpha * (data[i + 1] - data[i])
                    if dim_type in ["position", "velocity", "acceleration"]:
                        val = np.rad2deg(val)
                    pt_coords.append(val)
            points.append(pt_coords)

        return np.array(points), point_times

    def _render_poincare_3d(
        self,
        fig: Figure,
        points_arr: np.ndarray,
        point_times: list[float],
        dimensions: list[tuple[str, int]],
        cond_type: str,
        cond_idx: int,
        cond_val: float,
        title: str | None,
    ) -> None:
        """Render the 3D scatter plot for the Poincare section."""
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            points_arr[:, 0],
            points_arr[:, 1],
            points_arr[:, 2],
            c=point_times,
            cmap="viridis",
            s=50,
            edgecolors="k",
        )

        labels = []
        for dt, di in dimensions:
            name = self.data.get_joint_name(di)
            unit = (
                "deg"
                if dt == "position"
                else "deg/s" if dt == "velocity" else "Nm" if dt == "torque" else ""
            )
            labels.append(f"{name} {dt[:3]} ({unit})")

        ax.set_xlabel(labels[0], fontsize=9, fontweight="bold")
        ax.set_ylabel(labels[1], fontsize=9, fontweight="bold")
        ax.set_zlabel(labels[2], fontsize=9, fontweight="bold")

        cond_name = self.data.get_joint_name(cond_idx)
        if title is None:
            title = f"Poincaré Map\nSection: {cond_name} {cond_type} = {cond_val}"

        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)
        fig.tight_layout()

    def plot_phase_space_reconstruction(
        self,
        fig: Figure,
        joint_idx: int = 0,
        delay: int = 10,
        embedding_dim: int = 3,
        signal_type: str = "position",
    ) -> None:
        """Plot Phase Space Reconstruction using Time-Delay Embedding."""
        if signal_type == "position":
            times, data_full = self.data.get_series("joint_positions")
            data_full = np.rad2deg(np.asarray(data_full))
        elif signal_type == "velocity":
            times, data_full = self.data.get_series("joint_velocities")
            data_full = np.rad2deg(np.asarray(data_full))
        else:
            times, data_full = self.data.get_series("joint_torques")
            data_full = np.asarray(data_full)

        if len(times) == 0 or data_full.ndim < 2 or joint_idx >= data_full.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        x = data_full[:, joint_idx]
        N = len(x)

        if N < delay * (embedding_dim - 1) + 1:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Time series too short for embedding",
                ha="center",
                va="center",
            )
            return

        valid_len = N - delay * (embedding_dim - 1)

        vectors = np.zeros((valid_len, embedding_dim))
        for d in range(embedding_dim):
            start = d * delay
            end = start + valid_len
            vectors[:, d] = x[start:end]

        plot_times = times[:valid_len]

        if embedding_dim == 3:
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                vectors[:, 0],
                vectors[:, 1],
                vectors[:, 2],
                c=plot_times,
                cmap="magma",
                s=10,
                alpha=0.6,
            )
            ax.plot(
                vectors[:, 0],
                vectors[:, 1],
                vectors[:, 2],
                color="gray",
                alpha=0.2,
                linewidth=0.5,
            )

            ax.set_xlabel("x(t)", fontsize=10)
            ax.set_ylabel(f"x(t+{delay})", fontsize=10)
            ax.set_zlabel(f"x(t+{2 * delay})", fontsize=10)
        else:
            ax = fig.add_subplot(111)
            sc = ax.scatter(
                vectors[:, 0],
                vectors[:, 1],
                c=plot_times,
                cmap="magma",
                s=10,
                alpha=0.6,
            )
            ax.plot(
                vectors[:, 0], vectors[:, 1], color="gray", alpha=0.2, linewidth=0.5
            )

            ax.set_xlabel("x(t)", fontsize=10)
            ax.set_ylabel(f"x(t+{delay})", fontsize=10)

        joint_name = self.data.get_joint_name(joint_idx)
        ax.set_title(
            f"Reconstructed Phase Space: {joint_name}\n(Lag={delay}, Dim={embedding_dim})",
            fontsize=12,
            fontweight="bold",
        )
        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)
        fig.tight_layout()

    def plot_phase_space_density(
        self,
        fig: Figure,
        joint_idx: int = 0,
        bins: int = 50,
    ) -> None:
        """Plot 2D Phase Space Density (Histogram)."""
        times, positions = self.data.get_series("joint_positions")
        _, velocities = self.data.get_series("joint_velocities")

        positions = np.asarray(positions)
        velocities = np.asarray(velocities)

        if (
            len(times) == 0
            or positions.ndim < 2
            or joint_idx >= positions.shape[1]
            or velocities.ndim < 2
            or joint_idx >= velocities.shape[1]
        ):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        pos = np.rad2deg(positions[:, joint_idx])
        vel = np.rad2deg(velocities[:, joint_idx])

        # 2D Histogram
        h = ax.hist2d(
            pos,
            vel,
            bins=bins,
            cmap="inferno",
            cmin=1,  # Don't plot zero bins
        )

        joint_name = self.data.get_joint_name(joint_idx)
        ax.set_xlabel(f"{joint_name} Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{joint_name} Velocity (deg/s)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Phase Space Density: {joint_name}", fontsize=14, fontweight="bold"
        )

        fig.colorbar(h[3], ax=ax, label="Count")
        fig.tight_layout()
