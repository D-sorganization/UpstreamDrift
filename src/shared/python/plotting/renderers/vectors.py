"""Force and torque vector overlay renderer.

Renders quiver-style vector overlays on top of 3-D trajectory plots,
enabling visual comparison of forces, torques, velocities, and
accelerations along body paths.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer


class VectorOverlayRenderer(BaseRenderer):
    """Renderer for force / torque / velocity vector overlays.

    All methods accept a ``Figure`` and populate it with 3-D quiver or
    arrow overlays that can be combined with trajectory data.
    """

    # ------------------------------------------------------------------
    # Contact force vectors
    # ------------------------------------------------------------------
    def plot_contact_force_vectors(
        self,
        fig: Figure,
        positions: np.ndarray | None = None,
        forces: np.ndarray | None = None,
        scale: float = 0.01,
        subsample: int = 1,
    ) -> None:
        """Overlay contact-force arrows on a 3-D axes.

        Args:
            fig: Target figure (a 3-D subplot is created if needed).
            positions: ``(N, 3)`` contact application points.  If ``None``
                the renderer tries ``self.data.get_series("contact_positions")``.
            forces: ``(N, 3)`` force vectors.  Fallback to
                ``self.data.get_series("contact_forces")``.
            scale: Arrow length scaling factor.
            subsample: Plot every *n*-th vector for clarity.
        """
        if positions is None or forces is None:
            _t, pos_raw = self.data.get_series("contact_positions")
            _t, frc_raw = self.data.get_series("contact_forces")
            if len(pos_raw) == 0 or len(frc_raw) == 0:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No contact data", ha="center", va="center")
                return
            positions = np.asarray(pos_raw)
            forces = np.asarray(frc_raw)

        positions = np.asarray(positions)
        forces = np.asarray(forces)

        if positions.ndim != 2 or positions.shape[1] < 3:
            return
        if forces.shape != positions.shape:
            return

        idx = slice(None, None, subsample)
        p = positions[idx]
        f = forces[idx] * scale

        ax = fig.add_subplot(111, projection="3d")
        ax.quiver(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            f[:, 0],
            f[:, 1],
            f[:, 2],
            color=self.colors.get("quaternary", "red"),
            arrow_length_ratio=0.15,
            linewidth=1.2,
            alpha=0.8,
            label="Contact force",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Contact Force Vectors", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Joint torque vectors
    # ------------------------------------------------------------------
    def plot_joint_torque_vectors(
        self,
        fig: Figure,
        joint_positions: np.ndarray | None = None,
        torque_axes: np.ndarray | None = None,
        torque_magnitudes: np.ndarray | None = None,
        scale: float = 0.005,
    ) -> None:
        """Overlay joint-torque arrows at each joint.

        Args:
            fig: Target figure.
            joint_positions: ``(N, 3)`` joint world positions.
            torque_axes: ``(N, 3)`` unit torque axis directions.
            torque_magnitudes: ``(N,)`` signed magnitudes.
            scale: Arrow length scaling.
        """
        if joint_positions is None:
            _t, jp_raw = self.data.get_series("joint_world_positions")
            _t, tm_raw = self.data.get_series("joint_torques")
            if len(jp_raw) == 0 or len(tm_raw) == 0:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No torque data", ha="center", va="center")
                return
            joint_positions = np.asarray(jp_raw)
            torque_magnitudes = np.asarray(tm_raw)
            # Default: torque axis along z
            torque_axes = np.tile([0.0, 0.0, 1.0], (len(joint_positions), 1))

        joint_positions = np.asarray(joint_positions)
        if torque_axes is None:
            torque_axes = np.tile([0.0, 0.0, 1.0], (len(joint_positions), 1))
        if torque_magnitudes is None:
            torque_magnitudes = np.ones(len(joint_positions))

        torque_axes = np.asarray(torque_axes)
        torque_magnitudes = np.asarray(torque_magnitudes)

        vectors = torque_axes * (torque_magnitudes[:, None] * scale)

        # Colour positive green, negative blue
        colours = np.where(
            torque_magnitudes[:, None] >= 0,
            np.array([[0.0, 0.8, 0.0, 0.8]]),
            np.array([[0.0, 0.0, 0.8, 0.8]]),
        )

        ax = fig.add_subplot(111, projection="3d")
        for i in range(len(joint_positions)):
            ax.quiver(
                joint_positions[i, 0],
                joint_positions[i, 1],
                joint_positions[i, 2],
                vectors[i, 0],
                vectors[i, 1],
                vectors[i, 2],
                color=colours[i],
                arrow_length_ratio=0.2,
                linewidth=1.5,
            )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Joint Torque Vectors", fontsize=14, fontweight="bold")
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Velocity / acceleration vectors along a trajectory
    # ------------------------------------------------------------------
    def plot_trajectory_with_vectors(
        self,
        fig: Figure,
        positions: np.ndarray,
        vectors: np.ndarray,
        vector_label: str = "Velocity",
        trajectory_label: str = "Club head",
        scale: float = 0.01,
        subsample: int = 5,
        times: np.ndarray | None = None,
    ) -> None:
        """Plot a 3-D trajectory with vector arrows overlaid at each point.

        Args:
            fig: Target figure.
            positions: ``(N, 3)`` trajectory points.
            vectors: ``(N, 3)`` vector values at each point.
            vector_label: Legend label for the arrows.
            trajectory_label: Legend label for the path.
            scale: Arrow length scaling.
            subsample: Plot every *n*-th arrow.
            times: Optional timestamps for coloring the trajectory.
        """
        positions = np.asarray(positions)
        vectors = np.asarray(vectors)

        ax = fig.add_subplot(111, projection="3d")

        # Trajectory line
        if times is not None:
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c=np.asarray(times),
                cmap="viridis",
                s=3,
                label=trajectory_label,
            )
            fig.colorbar(scatter, ax=ax, label="Time (s)", shrink=0.6)
        else:
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=self.colors.get("primary", "#1f77b4"),
                linewidth=1.5,
                label=trajectory_label,
            )

        # Vector arrows
        idx = slice(None, None, subsample)
        p = positions[idx]
        v = vectors[idx] * scale
        ax.quiver(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            v[:, 0],
            v[:, 1],
            v[:, 2],
            color=self.colors.get("quaternary", "red"),
            arrow_length_ratio=0.12,
            linewidth=0.8,
            alpha=0.7,
            label=vector_label,
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(
            f"{trajectory_label} with {vector_label} Vectors",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Desired vs actual trajectory comparison
    # ------------------------------------------------------------------
    def plot_trajectory_comparison(
        self,
        fig: Figure,
        actual: np.ndarray,
        desired: np.ndarray,
        label_actual: str = "Actual",
        label_desired: str = "Desired",
        show_error_lines: bool = True,
        error_subsample: int = 10,
    ) -> None:
        """Plot desired vs actual 3-D trajectories with error indicators.

        Args:
            fig: Target figure.
            actual: ``(N, 3)`` actual trajectory.
            desired: ``(M, 3)`` desired trajectory.
            label_actual: Legend label for actual path.
            label_desired: Legend label for desired path.
            show_error_lines: Draw dashed lines between matched points.
            error_subsample: Subsample rate for error lines.
        """
        actual = np.asarray(actual)
        desired = np.asarray(desired)

        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            actual[:, 0],
            actual[:, 1],
            actual[:, 2],
            color="#FF4500",
            linewidth=2.0,
            label=label_actual,
        )
        ax.plot(
            desired[:, 0],
            desired[:, 1],
            desired[:, 2],
            color="#00CED1",
            linewidth=2.0,
            linestyle="--",
            label=label_desired,
            alpha=0.7,
        )

        # Error lines connecting corresponding points
        if show_error_lines:
            n = min(len(actual), len(desired))
            for i in range(0, n, error_subsample):
                ax.plot(
                    [actual[i, 0], desired[i, 0]],
                    [actual[i, 1], desired[i, 1]],
                    [actual[i, 2], desired[i, 2]],
                    color="gray",
                    linewidth=0.5,
                    alpha=0.4,
                )

        # Start / end markers
        ax.scatter(
            *actual[0, :3], color="#FF4500", s=60, marker="o", zorder=5, label="Start"
        )
        ax.scatter(
            *actual[-1, :3], color="#FF4500", s=60, marker="s", zorder=5, label="End"
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Trajectory Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Ground reaction force butterfly
    # ------------------------------------------------------------------
    def plot_grf_vectors(
        self,
        fig: Figure,
        cop_positions: np.ndarray | None = None,
        grf_vectors: np.ndarray | None = None,
        scale: float = 0.001,
        subsample: int = 3,
    ) -> None:
        """Overlay ground-reaction-force arrows at centre-of-pressure.

        Args:
            fig: Target figure.
            cop_positions: ``(N, 3)`` centre-of-pressure positions.
            grf_vectors: ``(N, 3)`` GRF vectors.
            scale: Arrow length scaling.
            subsample: Plot every *n*-th vector.
        """
        if cop_positions is None or grf_vectors is None:
            _t, cop_raw = self.data.get_series("cop_position")
            _t, grf_raw = self.data.get_series("ground_reaction_force")
            if len(cop_raw) == 0 or len(grf_raw) == 0:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No GRF data", ha="center", va="center")
                return
            cop_positions = np.asarray(cop_raw)
            grf_vectors = np.asarray(grf_raw)

        cop = np.asarray(cop_positions)
        grf = np.asarray(grf_vectors)

        idx = slice(None, None, subsample)
        p = cop[idx]
        f = grf[idx] * scale

        ax = fig.add_subplot(111, projection="3d")
        ax.quiver(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            f[:, 0],
            f[:, 1],
            f[:, 2],
            color="#228B22",
            arrow_length_ratio=0.12,
            linewidth=1.0,
            alpha=0.8,
            label="GRF",
        )
        ax.plot(
            cop[:, 0],
            cop[:, 1],
            cop[:, 2],
            color="#228B22",
            linewidth=0.8,
            alpha=0.5,
            label="CoP path",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Ground Reaction Force Vectors", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
