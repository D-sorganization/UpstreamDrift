"""Force vector visualization renderer.

Renders joint force vectors as 3-D quiver overlays, supporting:
- Total force vectors (with applied torques)
- ZTCF force vectors (zero-torque passive component)
- Delta force vectors (active control component = total − ZTCF)
- Combined decomposition subplots

Design by Contract
------------------
Preconditions:
  - Figure must not be None.
  - Positions and forces must be finite arrays of matching shapes.
Postconditions:
  - At least one axes is created on the figure.
  - All plotted data is finite.

DRY
---
Shared arrow-rendering logic is in ``_render_quiver_overlay``.
Color/style configuration is centralized in module-level constants.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from src.shared.python.plotting.renderers.base import BaseRenderer

# ── Force type color configuration ────────────────────────────────────
# Consistent color scheme across all force vector plots.
COLOR_TOTAL = "#32CD32"  # Lime green — total (driven) forces
COLOR_ZTCF = "#D278FF"  # Violet — passive / counterfactual
COLOR_DELTA = "#FF6347"  # Tomato — active control component

LINESTYLE_TOTAL = "-"
LINESTYLE_ZTCF = "--"
LINESTYLE_DELTA = "-."


class ForceVectorRenderer(BaseRenderer):
    """Renderer for joint force vector overlays (total, ZTCF, delta).

    Provides methods to plot force vectors at joint positions in 3-D,
    with support for single-type plots and combined decomposition views.
    """

    # ------------------------------------------------------------------
    # Total force vectors
    # ------------------------------------------------------------------

    def plot_joint_force_vectors(
        self,
        fig: Figure,
        *,
        positions: np.ndarray | None = None,
        forces: np.ndarray | None = None,
        frame_idx: int | None = None,
        scale: float = 0.01,
        title: str = "Joint Force Vectors",
    ) -> None:
        """Plot total joint force vectors as 3-D arrows.

        Args:
            fig: Target figure.
            positions: ``(n_joints, 3)`` joint world positions.
            forces: ``(n_joints, 3)`` force vectors in Newtons.
            frame_idx: If positions/forces are None, fetch from data at
                this frame index.
            scale: Arrow length scaling factor.
            title: Plot title.
        """
        assert fig is not None, "fig must be provided"
        positions, forces = self._resolve_force_data(
            positions, forces, frame_idx, "joint_forces"
        )
        self._render_quiver_overlay(
            fig,
            positions=positions,
            vectors=forces,
            scale=scale,
            color=COLOR_TOTAL,
            label="Total Force",
            title=title,
        )

    # ------------------------------------------------------------------
    # ZTCF force vectors
    # ------------------------------------------------------------------

    def plot_ztcf_force_vectors(
        self,
        fig: Figure,
        *,
        positions: np.ndarray | None = None,
        ztcf_forces: np.ndarray | None = None,
        frame_idx: int | None = None,
        scale: float = 0.01,
        title: str = "ZTCF Force Vectors (Passive)",
    ) -> None:
        """Plot zero-torque counterfactual force vectors.

        Args:
            fig: Target figure.
            positions: ``(n_joints, 3)`` joint world positions.
            ztcf_forces: ``(n_joints, 3)`` ZTCF force vectors.
            frame_idx: Frame index for data lookup if arrays not given.
            scale: Arrow length scaling factor.
            title: Plot title.
        """
        assert fig is not None, "fig must be provided"
        positions, ztcf_forces = self._resolve_force_data(
            positions, ztcf_forces, frame_idx, "ztcf_joint_forces"
        )
        self._render_quiver_overlay(
            fig,
            positions=positions,
            vectors=ztcf_forces,
            scale=scale,
            color=COLOR_ZTCF,
            label="ZTCF (Passive)",
            title=title,
            linestyle=LINESTYLE_ZTCF,
        )

    # ------------------------------------------------------------------
    # Delta (active control) force vectors
    # ------------------------------------------------------------------

    def plot_force_delta_vectors(
        self,
        fig: Figure,
        *,
        positions: np.ndarray,
        total_forces: np.ndarray,
        ztcf_forces: np.ndarray,
        scale: float = 0.01,
        title: str = "Active Control Force Vectors (Total − ZTCF)",
    ) -> None:
        """Plot delta force vectors (total − ZTCF = active component).

        Args:
            fig: Target figure.
            positions: ``(n_joints, 3)`` joint world positions.
            total_forces: ``(n_joints, 3)`` total force vectors.
            ztcf_forces: ``(n_joints, 3)`` ZTCF force vectors.
            scale: Arrow length scaling factor.
            title: Plot title.
        """
        assert fig is not None, "fig must be provided"
        total_forces = np.asarray(total_forces)
        ztcf_forces = np.asarray(ztcf_forces)

        assert (
            total_forces.shape == ztcf_forces.shape
        ), f"Shape mismatch: total {total_forces.shape} vs ztcf {ztcf_forces.shape}"

        delta = total_forces - ztcf_forces
        self._render_quiver_overlay(
            fig,
            positions=np.asarray(positions),
            vectors=delta,
            scale=scale,
            color=COLOR_DELTA,
            label="Delta (Active)",
            title=title,
            linestyle=LINESTYLE_DELTA,
        )

    # ------------------------------------------------------------------
    # Combined decomposition (3 subplots)
    # ------------------------------------------------------------------

    def plot_force_decomposition(
        self,
        fig: Figure,
        *,
        positions: np.ndarray,
        total_forces: np.ndarray,
        ztcf_forces: np.ndarray,
        scale: float = 0.01,
    ) -> None:
        """Plot side-by-side decomposition: Total | ZTCF | Delta.

        Creates three 3-D subplots showing the full force decomposition
        at a single time frame.

        Args:
            fig: Target figure (will be cleared).
            positions: ``(n_joints, 3)`` joint world positions.
            total_forces: ``(n_joints, 3)`` total force vectors.
            ztcf_forces: ``(n_joints, 3)`` ZTCF force vectors.
            scale: Arrow length scaling.
        """
        assert fig is not None, "fig must be provided"
        positions = np.asarray(positions)
        total_forces = np.asarray(total_forces)
        ztcf_forces = np.asarray(ztcf_forces)

        if positions.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No force data available", ha="center", va="center")
            return

        delta = total_forces - ztcf_forces

        configs = [
            ("Total Forces", total_forces, COLOR_TOTAL, "Total"),
            ("ZTCF (Passive)", ztcf_forces, COLOR_ZTCF, "ZTCF"),
            ("Delta (Active)", delta, COLOR_DELTA, "Delta"),
        ]

        for idx, (title, vectors, color, label) in enumerate(configs, start=1):
            ax = fig.add_subplot(1, 3, idx, projection="3d")
            self._draw_quiver_on_axes(
                ax,
                positions=positions,
                vectors=vectors,
                scale=scale,
                color=color,
                label=label,
            )
            ax.set_title(title, fontsize=11, fontweight="bold")
            self._format_3d_axes(ax)

        fig.suptitle(
            "Force Decomposition: Total = ZTCF + Delta",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Time-series force magnitude comparison
    # ------------------------------------------------------------------

    def plot_force_magnitude_timeseries(
        self,
        fig: Figure,
        *,
        times: np.ndarray,
        total_magnitudes: np.ndarray,
        ztcf_magnitudes: np.ndarray,
        joint_idx: int = 0,
    ) -> None:
        """Plot force magnitude time series: total vs ZTCF vs delta.

        Args:
            fig: Target figure.
            times: Time array, shape ``(N,)``.
            total_magnitudes: Total force magnitudes per frame, shape ``(N,)``.
            ztcf_magnitudes: ZTCF force magnitudes per frame, shape ``(N,)``.
            joint_idx: Joint index for the title label.
        """
        assert fig is not None, "fig must be provided"
        times = np.asarray(times)
        total_magnitudes = np.asarray(total_magnitudes)
        ztcf_magnitudes = np.asarray(ztcf_magnitudes)

        if times.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No time-series data", ha="center", va="center")
            return

        delta_magnitudes = np.abs(total_magnitudes - ztcf_magnitudes)
        ax = fig.add_subplot(111)

        ax.plot(
            times,
            total_magnitudes,
            color=COLOR_TOTAL,
            linewidth=2,
            label="Total",
        )
        ax.plot(
            times,
            ztcf_magnitudes,
            color=COLOR_ZTCF,
            linewidth=2,
            linestyle="--",
            label="ZTCF (Passive)",
        )
        ax.plot(
            times,
            delta_magnitudes,
            color=COLOR_DELTA,
            linewidth=1.5,
            linestyle="-.",
            label="Delta (Active)",
        )

        joint_name = self.data.get_joint_name(joint_idx)
        self.format_axis(
            ax,
            xlabel="Time (s)",
            ylabel="Force Magnitude (N)",
            title=f"Force Decomposition: {joint_name}",
        )
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _resolve_force_data(
        self,
        positions: np.ndarray | None,
        forces: np.ndarray | None,
        frame_idx: int | None,
        force_field: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve positions/forces from explicit args or DataManager.

        If explicit arrays are provided, use them directly.
        Otherwise, fetch from DataManager at the given frame index.
        """
        if positions is not None and forces is not None:
            return np.asarray(positions), np.asarray(forces)

        _times, pos_raw = self.data.get_series("joint_world_positions")
        _times, frc_raw = self.data.get_series(force_field)
        pos_arr = np.asarray(pos_raw)
        frc_arr = np.asarray(frc_raw)

        if pos_arr.size == 0 or frc_arr.size == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        idx = frame_idx if frame_idx is not None else 0
        idx = max(0, min(idx, len(pos_arr) - 1))
        return pos_arr[idx], frc_arr[idx]

    def _render_quiver_overlay(
        self,
        fig: Figure,
        *,
        positions: np.ndarray,
        vectors: np.ndarray,
        scale: float,
        color: str,
        label: str,
        title: str,
        linestyle: str = "-",
    ) -> None:
        """Render a single set of 3-D force arrows on the figure."""
        positions = np.asarray(positions)
        vectors = np.asarray(vectors)

        if positions.size == 0 or vectors.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No force data", ha="center", va="center")
            return

        # Ensure 2D shape
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Pad to 3D if needed
        positions = self._ensure_3d(positions)
        vectors = self._ensure_3d(vectors)

        ax = fig.add_subplot(111, projection="3d")
        self._draw_quiver_on_axes(
            ax,
            positions=positions,
            vectors=vectors,
            scale=scale,
            color=color,
            label=label,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        self._format_3d_axes(ax)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

    def _draw_quiver_on_axes(
        self,
        ax,
        *,
        positions: np.ndarray,
        vectors: np.ndarray,
        scale: float,
        color: str,
        label: str,
    ) -> None:
        """Draw quiver arrows on an existing 3D axes."""
        scaled = vectors * scale
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            scaled[:, 0],
            scaled[:, 1],
            scaled[:, 2],
            color=color,
            arrow_length_ratio=0.15,
            linewidth=1.5,
            alpha=0.85,
            label=label,
        )

        # Draw joint markers
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color="black",
            s=25,
            zorder=5,
        )

    @staticmethod
    def _format_3d_axes(ax) -> None:
        """Apply standard formatting to a 3-D axes."""
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_zlabel("Z (m)", fontsize=10)

    @staticmethod
    def _ensure_3d(arr: np.ndarray) -> np.ndarray:
        """Pad a (N, 2) array to (N, 3) by appending zeros."""
        if arr.ndim == 2 and arr.shape[1] == 2:
            return np.column_stack([arr, np.zeros(arr.shape[0])])
        return arr
