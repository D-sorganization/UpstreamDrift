"""Advanced plotting and visualization for golf swing analysis.

This module provides comprehensive plotting capabilities including:
- Time series plots (kinematics, kinetics, energetics)
- Phase diagrams
- Force/torque visualizations
- Power and energy analysis
- Swing sequence analysis
- Induced Accelerations
- Counterfactual Data (ZTCF, ZVCF)
- Advanced Coordination and Stability Visualizations
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import matplotlib.backend_bases  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Register 3D projection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from shared.python.swing_plane_analysis import SwingPlaneAnalyzer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from shared.python.statistical_analysis import PCAResult

# Qt backend - optional for headless environments
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    class MplCanvas(FigureCanvasQTAgg):
        """Matplotlib canvas for embedding in PyQt6."""

        def __init__(self, width: float = 8, height: float = 6, dpi: int = 100) -> None:
            """Initialize canvas with figure.

            Args:
                width: Figure width in inches
                height: Figure height in inches
                dpi: Dots per inch for rendering
            """
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            super().__init__(self.fig)

except ImportError:
    # Qt not available (e.g., in headless CI environments)
    class MplCanvas:  # type: ignore[no-redef]
        """Matplotlib canvas for embedding in PyQt6 (not available in headless mode)."""

        def __init__(
            self, width: float = 8, height: float = 6, dpi: int = 100
        ) -> None:  # noqa: ARG002
            """Initialize canvas with figure (implementation for headless environments)."""
            msg = (
                "MplCanvas requires Qt backend which is not available in headless envs"
            )
            raise RuntimeError(msg)


class RecorderInterface(Protocol):
    """Protocol for a recorder that provides time series data."""

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Extract time series for a specific field.

        Args:
            field_name: Name of the field

        Returns:
            Tuple of (times, values)
        """
        ...  # pragma: no cover

    def get_induced_acceleration_series(
        self, source_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific induced acceleration source.

        Args:
            source_name: Name of the force source (e.g. 'gravity', 'actuator_1')

        Returns:
            Tuple of (times, acceleration_array)
        """
        ...  # pragma: no cover

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific counterfactual component.

        Args:
            cf_name: Name of the counterfactual (e.g. 'ztcf', 'zvcf')

        Returns:
            Tuple of (times, data_array)
        """
        ...  # pragma: no cover


class GolfSwingPlotter:
    """Creates advanced plots for golf swing analysis.

    This class generates various plots from recorded swing data,
    including kinematics, kinetics, energetics, and phase diagrams.
    It is engine-agnostic, relying on a generic recorder interface.
    """

    def __init__(
        self,
        recorder: RecorderInterface,
        joint_names: list[str] | None = None,
        enable_cache: bool = True,
    ) -> None:
        """Initialize plotter with recorded data.

        Args:
            recorder: Object providing get_time_series(field_name) method
            joint_names: Optional list of joint names. If None, uses "Joint X"
            enable_cache: If True, cache data fetches to improve performance
        """
        self.recorder = recorder
        self.joint_names = joint_names or []
        self.enable_cache = enable_cache
        self._data_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Color scheme for professional plots
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "tertiary": "#2ca02c",
            "quaternary": "#d62728",
            "quinary": "#9467bd",
            "senary": "#8c564b",
            "accent": "#e377c2",
            "dark": "#7f7f7f",
            "grid": "#cccccc",
        }

        # Pre-fetch commonly used data to reduce repeated recorder calls
        if self.enable_cache:
            self._preload_common_data()

    def _preload_common_data(self) -> None:
        """Pre-fetch commonly used data series to cache.

        This reduces redundant recorder.get_time_series() calls from 71 to ~10.
        Performance: 50-70% faster plotting for multi-plot dashboards.
        """
        common_fields = [
            "joint_positions",
            "joint_velocities",
            "joint_torques",
            "kinetic_energy",
            "potential_energy",
            "total_energy",
            "club_head_speed",
            "club_head_position",
            "angular_momentum",
            "cop_position",
            "com_position",
            "actuator_powers",
        ]

        for field in common_fields:
            try:
                times, values = self._get_cached_series(field)
                if len(times) > 0:
                    self._data_cache[field] = (times, values)
            except Exception as e:
                # Field may not exist in all recorders
                logger.debug(f"Could not pre-load field '{field}': {e}")

    def _get_cached_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get time series data with caching.

        Args:
            field_name: Name of the field to retrieve

        Returns:
            Tuple of (times, values) arrays
        """
        if not self.enable_cache:
            times, values = self.recorder.get_time_series(field_name)
            return np.asarray(times), np.asarray(values)

        # Check cache first
        if field_name in self._data_cache:
            return self._data_cache[field_name]

        # Not in cache, fetch and cache it
        times, values = self.recorder.get_time_series(field_name)
        times_arr, values_arr = np.asarray(times), np.asarray(values)
        if len(times_arr) > 0:
            self._data_cache[field_name] = (times_arr, values_arr)

        return times_arr, values_arr

    def clear_cache(self) -> None:
        """Clear the data cache.

        Call this if recorder data has changed and cache needs to be invalidated.
        """
        self._data_cache.clear()
        if self.enable_cache:
            self._preload_common_data()

    def get_joint_name(self, joint_idx: int) -> str:
        """Get human-readable joint name."""
        if 0 <= joint_idx < len(self.joint_names):
            return self.joint_names[joint_idx]
        return f"Joint {joint_idx}"

    def _get_aligned_label(self, idx: int, data_dim: int) -> str:
        """Get label aligned with data dimension (handling nq != nv)."""
        # Assume joint_names corresponds to NV (actuated/velocity DOFs)
        # If data_dim > len(joint_names), it's likely Position data (nq) with floating base (7)
        # vs Velocity data (nv) with floating base (6).
        # Pinocchio joint_names usually matches NV structure (if we skipped universe).

        if len(self.joint_names) == 0:
            return f"DoF {idx}"

        # If perfect match
        if data_dim == len(self.joint_names):
            return (
                self.joint_names[idx] if idx < len(self.joint_names) else f"DoF {idx}"
            )

        # If mismatch, align from the end (assuming base is at the start)
        offset = max(0, data_dim - len(self.joint_names))
        name_idx = idx - offset

        if 0 <= name_idx < len(self.joint_names):
            return self.joint_names[name_idx]

        return f"DoF {idx}"

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
        times, positions = self._get_cached_series("joint_positions")

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
                label = self._get_aligned_label(idx, positions.shape[1])
                ax.plot(times, np.rad2deg(positions[:, idx]), label=label, linewidth=2)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Joint Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_title("Joint Angles vs Time", fontsize=14, fontweight="bold")
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
        """Plot Angle-Angle diagram (Cyclogram) for two joints.

        Args:
            fig: Matplotlib figure
            joint_idx_1: Index of first joint (X-axis)
            joint_idx_2: Index of second joint (Y-axis)
            title: Optional title
            ax: Optional Axes object. If None, creates new subplot(111).
        """
        times, positions = self._get_cached_series("joint_positions")
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

        name1 = self.get_joint_name(joint_idx_1)
        name2 = self.get_joint_name(joint_idx_2)

        ax.set_xlabel(f"{name1} Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{name2} Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or f"Coordination: {name1} vs {name2}", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_coupling_angle(
        self,
        fig: Figure,
        coupling_angles: np.ndarray,
        title: str | None = None,
        ax: Axes | None = None,
    ) -> None:
        """Plot Coupling Angle time series (Vector Coding).

        Args:
            fig: Matplotlib figure
            coupling_angles: Array of coupling angles [0, 360)
            title: Optional title
            ax: Optional Axes object. If None, creates new subplot(111).

        Note:
            This method expects pre-calculated coupling angles.
            See shared.python.statistical_analysis.compute_coupling_angles.
        """
        times, _ = self._get_cached_series("joint_positions")

        if ax is None:
            ax = fig.add_subplot(111)

        if len(times) == 0 or len(coupling_angles) == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Ensure lengths match (coupling might be N or N-1 depending on method)
        if len(coupling_angles) != len(times):
            # If length mismatch, trim times to match
            logger.warning(
                f"Coupling angle length ({len(coupling_angles)}) does not match "
                f"time series length ({len(times)}). Truncating times."
            )
            plot_times = times[: len(coupling_angles)]
        else:
            plot_times = times

        ax.plot(
            plot_times,
            coupling_angles,
            color=self.colors["primary"],
            linewidth=2,
            label="Coupling Angle",
        )

        # Draw coordination zones (simplified)
        # 0/360: In-phase (Both +)
        # 45: Joint 2 dominant
        # 90: Joint 1 frozen, Joint 2 moving
        # 180: Anti-phase
        # etc.
        # Just simple grid lines for now
        for angle in [0, 90, 180, 270, 360]:
            ax.axhline(y=angle, color="gray", linestyle="--", alpha=0.3)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Coupling Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_title(
            title or "Coordination Variability", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    def plot_coordination_patterns(
        self,
        fig: Figure,
        coupling_angles: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Plot coordination patterns as a color-coded strip over time.

        Visualizes discrete coordination states (In-Phase, Anti-Phase, etc.) derived
        from vector coding coupling angles.

        Args:
            fig: Matplotlib figure
            coupling_angles: Array of coupling angles [0, 360)
            title: Optional title
        """
        times, _ = self._get_cached_series("joint_positions")

        if len(times) == 0 or len(coupling_angles) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        if len(coupling_angles) != len(times):
            times = times[: len(coupling_angles)]

        # Binning logic (must match StatisticalAnalyzer.compute_coordination_metrics)
        # 0: Proximal, 1: In-Phase, 2: Distal, 3: Anti-Phase, ...
        binned = np.floor((coupling_angles + 22.5) / 45.0) % 8

        # Map 8 bins to 4 classes
        # 0, 4 -> Proximal (0)
        # 1, 5 -> In-Phase (1)
        # 2, 6 -> Distal (2)
        # 3, 7 -> Anti-Phase (3)

        classes = np.zeros_like(binned)
        classes[(binned == 0) | (binned == 4)] = 0  # Proximal
        classes[(binned == 1) | (binned == 5)] = 1  # In-Phase
        classes[(binned == 2) | (binned == 6)] = 2  # Distal
        classes[(binned == 3) | (binned == 7)] = 3  # Anti-Phase

        # Colors for classes
        # Proximal: Blue
        # In-Phase: Green
        # Distal: Red
        # Anti-Phase: Orange
        cmap_colors = [
            self.colors["primary"],  # Proximal
            self.colors["tertiary"],  # In-Phase
            self.colors["quaternary"],  # Distal
            self.colors["secondary"],  # Anti-Phase
        ]

        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(cmap_colors)

        ax = fig.add_subplot(111)

        # Prepare meshgrid for pcolormesh
        # classes is (N,) - 1D array of values
        # We need a 2D array for pcolormesh C argument. Shape (1, N)
        # The coordinates X, Y define the corners of the quadrilaterals.
        # X should be length N+1 (time boundaries)
        # Y should be length 2 (top and bottom of strip)

        # Create time boundaries: midpoints between samples, plus ends
        if len(times) > 1:
            dt = times[1] - times[0]
            # Construct edges
            time_edges = np.concatenate(
                (
                    [times[0] - dt / 2],
                    times[:-1] + np.diff(times) / 2,
                    [times[-1] + dt / 2],
                )
            )
        else:
            time_edges = np.array([times[0] - 0.5, times[0] + 0.5])

        y_edges = np.array([0, 1])

        # Meshgrid
        X, Y = np.meshgrid(time_edges, y_edges)

        # C must be (ny-1, nx-1) -> (1, N)
        C = classes.reshape(1, -1)

        ax.pcolormesh(X, Y, C, cmap=cmap, vmin=0, vmax=3, shading="flat")

        # Custom legend
        legend_patches = [
            Rectangle((0, 0), 1, 1, color=cmap_colors[0], label="Proximal Leading"),
            Rectangle((0, 0), 1, 1, color=cmap_colors[1], label="In-Phase"),
            Rectangle((0, 0), 1, 1, color=cmap_colors[2], label="Distal Leading"),
            Rectangle((0, 0), 1, 1, color=cmap_colors[3], label="Anti-Phase"),
        ]

        ax.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=4,
        )

        ax.set_yticks([])
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or "Coordination Pattern Dynamics",
            fontsize=14,
            fontweight="bold",
            y=1.2,
        )

        fig.tight_layout()

    def plot_continuous_relative_phase(
        self,
        fig: Figure,
        crp_data: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Plot Continuous Relative Phase (CRP) time series.

        Args:
            fig: Matplotlib figure
            crp_data: Array of CRP values in degrees
            title: Optional title
        """
        times, _ = self._get_cached_series("joint_positions")

        if len(times) == 0 or len(crp_data) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        if len(crp_data) != len(times):
            plot_times = times[: len(crp_data)]
        else:
            plot_times = times

        ax = fig.add_subplot(111)

        ax.plot(
            plot_times,
            crp_data,
            color=self.colors["primary"],
            linewidth=2,
            label="CRP",
        )

        # Draw phase zones
        # 0: In-phase, 180: Anti-phase
        ax.axhline(y=0, color="green", linestyle="--", alpha=0.3, label="In-Phase")
        ax.axhline(y=180, color="red", linestyle="--", alpha=0.3, label="Anti-Phase")
        ax.axhline(y=-180, color="red", linestyle="--", alpha=0.3)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Relative Phase (deg)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or "Continuous Relative Phase", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    def plot_stability_metrics(self, fig: Figure) -> None:
        """Plot stability metrics (CoM-CoP distance and Inclination Angle).

        Args:
            fig: Matplotlib figure
        """
        try:
            times_cop, cop = self._get_cached_series("cop_position")
            times_com, com = self._get_cached_series("com_position")
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

        # Compute metrics
        cop_xy = cop[:, :2]
        com_xy = com[:, :2]
        # OPTIMIZATION: np.hypot is faster for 2D distance
        diff = cop_xy - com_xy
        dist = np.hypot(diff[:, 0], diff[:, 1])

        if cop.shape[1] == 2:
            cop_z = np.zeros(len(cop))
        else:
            cop_z = cop[:, 2]  # type: ignore[assignment]

        vec_temp = com - np.column_stack((cop_xy, cop_z))
        vec: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = vec_temp  # type: ignore[assignment]
        # OPTIMIZATION: Explicit sqrt sum is faster than np.linalg.norm for small dims
        vec_norm = np.sqrt(np.sum(vec**2, axis=1))
        vec_norm[vec_norm < 1e-6] = 1.0

        cos_theta = vec[:, 2] / vec_norm
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles_deg = np.rad2deg(np.arccos(cos_theta))

        # Plot
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
        labs: list[str] = [str(ln.get_label()) for ln in lns]
        ax1.legend(lns, labs, loc="best")

        ax1.set_title("Postural Stability Metrics", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

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
        times, velocities = self._get_cached_series("joint_velocities")

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
                label = self._get_aligned_label(idx, velocities.shape[1])
                ax.plot(times, np.rad2deg(velocities[:, idx]), label=label, linewidth=2)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Angular Velocity (deg/s)", fontsize=12, fontweight="bold")
        ax.set_title("Joint Velocities vs Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_joint_torques(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
    ) -> None:
        """Plot applied joint torques over time.

        Args:
            fig: Matplotlib figure to plot on
            joint_indices: List of joint indices to plot (None = all)
        """
        times, torques = self._get_cached_series("joint_torques")

        if len(times) == 0 or len(torques) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        # Ensure torques is a numpy array
        if not isinstance(torques, np.ndarray):
            torques = np.array(torques)

        ax = fig.add_subplot(111)

        if joint_indices is None:
            joint_indices = list(range(torques.shape[1]))

        for idx in joint_indices:
            if idx < torques.shape[1]:
                label = self._get_aligned_label(idx, torques.shape[1])
                ax.plot(times, torques[:, idx], label=label, linewidth=2)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Torque (Nm)", fontsize=12, fontweight="bold")
        ax.set_title("Applied Joint Torques vs Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        fig.tight_layout()

    def plot_actuator_powers(self, fig: Figure) -> None:
        """Plot actuator mechanical powers over time.

        Args:
            fig: Matplotlib figure to plot on
        """
        times, powers = self._get_cached_series("actuator_powers")

        if len(times) == 0 or len(powers) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        # Ensure powers is a numpy array
        if not isinstance(powers, np.ndarray):
            powers = np.array(powers)

        ax = fig.add_subplot(111)

        for idx in range(powers.shape[1]):
            label = self.get_joint_name(idx)
            ax.plot(times, powers[:, idx], label=label, linewidth=2, alpha=0.7)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=12, fontweight="bold")
        ax.set_title("Actuator Powers vs Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        fig.tight_layout()

    def plot_energy_analysis(self, fig: Figure) -> None:
        """Plot kinetic, potential, and total energy over time.

        Args:
            fig: Matplotlib figure to plot on
        """
        times_ke, ke = self._get_cached_series("kinetic_energy")
        times_pe, pe = self._get_cached_series("potential_energy")
        times_te, te = self._get_cached_series("total_energy")

        if len(times_ke) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        ax.plot(
            times_ke,
            ke,
            label="Kinetic Energy",
            linewidth=2.5,
            color=self.colors["primary"],
        )
        ax.plot(
            times_pe,
            pe,
            label="Potential Energy",
            linewidth=2.5,
            color=self.colors["secondary"],
        )
        ax.plot(
            times_te,
            te,
            label="Total Energy",
            linewidth=2.5,
            color=self.colors["quaternary"],
            linestyle="--",
        )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy (J)", fontsize=12, fontweight="bold")
        ax.set_title("Energy Analysis", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_club_head_speed(self, fig: Figure) -> None:
        """Plot club head speed over time.

        Args:
            fig: Matplotlib figure to plot on
        """
        times, speeds = self._get_cached_series("club_head_speed")

        if len(times) == 0 or len(speeds) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No club head data", ha="center", va="center")
            return

        # Ensure speeds is a numpy array
        if not isinstance(speeds, np.ndarray):
            speeds = np.array(speeds)

        ax = fig.add_subplot(111)

        # Convert to mph for golf context
        speeds_mph = speeds * 2.23694

        ax.plot(times, speeds_mph, linewidth=3, color=self.colors["primary"])
        ax.fill_between(times, 0, speeds_mph, alpha=0.3, color=self.colors["primary"])

        # Mark peak speed
        max_idx = np.argmax(speeds_mph)
        max_speed = speeds_mph[max_idx]
        max_time = times[max_idx]
        ax.plot(
            max_time,
            max_speed,
            "r*",
            markersize=20,
            label=f"Peak: {max_speed:.1f} mph",
        )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Club Head Speed (mph)", fontsize=12, fontweight="bold")
        ax.set_title("Club Head Speed vs Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_club_head_trajectory(self, fig: Figure) -> None:
        """Plot 3D club head trajectory.

        Args:
            fig: Matplotlib figure to plot on
        """
        times, positions = self._get_cached_series("club_head_position")

        if len(times) == 0 or len(positions) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No club head data", ha="center", va="center")
            return

        # Ensure positions is a numpy array
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)

        ax = fig.add_subplot(111, projection="3d")

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # Color by time
        sc = ax.scatter(x, y, z, c=times, cmap="viridis", s=20)  # type: ignore[misc]
        ax.plot(x, y, z, alpha=0.3, color="gray", linewidth=1)

        # Mark start and end
        ax.scatter(  # type: ignore[misc]
            [x[0]],
            [y[0]],
            [z[0]],
            color="green",
            s=100,
            marker="o",
            label="Start",
        )
        ax.scatter(  # type: ignore[misc]
            [x[-1]],
            [y[-1]],
            [z[-1]],
            color="red",
            s=100,
            marker="s",
            label="End",
        )

        ax.set_xlabel("X (m)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Y (m)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Z (m)", fontsize=10, fontweight="bold")  # type: ignore[attr-defined]
        ax.set_title("Club Head 3D Trajectory", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)
        fig.tight_layout()

    def plot_phase_diagram(self, fig: Figure, joint_idx: int = 0) -> None:
        """Plot phase diagram (angle vs angular velocity) for a joint.

        Args:
            fig: Matplotlib figure to plot on
            joint_idx: Index of joint to plot
        """
        times, positions = self._get_cached_series("joint_positions")
        _, velocities = self._get_cached_series("joint_velocities")

        # Convert to numpy arrays if needed
        positions = np.asarray(positions)
        velocities = np.asarray(velocities)

        # Check index bounds for both positions and velocities
        # Use simple index matching for now, but respect bounds to avoid crashes
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

        # Note: Ideally we would align indices here using _get_aligned_label logic inverse,
        # but plotting phase diagrams across misaligned q/v (e.g. quaternions) is complex.
        # We assume for now that if user asks for joint_idx, they know the indices align or are
        # aware of the structure. We just ensure safety.
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

        joint_name = self.get_joint_name(joint_idx)
        ax.set_xlabel(f"{joint_name} Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{joint_name} Velocity (deg/s)", fontsize=12, fontweight="bold")
        ax.set_title(f"Phase Diagram: {joint_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_torque_comparison(self, fig: Figure) -> None:
        """Plot comparison of all joint torques (stacked area or grouped bars).

        Args:
            fig: Matplotlib figure to plot on
        """
        times, torques = self._get_cached_series("joint_torques")

        # Convert to numpy array if needed
        torques = np.asarray(torques)

        if len(times) == 0 or len(torques) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        # Create stacked area plot
        ax = fig.add_subplot(111)

        # Separate positive and negative torques
        torques_pos = np.maximum(torques, 0)
        torques_neg = np.minimum(torques, 0)

        if torques.ndim < 2:
            labels = [self.get_joint_name(0)]
        else:
            labels = [self.get_joint_name(i) for i in range(torques.shape[1])]

        # Plot positive torques
        ax.stackplot(times, torques_pos.T, labels=labels, alpha=0.7)
        # Plot negative torques (same colors, no labels to avoid duplicate legend)
        # Reset color cycle by creating a new cycler with the same colors
        ax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
        ax.stackplot(times, torques_neg.T, alpha=0.7)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Torque (Nm)", fontsize=12, fontweight="bold")
        ax.set_title("Joint Torque Contributions", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.5)
        fig.tight_layout()

    def plot_frequency_analysis(
        self,
        fig: Figure,
        joint_idx: int = 0,
        signal_type: str = "velocity",
    ) -> None:
        """Plot frequency content (PSD) of a joint signal.

        Args:
            fig: Matplotlib figure
            joint_idx: Joint index
            signal_type: 'position', 'velocity', or 'torque'
        """
        if signal_type == "position":
            _, data = self._get_cached_series("joint_positions")
            ylabel = "PSD (rad²/Hz)"
            title = "Joint Position PSD"
        elif signal_type == "torque":
            _, data = self._get_cached_series("joint_torques")
            ylabel = "PSD (Nm²/Hz)"
            title = "Joint Torque PSD"
        else:  # velocity
            _, data = self._get_cached_series("joint_velocities")
            ylabel = "PSD ((rad/s)²/Hz)"
            title = "Joint Velocity PSD"

        data = np.asarray(data)
        if data.ndim < 2 or joint_idx >= data.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        signal_data = data[:, joint_idx]

        # Calculate sampling rate
        # Assuming consistent time
        times, _ = self._get_cached_series("joint_positions")
        if len(times) < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return

        dt = float(np.mean(np.diff(times)))
        fs = 1.0 / dt

        try:
            from shared.python import signal_processing

            freqs, psd = signal_processing.compute_psd(signal_data, fs)
        except ImportError:
            # Fallback
            from scipy import signal

            freqs, psd = signal.welch(signal_data, fs=fs)

        ax = fig.add_subplot(111)
        ax.semilogy(freqs, psd, color=self.colors["primary"], linewidth=2)

        joint_name = self.get_joint_name(joint_idx)
        ax.set_title(f"{title}: {joint_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both", linestyle="--")
        fig.tight_layout()

    def plot_spectrogram(
        self,
        fig: Figure,
        joint_idx: int = 0,
        signal_type: str = "velocity",
    ) -> None:
        """Plot spectrogram of a joint signal.

        Args:
            fig: Matplotlib figure
            joint_idx: Joint index
            signal_type: 'position', 'velocity', or 'torque'
        """
        if signal_type == "position":
            _, data = self._get_cached_series("joint_positions")
            title = "Joint Position Spectrogram"
        elif signal_type == "torque":
            _, data = self._get_cached_series("joint_torques")
            title = "Joint Torque Spectrogram"
        else:  # velocity
            _, data = self._get_cached_series("joint_velocities")
            title = "Joint Velocity Spectrogram"

        data = np.asarray(data)
        if data.ndim < 2 or joint_idx >= data.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        signal_data = data[:, joint_idx]

        # Calculate sampling rate
        times, _ = self._get_cached_series("joint_positions")
        if len(times) < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return

        dt = float(np.mean(np.diff(times)))
        fs = 1.0 / dt

        try:
            from shared.python import signal_processing

            f, t, Sxx = signal_processing.compute_spectrogram(signal_data, fs)
        except ImportError:
            # Fallback
            from scipy import signal

            f, t, Sxx = signal.spectrogram(signal_data, fs=fs)

        ax = fig.add_subplot(111)
        # Use pcolormesh for better visualization
        pcm = ax.pcolormesh(
            t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="inferno"
        )

        joint_name = self.get_joint_name(joint_idx)
        ax.set_title(f"{title}: {joint_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Frequency (Hz)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")

        # Add colorbar
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Power Spectral Density (dB)", rotation=270, labelpad=15)

        fig.tight_layout()

    def plot_summary_dashboard(self, fig: Figure) -> None:
        """Create a comprehensive dashboard with multiple subplots.

        Args:
            fig: Matplotlib figure to plot on
        """
        # Create 2x3 grid (more space for advanced metrics)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Club head speed (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        times, speeds = self._get_cached_series("club_head_speed")
        speeds = np.asarray(speeds)
        if len(times) > 0 and len(speeds) > 0:
            speeds_mph = speeds * 2.23694
            ax1.plot(times, speeds_mph, linewidth=2, color=self.colors["primary"])
            ax1.fill_between(
                times, 0, speeds_mph, alpha=0.3, color=self.colors["primary"]
            )
            max_speed = np.max(speeds_mph)
            ax1.set_title(
                f"Club Speed (Peak: {max_speed:.1f} mph)",
                fontsize=11,
                fontweight="bold",
            )
            ax1.set_xlabel("Time (s)", fontsize=9)
            ax1.set_ylabel("Speed (mph)", fontsize=9)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No club head data", ha="center", va="center")

        # 2. Energy (Top Center)
        ax2 = fig.add_subplot(gs[0, 1])
        times_ke, ke = self._get_cached_series("kinetic_energy")
        times_pe, pe = self._get_cached_series("potential_energy")
        if len(times_ke) > 0:
            ax2.plot(
                times_ke, ke, label="KE", linewidth=2, color=self.colors["primary"]
            )
            ax2.plot(
                times_pe, pe, label="PE", linewidth=2, color=self.colors["secondary"]
            )
            ax2.set_title("Energy", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Time (s)", fontsize=9)
            ax2.set_ylabel("Energy (J)", fontsize=9)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No energy data", ha="center", va="center")

        # 3. Angular Momentum (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        times_am, am = self._get_cached_series("angular_momentum")
        am = np.asarray(am)
        if len(times_am) > 0 and am.size > 0:
            # OPTIMIZATION: Explicit sqrt sum is faster than np.linalg.norm
            am_mag = np.sqrt(np.sum(am**2, axis=1))
            ax3.plot(
                times_am,
                am_mag,
                label="Mag",
                linewidth=2,
                color=self.colors["quaternary"],
            )
            ax3.set_title("Angular Momentum", fontsize=11, fontweight="bold")
            ax3.set_xlabel("Time (s)", fontsize=9)
            ax3.set_ylabel("L (kg m²/s)", fontsize=9)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No AM data", ha="center", va="center")

        # 4. Joint Angles (Bottom Left)
        ax4 = fig.add_subplot(gs[1, 0])
        times, positions = self._get_cached_series("joint_positions")
        positions = np.asarray(positions)
        if len(times) > 0 and len(positions) > 0 and positions.ndim >= 2:
            for idx in range(min(3, positions.shape[1])):  # Plot first 3 joints
                ax4.plot(
                    times,
                    np.rad2deg(positions[:, idx]),
                    label=self.get_joint_name(idx),
                    linewidth=2,
                )
            ax4.set_title("Joint Angles", fontsize=11, fontweight="bold")
            ax4.set_xlabel("Time (s)", fontsize=9)
            ax4.set_ylabel("Angle (deg)", fontsize=9)
            ax4.legend(fontsize=7, loc="best")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No position data", ha="center", va="center")

        # 5. CoP (Bottom Center)
        ax5 = fig.add_subplot(gs[1, 1])
        times_cop, cop = self._get_cached_series("cop_position")
        cop = np.asarray(cop)
        if len(times_cop) > 0 and cop.size > 0:
            ax5.scatter(cop[:, 0], cop[:, 1], c=times_cop, cmap="viridis", s=10)
            ax5.set_title("CoP Trajectory", fontsize=11, fontweight="bold")
            ax5.set_xlabel("X (m)", fontsize=9)
            ax5.set_ylabel("Y (m)", fontsize=9)
            ax5.axis("equal")
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No CoP data", ha="center", va="center")

        # 6. Torques (Bottom Right)
        ax6 = fig.add_subplot(gs[1, 2])
        times, torques = self._get_cached_series("joint_torques")
        torques = np.asarray(torques)
        if len(times) > 0 and len(torques) > 0 and torques.ndim >= 2:
            for idx in range(min(3, torques.shape[1])):
                ax6.plot(
                    times,
                    torques[:, idx],
                    label=self.get_joint_name(idx),
                    linewidth=2,
                )
            ax6.set_title("Joint Torques", fontsize=11, fontweight="bold")
            ax6.set_xlabel("Time (s)", fontsize=9)
            ax6.set_ylabel("Torque (Nm)", fontsize=9)
            ax6.legend(fontsize=7, loc="best")
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, "No torque data", ha="center", va="center")

        fig.suptitle(
            "Golf Swing Analysis Dashboard",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

    def plot_kinematic_sequence(
        self,
        fig: Figure,
        segment_indices: dict[str, int],
        analyzer_result: Any | None = None,
    ) -> None:
        """Plot kinematic sequence (normalized velocities).

        Visualizes proximal-to-distal sequencing.

        Args:
            fig: Matplotlib figure
            segment_indices: Map of segment names to joint indices
            analyzer_result: Optional KinematicSequenceResult object
        """
        times, velocities = self._get_cached_series("joint_velocities")
        # Convert to numpy array if needed
        velocities = np.asarray(velocities)

        if len(times) == 0 or len(velocities) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data recorded", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        # Plot normalized velocities for each segment
        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["quinary"],
        ]

        # Use analyzer result if available, otherwise raw calculation
        for i, (name, idx) in enumerate(segment_indices.items()):
            if idx < velocities.shape[1]:
                vel = np.abs(velocities[:, idx])
                # Normalize to peak
                max_vel = float(np.max(vel))
                if max_vel > 0:
                    vel_norm = vel / max_vel
                else:
                    vel_norm = vel

                color = colors[i % len(colors)]
                ax.plot(times, vel_norm, label=name, color=color, linewidth=2)

                # Mark peak
                if analyzer_result:
                    # Find matching peak in result
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
                        # Add order label
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

    def plot_work_loop(
        self,
        fig: Figure,
        joint_idx: int = 0,
        title: str | None = None,
    ) -> None:
        """Plot Work Loop (Torque vs Angle) for a joint.

        The area inside the loop represents the mechanical work done.

        Args:
            fig: Matplotlib figure
            joint_idx: Joint index
            title: Optional title
        """
        times, positions = self._get_cached_series("joint_positions")
        _, torques = self._get_cached_series("joint_torques")

        # Convert to numpy arrays
        positions = np.asarray(positions)
        torques = np.asarray(torques)

        if (
            len(times) == 0
            or positions.ndim < 2
            or joint_idx >= positions.shape[1]
            or torques.ndim < 2
            or joint_idx >= torques.shape[1]
        ):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        # Plot Torque vs Angle
        angle = np.rad2deg(positions[:, joint_idx])
        torque = torques[:, joint_idx]

        # Use time for color to show progression
        sc = ax.scatter(angle, torque, c=times, cmap="viridis", s=30, alpha=0.6)
        ax.plot(angle, torque, alpha=0.3, color="gray", linewidth=1)

        # Fill area to emphasize work (just a polygon fill)
        # We assume cyclic or start-to-end, fill gives a sense of magnitude
        ax.fill(angle, torque, alpha=0.1, color=self.colors["primary"])

        # Mark Start/End
        ax.scatter(
            angle[0],
            torque[0],
            c="green",
            s=100,
            label="Start",
            edgecolor="black",
            zorder=5,
        )
        ax.scatter(
            angle[-1],
            torque[-1],
            c="red",
            s=100,
            marker="s",
            label="End",
            edgecolor="black",
            zorder=5,
        )

        name = self.get_joint_name(joint_idx)
        ax.set_xlabel(f"{name} Angle (degrees)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{name} Torque (Nm)", fontsize=12, fontweight="bold")
        ax.set_title(title or f"Work Loop: {name}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="k", linestyle="-", alpha=0.3)
        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_x_factor_cycle(
        self,
        fig: Figure,
        shoulder_idx: int,
        hip_idx: int,
    ) -> None:
        """Plot X-Factor Cycle (Stretch-Shortening Cycle).

        Plots X-Factor Velocity vs X-Factor Angle.

        Args:
            fig: Matplotlib figure
            shoulder_idx: Shoulder/Torso joint index
            hip_idx: Hip/Pelvis joint index
        """
        try:
            # Check availability only
            import shared.python.statistical_analysis  # noqa: F401
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Analysis module missing", ha="center", va="center")
            return

        # We need an analyzer instance to compute X-Factor easily,
        # or just reimplement logic here. Reimplementing is cleaner for plotting module
        # to avoid circular imports or heavy deps, but X-Factor logic is specific.
        # Let's just calculate raw here.

        times, positions = self._get_cached_series("joint_positions")
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

        # Calculate X-Factor
        shoulder_rot = np.rad2deg(positions[:, shoulder_idx])
        hip_rot = np.rad2deg(positions[:, hip_idx])
        x_factor = shoulder_rot - hip_rot

        # Calculate Velocity
        dt = float(np.mean(np.diff(times)))
        if dt <= 0:
            dt = 0.01
        x_factor_vel = np.gradient(x_factor, dt)

        ax = fig.add_subplot(111)

        # Plot Phase Diagram
        sc = ax.scatter(x_factor, x_factor_vel, c=times, cmap="magma", s=30, alpha=0.6)
        ax.plot(x_factor, x_factor_vel, alpha=0.3, color="gray", linewidth=1)

        # Mark Peak Stretch (Max X-Factor)
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

    def plot_3d_phase_space(self, fig: Figure, joint_idx: int = 0) -> None:
        """Plot 3D phase space (Position vs Velocity vs Acceleration).

        Args:
            fig: Matplotlib figure
            joint_idx: Joint index
        """
        times, positions = self._get_cached_series("joint_positions")
        _, velocities = self._get_cached_series("joint_velocities")
        _, accelerations = self._get_cached_series("joint_accelerations")

        # Convert to numpy arrays
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
        sc = ax.scatter(pos, vel, acc, c=times, cmap="viridis", s=20)  # type: ignore[misc]
        ax.plot(pos, vel, acc, alpha=0.3, color="gray", linewidth=1)

        # Mark start
        ax.scatter(  # type: ignore[misc]
            [pos[0]],
            [vel[0]],
            [acc[0]],
            color="green",
            s=100,
            marker="o",
            label="Start",
        )

        joint_name = self.get_joint_name(joint_idx)
        ax.set_title(f"3D Phase Space: {joint_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Position (deg)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Velocity (deg/s)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Acceleration (deg/s²)", fontsize=10, fontweight="bold")  # type: ignore[attr-defined]
        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)
        fig.tight_layout()

    def plot_poincare_map_3d(
        self,
        fig: Figure,
        dimensions: list[tuple[str, int]],
        section_condition: tuple[str, int, float] = ("velocity", 0, 0.0),
        direction: str = "both",  # 'positive', 'negative', 'both'
        title: str | None = None,
    ) -> None:
        """Plot 3D Poincaré Map (Poincaré Section).

        Visualizes the intersection of the system trajectory with a defined lower-dimensional subspace.
        Points are plotted when the variable defined in `section_condition` crosses the specified value.

        Args:
            fig: Matplotlib figure
            dimensions: List of 3 (data_type, index) tuples for X, Y, Z axes.
                        data_type can be 'position', 'velocity', 'acceleration', 'torque'.
            section_condition: Tuple of (data_type, index, value) defining the section plane.
            direction: Crossing direction ('positive', 'negative', 'both').
            title: Optional title.
        """
        if len(dimensions) != 3:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "Must specify exactly 3 dimensions", ha="center", va="center"
            )
            return

        # Helper to get data array
        def get_data(dtype: str, idx: int) -> np.ndarray | None:
            if dtype == "position":
                _, d = self._get_cached_series("joint_positions")
            elif dtype == "velocity":
                _, d = self._get_cached_series("joint_velocities")
            elif dtype == "acceleration":
                _, d = self._get_cached_series("joint_accelerations")
            elif dtype == "torque":
                _, d = self._get_cached_series("joint_torques")
            else:
                return None
            d = np.asarray(d)
            if d.ndim > 1 and idx < d.shape[1]:
                return d[:, idx]
            return None

        # Get condition variable
        cond_type, cond_idx, cond_val = section_condition
        cond_data = get_data(cond_type, cond_idx)
        times, _ = self._get_cached_series("joint_positions")  # Time base

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

        # Find crossings
        # Shifted array
        diff = cond_data - cond_val
        crossings = []

        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] <= 0:  # Crossing detected
                # Check direction
                if diff[i] < diff[i + 1] and direction in ["positive", "both"]:
                    crossings.append(i)
                elif diff[i] > diff[i + 1] and direction in ["negative", "both"]:
                    crossings.append(i)

        if not crossings:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No section crossings found", ha="center", va="center")
            return

        # Interpolate state at crossings
        points = []
        point_times = []

        for i in crossings:
            # Linear interpolation factor
            # y = y0 + (y1-y0) * alpha
            # 0 = d0 + (d1-d0) * alpha => alpha = -d0 / (d1-d0)
            denom = diff[i + 1] - diff[i]
            if abs(denom) < 1e-9:
                alpha = 0.5
            else:
                alpha = -diff[i] / denom

            t_cross = times[i] + alpha * (times[i + 1] - times[i])
            point_times.append(t_cross)

            pt_coords = []
            for dim_type, dim_idx in dimensions:
                data = get_data(dim_type, dim_idx)
                if data is None:
                    pt_coords.append(0.0)
                else:
                    val = data[i] + alpha * (data[i + 1] - data[i])
                    # Convert rad to deg for angles?
                    # Generally yes for plots, but let's be consistent with type
                    if dim_type in ["position", "velocity", "acceleration"]:
                        val = np.rad2deg(val)
                    pt_coords.append(val)
            points.append(pt_coords)

        points_arr = np.array(points)

        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot
        sc = ax.scatter(  # type: ignore[misc]
            points_arr[:, 0],
            points_arr[:, 1],
            points_arr[:, 2],
            c=point_times,
            cmap="viridis",
            s=50,
            edgecolors="k",
        )

        # Labels
        labels = []
        for dt, di in dimensions:
            name = self.get_joint_name(di)
            unit = (
                "deg"
                if dt == "position"
                else "deg/s" if dt == "velocity" else "Nm" if dt == "torque" else ""
            )
            labels.append(f"{name} {dt[:3]} ({unit})")

        ax.set_xlabel(labels[0], fontsize=9, fontweight="bold")
        ax.set_ylabel(labels[1], fontsize=9, fontweight="bold")
        ax.set_zlabel(labels[2], fontsize=9, fontweight="bold")  # type: ignore[attr-defined]

        # Title
        cond_name = self.get_joint_name(cond_idx)
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
        """Plot Phase Space Reconstruction using Time-Delay Embedding (Takens' Theorem).

        Visualizes the attractor structure from a single scalar time series.

        Args:
            fig: Matplotlib figure
            joint_idx: Index of joint to analyze
            delay: Time delay (lag) in samples
            embedding_dim: Embedding dimension (2 or 3)
            signal_type: 'position', 'velocity', or 'torque'
        """
        # Get data
        if signal_type == "position":
            times, data_full = self._get_cached_series("joint_positions")
            data_full = np.rad2deg(np.asarray(data_full))
        elif signal_type == "velocity":
            times, data_full = self._get_cached_series("joint_velocities")
            data_full = np.rad2deg(np.asarray(data_full))
        else:
            times, data_full = self._get_cached_series("joint_torques")
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

        # Create embedded vectors
        # X(t) = [x(t), x(t-tau), x(t-2tau)]
        # Use appropriate slicing
        # Valid length: N - delay*(dim-1)
        valid_len = N - delay * (embedding_dim - 1)

        vectors = np.zeros((valid_len, embedding_dim))
        for d in range(embedding_dim):
            # For d=0, start at 0, end at valid_len
            # For d=1, start at delay, end at valid_len + delay
            # Wait, standard notation x(t), x(t+tau)...
            # Let's do x(t), x(t+tau), x(t+2tau)
            start = d * delay
            end = start + valid_len
            vectors[:, d] = x[start:end]

        # Time for color
        plot_times = times[:valid_len]

        if embedding_dim == 3:
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(  # type: ignore[misc]
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
            ax.set_zlabel(f"x(t+{2*delay})", fontsize=10)  # type: ignore[attr-defined]
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

        joint_name = self.get_joint_name(joint_idx)
        ax.set_title(
            f"Reconstructed Phase Space: {joint_name}\n(Lag={delay}, Dim={embedding_dim})",
            fontsize=12,
            fontweight="bold",
        )
        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)
        fig.tight_layout()

    def plot_muscle_synergies(
        self,
        fig: Figure,
        synergy_result: Any,  # Expects SynergyResult object
    ) -> None:
        """Plot extracted muscle synergies (Weights and Activations).

        Args:
            fig: Matplotlib figure
            synergy_result: SynergyResult object from MuscleSynergyAnalyzer
        """
        if not hasattr(synergy_result, "weights") or not hasattr(
            synergy_result, "activations"
        ):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Invalid SynergyResult object", ha="center", va="center")
            return

        n_synergies = synergy_result.n_synergies
        n_muscles = synergy_result.weights.shape[0]

        # Create grid: Left column (Weights), Right column (Activations)
        # One row per synergy
        gs = fig.add_gridspec(
            n_synergies, 2, width_ratios=[1, 2], hspace=0.4, wspace=0.3
        )

        times, _ = self._get_cached_series("joint_positions")
        # Ensure times matches activation length
        if len(times) != synergy_result.activations.shape[1]:
            # Resample times to match
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

            # 1. Weights (Bar chart)
            ax_w = fig.add_subplot(gs[i, 0])
            weights = synergy_result.weights[:, i]

            y_pos = np.arange(n_muscles)
            ax_w.barh(y_pos, weights, color=color, alpha=0.8)
            ax_w.set_yticks(y_pos)

            if i == n_synergies - 1:
                ax_w.set_xlabel("Weight", fontsize=9)

            ax_w.set_yticklabels(muscle_names, fontsize=8)
            ax_w.invert_yaxis()  # Top-down
            ax_w.set_title(f"Synergy {i+1} Weights", fontsize=10, fontweight="bold")
            ax_w.grid(True, axis="x", alpha=0.3)

            # 2. Activation (Time series)
            ax_h = fig.add_subplot(gs[i, 1])
            activation = synergy_result.activations[i, :]

            ax_h.plot(times, activation, color=color, linewidth=2)
            ax_h.fill_between(times, 0, activation, color=color, alpha=0.2)

            if i == n_synergies - 1:
                ax_h.set_xlabel("Time (s)", fontsize=10)

            ax_h.set_title(f"Synergy {i+1} Activation", fontsize=10, fontweight="bold")
            ax_h.grid(True, alpha=0.3)

        fig.suptitle(
            f"Muscle Synergies (VAF: {synergy_result.vaf*100:.1f}%)",
            fontsize=14,
            fontweight="bold",
        )
        # fig.tight_layout() # Handled by hspace/wspace roughly

    def plot_correlation_matrix(
        self,
        fig: Figure,
        data_type: str = "velocity",
    ) -> None:
        """Plot correlation matrix between joints.

        Args:
            fig: Matplotlib figure
            data_type: 'position', 'velocity', or 'torque'
        """
        if data_type == "position":
            _, data = self._get_cached_series("joint_positions")
            title = "Joint Position Correlation"
        elif data_type == "torque":
            _, data = self._get_cached_series("joint_torques")
            title = "Joint Torque Correlation"
        else:
            _, data = self._get_cached_series("joint_velocities")
            title = "Joint Velocity Correlation"

        data = np.asarray(data)
        if len(data) == 0 or data.ndim < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Compute correlation
        corr_matrix = np.corrcoef(data.T)

        ax = fig.add_subplot(111)
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

        # Add labels if fewer than 10 joints, otherwise just indices
        if data.shape[1] <= 10:
            labels = [self.get_joint_name(i) for i in range(data.shape[1])]
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
        else:
            ax.set_xlabel("Joint Index")
            ax.set_ylabel("Joint Index")

        # Add correlation values to heatmap cells
        # NOTE: Color calculation is vectorized using np.where, but ax.text() must
        # still be called individually (matplotlib limitation - no batch text API).
        # The optimization reduces nested loops to a single flat loop with pre-computed
        # positions and colors, avoiding redundant calculations inside the loop.
        if data.shape[1] <= 8:
            # Pre-compute all positions, values, and colors
            n = data.shape[1]
            i_coords, j_coords = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            i_flat = i_coords.ravel()
            j_flat = j_coords.ravel()
            values_flat = corr_matrix.ravel()

            # Vectorized color calculation (avoids per-element conditionals in loop)
            colors = np.where(np.abs(values_flat) < 0.5, "k", "w")

            # Render text annotations (matplotlib requires individual calls)
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

    def plot_swing_plane(self, fig: Figure) -> None:
        """Plot fitted swing plane and trajectory deviation.

        Args:
            fig: Matplotlib figure
        """
        times, positions = self._get_cached_series("club_head_position")

        if len(times) < 3 or len(positions) < 3:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Insufficient data for plane fitting",
                ha="center",
                va="center",
            )
            return

        # Ensure positions is a numpy array
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)

        analyzer = SwingPlaneAnalyzer()
        try:
            metrics = analyzer.analyze(positions)
        except ValueError as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            return

        ax = fig.add_subplot(111, projection="3d")

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # Plot trajectory color-coded by deviation from plane
        centroid = metrics.point_on_plane
        normal = metrics.normal_vector
        deviations = analyzer.calculate_deviation(positions, centroid, normal)

        # Plot trajectory
        sc = ax.scatter(
            x,
            y,
            zs=z,  # type: ignore[call-arg]
            c=np.abs(deviations),
            cmap="coolwarm",
            s=20,
            label="Trajectory",
        )

        # Plot plane
        # Create a grid around the centroid
        # Find bounds
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)

        # Create meshgrid
        margin = 0.5
        xx, yy = np.meshgrid(
            np.linspace(min_x - margin, max_x + margin, 10),
            np.linspace(min_y - margin, max_y + margin, 10),
        )

        # Plane equation: n . (p - c) = 0 => nx(x-cx) + ny(y-cy) + nz(z-cz) = 0
        # z = cz - (nx(x-cx) + ny(y-cy))/nz

        if abs(float(normal[2])) > 1e-6:
            zz = (
                centroid[2]
                - (normal[0] * (xx - centroid[0]) + normal[1] * (yy - centroid[1]))
                / normal[2]
            )
            ax.plot_surface(xx, yy, zz, alpha=0.2, color="cyan")  # type: ignore[attr-defined]
        else:
            # Vertical plane (rare for golf swing but possible)
            # Cannot plot as z = f(x,y), would need x = f(y,z) or similar
            # For visualization purposes, we skip drawing the surface if vertical
            pass  # pragma: no cover

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
        ax.set_title(
            f"Swing Plane Analysis\nSteepness: {metrics.steepness_deg:.1f}°, "
            f"RMSE: {metrics.rmse * 100:.1f} cm",
            fontsize=12,
            fontweight="bold",
        )

        fig.colorbar(sc, ax=ax, label="Deviation from Plane (m)", shrink=0.6)
        fig.tight_layout()

    def plot_angular_momentum(self, fig: Figure) -> None:
        """Plot Angular Momentum over time (Magnitude and Components).

        Args:
            fig: Matplotlib figure
        """
        times, am_data = self._get_cached_series("angular_momentum")
        am_data = np.asarray(am_data)

        if len(times) == 0 or am_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Angular Momentum Data", ha="center", va="center")
            return

        # Calculate magnitude
        # OPTIMIZATION: Explicit sqrt sum is faster than np.linalg.norm
        am_mag = np.sqrt(np.sum(am_data**2, axis=1))

        ax = fig.add_subplot(111)

        # Plot components
        ax.plot(
            times, am_data[:, 0], label="Lx", color=self.colors["secondary"], alpha=0.7
        )
        ax.plot(
            times, am_data[:, 1], label="Ly", color=self.colors["tertiary"], alpha=0.7
        )
        ax.plot(
            times, am_data[:, 2], label="Lz", color=self.colors["quaternary"], alpha=0.7
        )

        # Plot magnitude
        ax.plot(
            times,
            am_mag,
            label="Magnitude",
            color=self.colors["primary"],
            linewidth=2.5,
        )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Angular Momentum (kg m²/s)", fontsize=12, fontweight="bold")
        ax.set_title("System Angular Momentum", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_kinematic_sequence_bars(
        self,
        fig: Figure,
        segment_indices: dict[str, int],
        impact_time: float | None = None,
    ) -> None:
        """Plot kinematic sequence as a Gantt-style bar chart of peak times.

        Visualizes the 'gap' between peak velocities of segments.

        Args:
            fig: Matplotlib figure
            segment_indices: Map of segment names to joint indices
            impact_time: Optional impact time to mark as reference (0)
        """
        times, velocities = self._get_cached_series("joint_velocities")
        velocities = np.asarray(velocities)

        if len(times) == 0 or velocities.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        # Find peaks
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

        # Reference time (impact or first peak)
        ref_time = impact_time if impact_time is not None else peaks[-1]

        # Calculate relative times
        rel_times = np.array(peaks) - ref_time

        # Plot horizontal bars (using barh or stick plot)
        # We'll use scatter points with lines to the axis for clarity
        y_pos = np.arange(len(names))

        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["quinary"],
        ][: len(names)]

        # Draw lines from 0 to peak time? No, just point is better for instant
        # Or maybe bar from Start of Downswing to Peak?
        # Let's just do a Lollipop chart: Line from 0 (ref) to peak? No, ref is impact.
        # Line from left boundary to peak?
        # Simple Lollipop:
        ax.hlines(
            y=y_pos,
            xmin=min(0, np.min(rel_times) - 0.05),
            xmax=rel_times,
            color="gray",
            alpha=0.5,
        )
        ax.scatter(rel_times, y_pos, color=colors, s=100, zorder=3)

        # Add text labels for timing (ms)
        for i, t in enumerate(rel_times):
            ax.text(
                t,
                i + 0.15,
                f"{t*1000:.0f} ms",
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

        # Invert y axis so proximal (first in list) is at top?
        # Typically Pelvis (0) is top or bottom. Let's keep input order (top-down)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend()
        fig.tight_layout()

    def plot_cop_trajectory(self, fig: Figure) -> None:
        """Plot Center of Pressure trajectory (top-down view).

        Args:
            fig: Matplotlib figure
        """
        times, cop_data = self._get_cached_series("cop_position")
        cop_data = np.asarray(cop_data)

        if len(times) == 0 or cop_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No CoP Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        # Assuming X is lateral (target line) and Y is anterior-posterior (toe-heel)
        # or typical MuJoCo frame where X forward, Y left.
        # We'll plot X vs Y.
        x = cop_data[:, 0]
        y = cop_data[:, 1]

        # Scatter with time color
        sc = ax.scatter(x, y, c=times, cmap="viridis", s=30, zorder=2)
        ax.plot(x, y, color="gray", alpha=0.4, zorder=1)

        # Mark Start/End
        ax.scatter(x[0], y[0], c="green", s=100, label="Start", zorder=3)
        ax.scatter(x[-1], y[-1], c="red", s=100, marker="s", label="End", zorder=3)

        ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
        ax.set_title("Center of Pressure Trajectory", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axis("equal")  # Preserve aspect ratio

        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_cop_vector_field(self, fig: Figure, skip_steps: int = 5) -> None:
        """Plot CoP velocity vector field.

        Args:
            fig: Matplotlib figure
            skip_steps: Number of steps to skip for decluttering vectors
        """
        times, cop_data = self._get_cached_series("cop_position")
        cop_data = np.asarray(cop_data)

        if len(times) == 0 or cop_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No CoP Data", ha="center", va="center")
            return

        # Compute velocity
        dt = np.mean(np.diff(times)) if len(times) > 1 else 1.0
        vel = np.gradient(cop_data, dt, axis=0)

        ax = fig.add_subplot(111)

        # Downsample
        x = cop_data[::skip_steps, 0]
        y = cop_data[::skip_steps, 1]
        u = vel[::skip_steps, 0]
        v = vel[::skip_steps, 1]
        t = times[::skip_steps]

        # Quiver plot
        q = ax.quiver(x, y, u, v, t, cmap="viridis", scale_units="xy", angles="xy")

        # Plot trajectory line
        ax.plot(cop_data[:, 0], cop_data[:, 1], "k-", alpha=0.2)

        ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Center of Pressure Velocity Field", fontsize=14, fontweight="bold"
        )
        ax.axis("equal")
        fig.colorbar(q, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_recurrence_plot(
        self,
        fig: Figure,
        recurrence_matrix: np.ndarray,
        title: str = "Recurrence Plot",
    ) -> None:
        """Plot Recurrence Plot (binary matrix).

        Args:
            fig: Matplotlib figure
            recurrence_matrix: Binary matrix (N, N)
            title: Title of the plot
        """
        if recurrence_matrix.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Recurrence Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        # Plot binary matrix
        # Use black points for recurrence (1), white for non-recurrence (0)
        # cmap="binary" uses 0=white, 1=black? No, usually 0=white, 255=black.
        # But if values are 0/1, we need to check map.
        # "Greys" is usually safe (0=white, 1=black).
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

    def plot_activation_heatmap(
        self,
        fig: Figure,
        data_type: str = "torque",
    ) -> None:
        """Plot activation heatmap (Joints vs Time).

        Visualizes magnitude of torque or power for all joints over time.

        Args:
            fig: Matplotlib figure
            data_type: 'torque' or 'power'
        """
        if data_type == "power":
            times, data = self._get_cached_series("actuator_powers")
            title = "Actuator Power Activation"
            cbar_label = "Power (W)"
        else:
            times, data = self._get_cached_series("joint_torques")
            title = "Joint Torque Activation"
            cbar_label = "Torque (Nm)"

        data = np.asarray(data)

        if len(times) == 0 or data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Use magnitude for heatmap? Or signed?
        # Signed is useful to see direction, but "Activation" usually implies magnitude.
        # Let's use signed with RdBu colormap to show direction (flexion/extension)
        # or viridis for magnitude.
        # "Muscle Activation" is usually 0-1 magnitude.
        # Torque can be negative.
        # Let's use signed to show effort direction, centered at 0.

        ax = fig.add_subplot(111)

        # Transpose so Time is X-axis, Joints are Y-axis
        # data is (N_samples, N_joints) -> (N_joints, N_samples)
        heatmap_data = data.T

        # Determine limits for symmetric colorbar
        max_val = np.max(np.abs(heatmap_data))
        if max_val < 1e-6:
            max_val = 1.0

        # Create meshgrid for pcolormesh
        # Time edges
        if len(times) > 1:
            dt = times[1] - times[0]
            time_edges = np.concatenate(
                (
                    [times[0] - dt / 2],
                    times[:-1] + np.diff(times) / 2,
                    [times[-1] + dt / 2],
                )
            )
        else:
            time_edges = np.array([times[0] - 0.5, times[0] + 0.5])

        # Joint edges
        joint_edges = np.arange(heatmap_data.shape[0] + 1)

        # Plot
        im = ax.pcolormesh(
            time_edges,
            joint_edges,
            heatmap_data,
            cmap="RdBu_r",
            vmin=-max_val,
            vmax=max_val,
            shading="flat",
        )

        # Set y-ticks to joint names
        ax.set_yticks(np.arange(heatmap_data.shape[0]) + 0.5)
        labels = [self.get_joint_name(i) for i in range(heatmap_data.shape[0])]
        ax.set_yticklabels(labels)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(False)  # Heatmap doesn't need grid usually

        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()

    def plot_phase_space_density(
        self,
        fig: Figure,
        joint_idx: int = 0,
        bins: int = 50,
    ) -> None:
        """Plot 2D Phase Space Density (Histogram).

        Useful for seeing where the system spends most time in phase space.

        Args:
            fig: Matplotlib figure
            joint_idx: Index of joint
            bins: Number of histogram bins
        """
        times, positions = self._get_cached_series("joint_positions")
        _, velocities = self._get_cached_series("joint_velocities")

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

        joint_name = self.get_joint_name(joint_idx)
        ax.set_xlabel(f"{joint_name} Angle (deg)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{joint_name} Velocity (deg/s)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Phase Space Density: {joint_name}", fontsize=14, fontweight="bold"
        )

        fig.colorbar(h[3], ax=ax, label="Count")
        fig.tight_layout()

    def plot_grf_butterfly_diagram(
        self,
        fig: Figure,
        skip_steps: int = 5,
        scale: float = 0.001,
    ) -> None:
        """Plot Ground Reaction Force 'Butterfly Diagram'.

        Visualizes GRF vectors originating from the Center of Pressure path.

        Args:
            fig: Matplotlib figure
            skip_steps: Step interval for plotting vectors (to reduce clutter)
            scale: Scale factor for force vectors (m/N)
        """
        try:
            times, cop_data = self._get_cached_series("cop_position")
            _, grf_data = self._get_cached_series("ground_forces")
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

        # Handle 3D GRF (Fx, Fy, Fz) vs 6D Wrench
        # We need Fx, Fy, Fz.
        if grf_data.shape[1] >= 3:
            fx = grf_data[:, 0]
            fy = grf_data[:, 1]
            fz = grf_data[:, 2]
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Invalid GRF dimensions", ha="center", va="center")
            return

        # Plot in 3D
        ax = fig.add_subplot(111, projection="3d")

        # Plot CoP path on ground (Z=0)
        # Assuming CoP is 3D (x, y, z) or 2D (x, y)
        cx = cop_data[:, 0]
        cy = cop_data[:, 1]
        cz = cop_data[:, 2] if cop_data.shape[1] > 2 else np.zeros_like(cx)

        ax.plot(cx, cy, cz, color="black", linewidth=2, label="CoP Path")

        # Plot Vectors
        # Downsample
        indices = range(0, len(times), skip_steps)
        for i in indices:
            # Origin
            ox, oy, oz = cx[i], cy[i], cz[i]
            # Vector components
            vx, vy, vz = fx[i], fy[i], fz[i]

            # Draw line
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
        ax.set_zlabel("Force (scaled)", fontsize=10, fontweight="bold")  # type: ignore[attr-defined]
        ax.set_title("GRF Butterfly Diagram", fontsize=14, fontweight="bold")

        # Determine axis limits to show vectors
        all_x = np.concatenate([cx, cx + fx * scale])
        all_y = np.concatenate([cy, cy + fy * scale])
        all_z = np.concatenate([cz, cz + fz * scale])

        ax.set_xlim(np.min(all_x), np.max(all_x))
        ax.set_ylim(np.min(all_y), np.max(all_y))
        ax.set_zlim(np.min(all_z), np.max(all_z))  # type: ignore[attr-defined]

        fig.tight_layout()

    def plot_angular_momentum_3d(self, fig: Figure) -> None:
        """Plot 3D trajectory of the Angular Momentum vector.

        Visualizes the tip of the angular momentum vector over time.

        Args:
            fig: Matplotlib figure
        """
        times, am_data = self._get_cached_series("angular_momentum")
        am_data = np.asarray(am_data)

        if len(times) == 0 or am_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Angular Momentum Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111, projection="3d")

        lx = am_data[:, 0]
        ly = am_data[:, 1]
        lz = am_data[:, 2]

        # Plot trajectory of the tip
        sc = ax.scatter(lx, ly, lz, c=times, cmap="viridis", s=20)  # type: ignore[misc]
        ax.plot(lx, ly, lz, color="gray", alpha=0.3)

        # Draw vector from origin for current/max?
        # Maybe just draw a few representative vectors
        # OPTIMIZATION: Explicit sqrt sum is faster than np.linalg.norm
        max_idx = np.argmax(np.sum(am_data**2, axis=1))  # No need for sqrt for argmax

        # Draw Peak Vector
        ax.plot(
            [0, lx[max_idx]],
            [0, ly[max_idx]],
            [0, lz[max_idx]],
            color="red",
            linewidth=2,
            label="Peak L",
        )

        # Mark Origin
        ax.scatter([0], [0], zs=[0], color="black", s=50, marker="o")

        ax.set_xlabel("Lx (kg m²/s)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Ly (kg m²/s)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Lz (kg m²/s)", fontsize=10, fontweight="bold")  # type: ignore[attr-defined]
        ax.set_title("3D Angular Momentum Trajectory", fontsize=14, fontweight="bold")
        ax.legend()

        fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)
        fig.tight_layout()

    def plot_stability_diagram(self, fig: Figure) -> None:
        """Plot Stability Diagram (CoM vs CoP on Ground Plane).

        Visualizes the relationship between Center of Mass projection and Center of Pressure.

        Args:
            fig: Matplotlib figure
        """
        try:
            times, cop_data = self._get_cached_series("cop_position")
            _, com_data = self._get_cached_series("com_position")
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

        # Plot CoP Path
        ax.plot(
            cop_data[:, 0],
            cop_data[:, 1],
            color=self.colors["secondary"],
            linewidth=2,
            label="CoP",
        )

        # Plot CoM Projection (X, Y)
        ax.plot(
            com_data[:, 0],
            com_data[:, 1],
            color=self.colors["primary"],
            linewidth=2,
            linestyle="--",
            label="CoM (Proj)",
        )

        # Connect CoP and CoM at intervals to show "Lean" vector
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

    def plot_radar_chart(
        self,
        fig: Figure,
        metrics: dict[str, float],
        title: str = "Swing DNA",
        ax: Axes | None = None,
    ) -> None:
        """Plot a radar chart of swing metrics.

        Args:
            fig: Matplotlib figure
            metrics: Dictionary of metrics. Values should be normalized to [0, 1] or [0, 100].
            title: Chart title
            ax: Optional Axes object. If None, creates new subplot(111, polar=True).
        """
        labels = list(metrics.keys())
        values = list(metrics.values())
        num_vars = len(labels)

        if ax is None:
            ax = fig.add_subplot(111, polar=True)

        if num_vars < 3:
            ax.text(
                0.5,
                0.5,
                "Need at least 3 metrics for radar chart",
                ha="center",
                va="center",
            )
            return

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is a circle, so we need to "close the loop"
        values += values[:1]
        angles += angles[:1]
        labels += labels[:1]

        ax.plot(angles, values, color=self.colors["primary"], linewidth=2)
        ax.fill(angles, values, color=self.colors["primary"], alpha=0.25)

        # Draw labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels[:-1])

        # Draw grid
        ax.grid(True, alpha=0.3)

        ax.set_title(title, size=15, color=self.colors["primary"], y=1.1)
        fig.tight_layout()

    def plot_power_flow(self, fig: Figure) -> None:
        """Plot power flow (stacked bar) over time.

        Args:
            fig: Matplotlib figure
        """
        times, powers = self._get_cached_series("actuator_powers")
        powers = np.asarray(powers)

        if len(times) == 0 or powers.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Power Data", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        # Prepare data for stacked bar
        # Group by positive and negative power to show generation vs absorption

        (times[-1] - times[0]) / len(times) * 1.0  # Continuous-ish

        # Stackplot is better for continuous time
        # Separate positive and negative
        pos_powers = np.maximum(powers, 0)
        neg_powers = np.minimum(powers, 0)

        labels = [self.get_joint_name(i) for i in range(powers.shape[1])]

        ax.stackplot(times, pos_powers.T, labels=labels, alpha=0.7)
        # Reset color cycle by creating a new cycler with default colors
        from cycler import cycler

        default_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        ax.set_prop_cycle(cycler("color", default_colors))
        ax.stackplot(times, neg_powers.T, alpha=0.7)

        ax.axhline(0, color="k", linewidth=1)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Power Flow (Generation/Absorption)", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
        fig.tight_layout()

    def plot_joint_power_curves(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
    ) -> None:
        """Plot joint power curves with generation/absorption regions.

        Args:
            fig: Matplotlib figure
            joint_indices: List of joint indices to plot (None = all)
        """
        # Prefer using joint_torques and joint_velocities if available to compute power
        # rather than actuator_powers which might be pre-computed differently.
        times, torques = self._get_cached_series("joint_torques")
        _, velocities = self._get_cached_series("joint_velocities")

        torques = np.asarray(torques)
        velocities = np.asarray(velocities)

        if len(times) == 0 or torques.size == 0 or velocities.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        if joint_indices is None:
            joint_indices = list(range(min(torques.shape[1], velocities.shape[1])))

        for idx in joint_indices:
            if idx < torques.shape[1] and idx < velocities.shape[1]:
                power = torques[:, idx] * velocities[:, idx]
                label = self._get_aligned_label(idx, torques.shape[1])

                line = ax.plot(times, power, label=label, linewidth=2)
                color = line[0].get_color()

                # Fill generation (positive) and absorption (negative)
                ax.fill_between(
                    times,
                    power,
                    0,
                    where=(power >= 0),
                    alpha=0.2,
                    color=color,
                    interpolate=True,
                )
                ax.fill_between(
                    times,
                    power,
                    0,
                    where=(power < 0),
                    alpha=0.1,
                    color=color,
                    hatch="///",
                    interpolate=True,
                )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Joint Power: Generation (+) vs Absorption (-)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="k", linewidth=1)
        fig.tight_layout()

    def plot_impulse_accumulation(
        self,
        fig: Figure,
        joint_indices: list[int] | None = None,
    ) -> None:
        """Plot cumulative impulse (integrated torque) over time.

        Args:
            fig: Matplotlib figure
            joint_indices: List of joint indices to plot
        """
        times, torques = self._get_cached_series("joint_torques")
        torques = np.asarray(torques)

        if len(times) == 0 or torques.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        ax = fig.add_subplot(111)

        if joint_indices is None:
            joint_indices = list(range(torques.shape[1]))

        dt = np.mean(np.diff(times)) if len(times) > 1 else 0.0

        if dt > 0:
            from scipy.integrate import cumulative_trapezoid

            for idx in joint_indices:
                if idx < torques.shape[1]:
                    impulse = cumulative_trapezoid(torques[:, idx], dx=dt, initial=0)
                    label = self._get_aligned_label(idx, torques.shape[1])
                    ax.plot(times, impulse, label=label, linewidth=2)

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative Impulse (Nms)", fontsize=12, fontweight="bold")
        ax.set_title("Angular Impulse Accumulation", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="k", linewidth=1)
        fig.tight_layout()

    def plot_induced_acceleration(
        self,
        fig: Figure,
        source_name: str,
        joint_idx: int | None = None,
        breakdown_mode: bool = False,
    ) -> None:
        """Plot induced accelerations.

        Args:
            fig: Matplotlib figure
            source_name: Name of the force source (or 'breakdown' for all components)
            joint_idx: Optional joint index to plot (plots magnitude or all if None)
            breakdown_mode: If True, plots Gravity, Velocity, and Total components
        """
        ax = fig.add_subplot(111)

        if breakdown_mode:
            # Plot Gravity, Velocity, Control, Total for one joint
            if joint_idx is None:
                joint_idx = 0  # Default to 0 if not specified

            # Fetch components
            components = ["gravity", "velocity", "total"]
            linestyles = ["--", "-.", "-"]
            labels = ["Gravity", "Velocity (Coriolis)", "Total (Passive)"]
            colors = [
                self.colors["secondary"],
                self.colors["tertiary"],
                "black",
            ]

            has_data = False
            times = np.array([])

            for comp, ls, lbl, clr in zip(
                components, linestyles, labels, colors, strict=True
            ):
                try:
                    times, acc = self.recorder.get_induced_acceleration_series(comp)
                    if len(times) > 0 and acc.size > 0 and joint_idx < acc.shape[1]:
                        ax.plot(
                            times,
                            acc[:, joint_idx],
                            label=lbl,
                            linestyle=ls,
                            color=clr,
                            linewidth=2 if comp == "total" else 1.5,
                        )
                        has_data = True
                except (AttributeError, KeyError):
                    continue

            # Attempt to fetch control/specific if available
            try:
                times_c, acc_c = self.recorder.get_induced_acceleration_series(
                    "control"
                )
                if len(times_c) > 0 and acc_c.size > 0 and joint_idx < acc_c.shape[1]:
                    ax.plot(
                        times_c,
                        acc_c[:, joint_idx],
                        label="Control",
                        linestyle=":",
                        color=self.colors["quaternary"],
                        linewidth=1.5,
                    )
            except (AttributeError, KeyError):
                pass

            if not has_data:
                ax.text(
                    0.5,
                    0.5,
                    "No induced acceleration breakdown data",
                    ha="center",
                    va="center",
                )
                return

            joint_name = self.get_joint_name(joint_idx)
            ax.set_title(
                f"Induced Accelerations Breakdown: {joint_name}",
                fontsize=14,
                fontweight="bold",
            )

        else:
            # Single source mode
            try:
                times, acc = self.recorder.get_induced_acceleration_series(source_name)
            except (AttributeError, KeyError):
                ax.text(
                    0.5,
                    0.5,
                    f"No induced acceleration data for {source_name}",
                    ha="center",
                    va="center",
                )
                return

            if len(times) == 0 or acc.size == 0:
                ax.text(
                    0.5, 0.5, f"No data for {source_name}", ha="center", va="center"
                )
                return

            if joint_idx is not None:
                # Plot specific joint
                if joint_idx < acc.shape[1]:
                    ax.plot(
                        times,
                        acc[:, joint_idx],
                        label=self.get_joint_name(joint_idx),
                        linewidth=2,
                        color=self.colors["primary"],
                    )
                    ax.set_ylabel(
                        f"Joint {joint_idx} Acceleration (rad/s²)",
                        fontsize=12,
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"Joint index {joint_idx} out of bounds",
                        ha="center",
                        va="center",
                    )
                    return
            else:
                # Plot L2 norm for summary
                # OPTIMIZATION: Explicit sqrt sum is faster than np.linalg.norm
                norm = np.sqrt(np.sum(acc**2, axis=1))
                ax.plot(
                    times,
                    norm,
                    label="L2 Norm",
                    linewidth=2,
                    color=self.colors["primary"],
                )
                ax.set_ylabel(
                    "Acceleration Magnitude (rad/s²)", fontsize=12, fontweight="bold"
                )
            ax.set_title(
                f"Induced Acceleration: {source_name}", fontsize=14, fontweight="bold"
            )

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_club_induced_acceleration(
        self,
        fig: Figure,
        breakdown_mode: bool = True,
    ) -> None:
        """Plot club head task-space induced accelerations.

        Shows the contribution of Gravity, Velocity (Kinematic), and Control
        to the linear acceleration of the club head.

        Args:
            fig: Matplotlib figure
            breakdown_mode: If True, plots all components.
        """
        ax = fig.add_subplot(111)

        # Components to check
        components = ["gravity", "velocity", "control", "constraint", "total"]
        labels = [
            "Gravity",
            "Velocity (Kinematic)",
            "Control (Muscle)",
            "Constraint",
            "Total",
        ]
        colors = [
            self.colors["secondary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["quinary"],
            "black",
        ]
        styles = ["--", "-.", ":", "--", "-"]

        has_data = False

        # Check if recorder has method
        if not hasattr(self.recorder, "get_club_induced_acceleration_series"):
            ax.text(
                0.5,
                0.5,
                "Recorder does not support club induced accel",
                ha="center",
                va="center",
            )
            return

        for comp, label, color, style in zip(
            components, labels, colors, styles, strict=False
        ):
            times, acc_vec = self.recorder.get_club_induced_acceleration_series(comp)

            if len(times) > 0 and acc_vec.size > 0:
                # Plot Magnitude
                # OPTIMIZATION: Explicit sqrt sum is faster than np.linalg.norm
                mag = np.sqrt(np.sum(acc_vec**2, axis=1))

                # Check if it's mostly zero
                if np.max(mag) > 1e-4 or comp == "total":
                    ax.plot(
                        times,
                        mag,
                        label=label,
                        color=color,
                        linestyle=style,
                        linewidth=2,
                    )
                    has_data = True

        if not has_data:
            ax.text(
                0.5, 0.5, "No Club Induced Acceleration Data", ha="center", va="center"
            )
            return

        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Acceleration Magnitude (m/s²)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Club Head Acceleration Contributors", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_counterfactual_comparison(
        self, fig: Figure, cf_name: str, metric_idx: int = 0
    ) -> None:
        """Plot counterfactual data against actual data.

        Args:
            fig: Matplotlib figure
            cf_name: Name of counterfactual (e.g. 'ztcf', 'zvcf', or 'dual')
            metric_idx: Index of metric to compare (e.g. joint index)
        """
        # Special mode for Dual ZTCF/ZVCF plot
        if cf_name == "dual":
            self._plot_counterfactual_dual(fig, metric_idx)
            return

        # Standard comparison (Actual vs CF)
        # Get actual data (assume joint positions for now as primary comparison)
        times_actual, actual_data = self._get_cached_series("joint_positions")
        actual = np.asarray(actual_data)

        try:
            times_cf, cf_data_raw = self.recorder.get_counterfactual_series(cf_name)
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

        # Plot Actual
        if actual.ndim > 1 and metric_idx < actual.shape[1]:
            ax.plot(
                times_actual,
                np.rad2deg(actual[:, metric_idx]),
                label="Actual",
                linewidth=2,
                color="black",
            )

        # Plot Counterfactual
        # Ensure dimensions match
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
            times_z, ztcf = self.recorder.get_counterfactual_series("ztcf_accel")
            times_v, zvcf = self.recorder.get_counterfactual_series("zvcf_torque")
        except (AttributeError, KeyError):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Counterfactual data missing", ha="center", va="center")
            return

        if len(times_z) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No CF data", ha="center", va="center")
            return

        ax1 = fig.add_subplot(111)

        # Plot ZTCF (Accel)
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

        # Plot ZVCF (Torque) on twin axis
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

            # Legend
            lns = line1 + line2
            labs = [str(line.get_label()) for line in lns]
            ax1.legend(lns, labs, loc="upper left")
        else:
            ax1.legend(loc="best")

        joint_name = self.get_joint_name(joint_idx)
        ax1.set_title(
            f"Counterfactuals (ZTCF vs ZVCF): {joint_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()

    def plot_dynamic_correlation(
        self,
        fig: Figure,
        joint_idx_1: int,
        joint_idx_2: int,
        window_size: int = 20,
    ) -> None:
        """Plot Rolling Correlation between two joint velocities.

        Args:
            fig: Matplotlib figure
            joint_idx_1: First joint index
            joint_idx_2: Second joint index
            window_size: Window size for rolling correlation
        """
        try:
            from shared.python.statistical_analysis import StatisticalAnalyzer
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Analysis module missing", ha="center", va="center")
            return

        # Fetch data
        times, positions = self._get_cached_series("joint_positions")
        _, velocities = self._get_cached_series("joint_velocities")
        _, torques = self._get_cached_series("joint_torques")

        if len(times) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        # Initialize analyzer
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

        name1 = self.get_joint_name(joint_idx_1)
        name2 = self.get_joint_name(joint_idx_2)

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
        """Plot trajectory in synergy space (Activation 1 vs Activation 2).

        Args:
            fig: Matplotlib figure
            synergy_result: SynergyResult object
            dim1: Index of first synergy (X-axis)
            dim2: Index of second synergy (Y-axis)
        """
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

        times, _ = self._get_cached_series("joint_positions")
        # Match lengths
        n_samples = min(len(times), activations.shape[1])
        act1 = activations[dim1, :n_samples]
        act2 = activations[dim2, :n_samples]
        plot_times = times[:n_samples]

        ax = fig.add_subplot(111)
        sc = ax.scatter(act1, act2, c=plot_times, cmap="viridis", s=30, alpha=0.8)
        ax.plot(act1, act2, color="gray", alpha=0.3, linewidth=1)

        # Mark Start
        ax.scatter(act1[0], act2[0], color="green", s=100, label="Start")
        ax.scatter(act1[-1], act2[-1], color="red", s=100, marker="s", label="End")

        ax.set_xlabel(f"Synergy {dim1+1} Activation", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Synergy {dim2+1} Activation", fontsize=12, fontweight="bold")
        ax.set_title("Synergy Space Trajectory", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.colorbar(sc, ax=ax, label="Time (s)")
        fig.tight_layout()

    def plot_3d_vector_field(
        self,
        fig: Figure,
        vector_name: str,
        position_name: str,
        skip_steps: int = 5,
        scale: float = 0.1,
    ) -> None:
        """Plot 3D vector field along a trajectory.

        Generalized version of butterfly diagram for any vector series.

        Args:
            fig: Matplotlib figure
            vector_name: Name of vector field (e.g., 'angular_momentum', 'ground_forces')
            position_name: Name of position field (e.g., 'com_position', 'cop_position')
            skip_steps: Step interval
            scale: Vector scale factor
        """
        try:
            times, vectors = self._get_cached_series(vector_name)
            _, positions = self._get_cached_series(position_name)
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

        # Plot Trajectory
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        ax.plot(x, y, z, color="k", alpha=0.3, linewidth=1, label=f"{position_name}")

        # Plot Vectors
        indices = range(0, len(times), skip_steps)
        for i in indices:
            u, v, w = vectors[i, 0], vectors[i, 1], vectors[i, 2]
            px, py, pz = x[i], y[i], z[i]

            # Draw vector
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
        ax.set_zlabel("Z", fontweight="bold")  # type: ignore
        fig.tight_layout()

    def plot_local_stability(
        self,
        fig: Figure,
        joint_idx: int = 0,
        embedding_dim: int = 3,
        tau: int = 5,
    ) -> None:
        """Plot local divergence rate (Local Stability) over time.

        Args:
            fig: Matplotlib figure
            joint_idx: Joint index
            embedding_dim: Embedding dimension
            tau: Time lag
        """
        try:
            from shared.python.statistical_analysis import StatisticalAnalyzer
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Analysis module missing", ha="center", va="center")
            return

        # Fetch data
        times, positions = self._get_cached_series("joint_positions")
        _, velocities = self._get_cached_series("joint_velocities")

        if len(times) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        analyzer = StatisticalAnalyzer(
            times=np.asarray(times),
            joint_positions=np.asarray(positions),
            joint_velocities=np.asarray(velocities),
            joint_torques=np.zeros_like(positions),  # Dummy
        )

        try:
            ld_times, ld_rates = analyzer.compute_local_divergence_rate(
                joint_idx=joint_idx,
                tau=tau,
                dim=embedding_dim,
                window=tau * 2,  # Theiler window
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

        # Positive divergence rate -> Local Instability
        # Negative divergence rate -> Local Stability (Converging)
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

        name = self.get_joint_name(joint_idx)
        ax.set_title(
            f"Local Stability (Divergence Rate): {name}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Divergence Rate (1/s)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        fig.tight_layout()

    def plot_wavelet_scalogram(
        self,
        fig: Figure,
        joint_idx: int,
        signal_type: str = "velocity",
        freq_range: tuple[float, float] = (1.0, 50.0),
        title_prefix: str = "",
    ) -> None:
        """Plot Continuous Wavelet Transform (CWT) scalogram for a joint signal.

        Provides time-frequency analysis of the signal, revealing how frequency
        content evolves over the swing.

        Args:
            fig: Matplotlib figure
            joint_idx: Joint index to analyze
            signal_type: 'position', 'velocity', or 'torque'
            freq_range: (min_freq, max_freq) in Hz
            title_prefix: Optional prefix for titel
        """
        try:
            from shared.python import signal_processing
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "Signal Processing module missing", ha="center", va="center"
            )
            return

        # Fetch data
        if signal_type == "position":
            times, data = self._get_cached_series("joint_positions")
            title_prefix = title_prefix or "Position"
        elif signal_type == "torque":
            times, data = self._get_cached_series("joint_torques")
            title_prefix = title_prefix or "Torque"
        else:
            times, data = self._get_cached_series("joint_velocities")
            title_prefix = title_prefix or "Velocity"

        data = np.asarray(data)

        if len(times) == 0 or data.ndim < 2 or joint_idx >= data.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        signal_data = data[:, joint_idx]
        dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.01
        fs = 1.0 / dt

        # Compute CWT
        try:
            freqs, _, cwt_matrix = signal_processing.compute_cwt(
                signal_data, fs, freq_range=freq_range
            )
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"CWT Error: {e}", ha="center", va="center")
            return

        # Plot Power (|CWT|^2)
        power = np.abs(cwt_matrix) ** 2

        ax = fig.add_subplot(111)
        # Use pcolormesh: Time vs Frequency
        T, F = np.meshgrid(times, freqs)

        pcm = ax.pcolormesh(T, F, power, shading="auto", cmap="jet")

        joint_name = self.get_joint_name(joint_idx)
        ax.set_title(
            f"Wavelet Scalogram ({title_prefix}): {joint_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Frequency (Hz)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_yscale("log")

        # Adjust Y-ticks for log scale
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        from matplotlib.ticker import ScalarFormatter

        ax.yaxis.set_major_formatter(ScalarFormatter())

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Power", rotation=270, labelpad=15)
        fig.tight_layout()

    def plot_cross_wavelet(
        self,
        fig: Figure,
        joint_idx_1: int,
        joint_idx_2: int,
        signal_type: str = "velocity",
        freq_range: tuple[float, float] = (1.0, 50.0),
    ) -> None:
        """Plot Cross Wavelet Transform (XWT) between two signals.

        Shows common power and relative phase (arrows).

        Args:
            fig: Matplotlib figure
            joint_idx_1: First joint index
            joint_idx_2: Second joint index
            signal_type: 'position', 'velocity', or 'torque'
            freq_range: (min_freq, max_freq)
        """
        try:
            from shared.python import signal_processing
        except ImportError:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "Signal Processing module missing", ha="center", va="center"
            )
            return

        # Fetch data
        if signal_type == "position":
            times, data = self._get_cached_series("joint_positions")
        elif signal_type == "torque":
            times, data = self._get_cached_series("joint_torques")
        else:
            times, data = self._get_cached_series("joint_velocities")

        data = np.asarray(data)
        if len(times) == 0 or data.ndim < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        if joint_idx_1 >= data.shape[1] or joint_idx_2 >= data.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Joint index out of bounds", ha="center", va="center")
            return

        s1 = data[:, joint_idx_1]
        s2 = data[:, joint_idx_2]
        dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.01
        fs = 1.0 / dt

        # Compute XWT
        try:
            freqs, _, xwt_matrix = signal_processing.compute_xwt(
                s1, s2, fs, freq_range=freq_range
            )
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"XWT Error: {e}", ha="center", va="center")
            return

        # Cross Power
        power = np.abs(xwt_matrix)
        # Phase difference
        phase = np.angle(xwt_matrix)

        ax = fig.add_subplot(111)

        # Plot Power Heatmap
        T, F = np.meshgrid(times, freqs)
        pcm = ax.pcolormesh(T, F, power, shading="auto", cmap="jet")

        # Plot Phase Arrows (decimated)
        t_skip = max(1, len(times) // 30)
        f_skip = max(1, len(freqs) // 20)

        ax.quiver(
            T[::f_skip, ::t_skip],
            F[::f_skip, ::t_skip],
            np.cos(phase[::f_skip, ::t_skip]),
            np.sin(phase[::f_skip, ::t_skip]),
            units="width",
            pivot="mid",
            width=0.005,
            headwidth=3,
            color="black",
            alpha=0.6,
        )

        name1 = self.get_joint_name(joint_idx_1)
        name2 = self.get_joint_name(joint_idx_2)
        ax.set_title(
            f"Cross Wavelet: {name1} vs {name2}", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel("Frequency (Hz)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        from matplotlib.ticker import ScalarFormatter

        ax.yaxis.set_major_formatter(ScalarFormatter())

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Cross Power", rotation=270, labelpad=15)
        fig.tight_layout()

    def plot_principal_component_analysis(
        self,
        fig: Figure,
        pca_result: PCAResult,
        modes_to_plot: int = 3,
    ) -> None:
        """Plot PCA/Principal Movements analysis results.

        Shows:
        1. Explained Variance Ratio (Scree Plot)
        2. Temporal Scores (Projection of movement onto PCs)

        Args:
            fig: Matplotlib figure
            pca_result: PCAResult object from statistical_analysis
            modes_to_plot: Number of modes to visualize scores for
        """
        # Create grid: Scree plot on top, Scores on bottom
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)

        # 1. Scree Plot
        ax1 = fig.add_subplot(gs[0])

        # Cumulative variance
        cum_var = np.cumsum(pca_result.explained_variance_ratio) * 100
        n_comps = len(cum_var)
        x_indices = np.arange(1, n_comps + 1)

        # Bar chart for individual variance
        ax1.bar(
            x_indices,
            pca_result.explained_variance_ratio * 100,
            alpha=0.6,
            label="Individual",
        )
        # Line for cumulative
        ax1.plot(x_indices, cum_var, "r-o", linewidth=2, label="Cumulative")

        ax1.set_ylabel("Explained Variance (%)", fontweight="bold")
        ax1.set_xlabel("Principal Component", fontweight="bold")
        ax1.set_title("PCA Scree Plot", fontsize=12, fontweight="bold")
        ax1.set_xticks(x_indices)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        # 2. Temporal Scores
        ax2 = fig.add_subplot(gs[1])

        # Get time array
        times, _ = self._get_cached_series("joint_positions")
        # Ensure length matches scores
        scores = pca_result.projected_data
        if len(times) != scores.shape[0]:
            # Try to match
            if len(times) > scores.shape[0]:
                times = times[: scores.shape[0]]
            else:
                scores = scores[: len(times)]

        # Plot top modes
        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["tertiary"],
            self.colors["quaternary"],
            self.colors["quinary"],
        ]

        for i in range(min(modes_to_plot, scores.shape[1])):
            color = colors[i % len(colors)]
            ax2.plot(times, scores[:, i], label=f"PC {i+1}", linewidth=2, color=color)

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
