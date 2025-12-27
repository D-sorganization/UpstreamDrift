"""Advanced plotting and visualization for golf swing analysis.

This module provides comprehensive plotting capabilities including:
- Time series plots (kinematics, kinetics, energetics)
- Phase diagrams
- Force/torque visualizations
- Power and energy analysis
- Swing sequence analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from matplotlib.figure import Figure

from shared.python.swing_plane_analysis import SwingPlaneAnalyzer

if TYPE_CHECKING:
    pass

# Qt backend - optional for headless environments
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    class MplCanvas(FigureCanvasQTAgg):  # type: ignore[misc]
        """Matplotlib canvas for embedding in PyQt6."""

        def __init__(self, width=8, height=6, dpi=100) -> None:
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

        def __init__(self, width=8, height=6, dpi=100) -> None:  # noqa: ARG002
            """Initialize canvas with figure (placeholder for headless environments)."""
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
        ...


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
    ) -> None:
        """Initialize plotter with recorded data.

        Args:
            recorder: Object providing get_time_series(field_name) method
            joint_names: Optional list of joint names. If None, uses "Joint X"
        """
        self.recorder = recorder
        self.joint_names = joint_names or []

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
        times, positions = self.recorder.get_time_series("joint_positions")

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
        times, velocities = self.recorder.get_time_series("joint_velocities")

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
        times, torques = self.recorder.get_time_series("joint_torques")

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
        times, powers = self.recorder.get_time_series("actuator_powers")

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
        times_ke, ke = self.recorder.get_time_series("kinetic_energy")
        times_pe, pe = self.recorder.get_time_series("potential_energy")
        times_te, te = self.recorder.get_time_series("total_energy")

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
        times, speeds = self.recorder.get_time_series("club_head_speed")

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
        times, positions = self.recorder.get_time_series("club_head_position")

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
        ax.scatter(
            [x[0]],
            [y[0]],
            [z[0]],
            color="green",
            s=100,  # type: ignore[misc]
            marker="o",
            label="Start",
        )
        ax.scatter(
            [x[-1]],
            [y[-1]],
            [z[-1]],
            color="red",
            s=100,  # type: ignore[misc]
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
        times, positions = self.recorder.get_time_series("joint_positions")
        _, velocities = self.recorder.get_time_series("joint_velocities")

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
        times, torques = self.recorder.get_time_series("joint_torques")

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
        ax.set_prop_cycle(None)  # Reset color cycle
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
            _, data = self.recorder.get_time_series("joint_positions")
            ylabel = "PSD (rad²/Hz)"
            title = "Joint Position PSD"
        elif signal_type == "torque":
            _, data = self.recorder.get_time_series("joint_torques")
            ylabel = "PSD (Nm²/Hz)"
            title = "Joint Torque PSD"
        else:  # velocity
            _, data = self.recorder.get_time_series("joint_velocities")
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
        times, _ = self.recorder.get_time_series("joint_positions")
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
            _, data = self.recorder.get_time_series("joint_positions")
            title = "Joint Position Spectrogram"
        elif signal_type == "torque":
            _, data = self.recorder.get_time_series("joint_torques")
            title = "Joint Torque Spectrogram"
        else:  # velocity
            _, data = self.recorder.get_time_series("joint_velocities")
            title = "Joint Velocity Spectrogram"

        data = np.asarray(data)
        if data.ndim < 2 or joint_idx >= data.shape[1]:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return

        signal_data = data[:, joint_idx]

        # Calculate sampling rate
        times, _ = self.recorder.get_time_series("joint_positions")
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
        times, speeds = self.recorder.get_time_series("club_head_speed")
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
        times_ke, ke = self.recorder.get_time_series("kinetic_energy")
        times_pe, pe = self.recorder.get_time_series("potential_energy")
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
        times_am, am = self.recorder.get_time_series("angular_momentum")
        am = np.asarray(am)
        if len(times_am) > 0 and am.size > 0:
            am_mag = np.linalg.norm(am, axis=1)
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
        times, positions = self.recorder.get_time_series("joint_positions")
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
        times_cop, cop = self.recorder.get_time_series("cop_position")
        cop = np.asarray(cop)
        if len(times_cop) > 0 and cop.size > 0:
            sc = ax5.scatter(
                cop[:, 0], cop[:, 1], c=times_cop, cmap="viridis", s=10
            )
            ax5.set_title("CoP Trajectory", fontsize=11, fontweight="bold")
            ax5.set_xlabel("X (m)", fontsize=9)
            ax5.set_ylabel("Y (m)", fontsize=9)
            ax5.axis("equal")
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No CoP data", ha="center", va="center")

        # 6. Torques (Bottom Right)
        ax6 = fig.add_subplot(gs[1, 2])
        times, torques = self.recorder.get_time_series("joint_torques")
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
    ) -> None:
        """Plot kinematic sequence (normalized velocities).

        Visualizes proximal-to-distal sequencing.

        Args:
            fig: Matplotlib figure
            segment_indices: Map of segment names to joint indices
        """
        times, velocities = self.recorder.get_time_series("joint_velocities")
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

        for i, (name, idx) in enumerate(segment_indices.items()):
            if idx < velocities.shape[1]:
                vel = np.abs(velocities[:, idx])
                # Normalize to peak
                max_vel = np.max(vel)
                if max_vel > 0:
                    vel_norm = vel / max_vel
                else:
                    vel_norm = vel

                color = colors[i % len(colors)]
                ax.plot(times, vel_norm, label=name, color=color, linewidth=2)

                # Mark peak
                max_t_idx = np.argmax(vel)
                ax.plot(
                    times[max_t_idx],
                    vel_norm[max_t_idx],
                    "o",
                    color=color,
                    markersize=8,
                )

        ax.set_title("Kinematic Sequence (Normalized)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Normalized Velocity", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

    def plot_3d_phase_space(self, fig: Figure, joint_idx: int = 0) -> None:
        """Plot 3D phase space (Position vs Velocity vs Acceleration).

        Args:
            fig: Matplotlib figure
            joint_idx: Joint index
        """
        times, positions = self.recorder.get_time_series("joint_positions")
        _, velocities = self.recorder.get_time_series("joint_velocities")
        _, accelerations = self.recorder.get_time_series("joint_accelerations")

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
        ax.scatter(
            [pos[0]],
            [vel[0]],
            [acc[0]],
            color="green",
            s=100,  # type: ignore[misc]
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
            _, data = self.recorder.get_time_series("joint_positions")
            title = "Joint Position Correlation"
        elif data_type == "torque":
            _, data = self.recorder.get_time_series("joint_torques")
            title = "Joint Torque Correlation"
        else:
            _, data = self.recorder.get_time_series("joint_velocities")
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

        # Add correlation values
        if data.shape[1] <= 8:
            for i in range(data.shape[1]):
                for j in range(data.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{corr_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="k" if abs(corr_matrix[i, j]) < 0.5 else "w",
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
        times, positions = self.recorder.get_time_series("club_head_position")

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
            z,
            c=np.abs(deviations),
            cmap="coolwarm",
            s=20,  # type: ignore[misc]
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

        if abs(normal[2]) > 1e-6:
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
            pass

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
        ax.set_title(
            f"Swing Plane Analysis\nSteepness: {metrics.steepness_deg:.1f}°, "
            f"RMSE: {metrics.rmse*100:.1f} cm",
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
        times, am_data = self.recorder.get_time_series("angular_momentum")
        am_data = np.asarray(am_data)

        if len(times) == 0 or am_data.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Angular Momentum Data", ha="center", va="center")
            return

        # Calculate magnitude
        am_mag = np.linalg.norm(am_data, axis=1)

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

    def plot_cop_trajectory(self, fig: Figure) -> None:
        """Plot Center of Pressure trajectory (top-down view).

        Args:
            fig: Matplotlib figure
        """
        times, cop_data = self.recorder.get_time_series("cop_position")
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
