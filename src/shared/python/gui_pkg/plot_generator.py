"""Plot Generation Module for Simulation Data.

Provides configurable plot generation for simulation datasets. Integrates with
the data explorer and export system to produce standardized plot sets for
kinematics, kinetics, energy analysis, and phase portraits.

Design by Contract:
    Preconditions:
        - Data must be valid numpy arrays with consistent dimensions
        - Output directories must be writable
    Postconditions:
        - Generated plots are saved in requested format
        - Plot metadata is recorded for reproducibility
    Invariants:
        - Plot styling is consistent across all plot types
        - Data is never modified during plotting

Usage:
    >>> from src.shared.python.gui_pkg.plot_generator import PlotGenerator
    >>> gen = PlotGenerator()
    >>> gen.generate_standard_plots(data, output_dir="plots/")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment, misc]


class PlotType(str):
    """Standard plot type identifiers."""

    JOINT_POSITIONS = "joint_positions"
    JOINT_VELOCITIES = "joint_velocities"
    JOINT_ACCELERATIONS = "joint_accelerations"
    JOINT_TORQUES = "joint_torques"
    ENERGY = "energy"
    PHASE_PORTRAIT = "phase_portrait"
    CONTACT_FORCES = "contact_forces"
    DRIFT_VS_CONTROL = "drift_vs_control"
    COMPARISON = "comparison"
    POWER = "power"
    MASS_MATRIX_CONDITION = "mass_matrix_condition"
    TRAJECTORY_3D = "trajectory_3d"


# All available standard plot types
ALL_PLOT_TYPES = [
    PlotType.JOINT_POSITIONS,
    PlotType.JOINT_VELOCITIES,
    PlotType.JOINT_ACCELERATIONS,
    PlotType.JOINT_TORQUES,
    PlotType.ENERGY,
    PlotType.PHASE_PORTRAIT,
    PlotType.CONTACT_FORCES,
    PlotType.DRIFT_VS_CONTROL,
    PlotType.POWER,
    PlotType.MASS_MATRIX_CONDITION,
]


@dataclass
class PlotConfig:
    """Configuration for plot generation.

    Attributes:
        plot_types: Which plots to generate (None = all).
        output_format: Image format ('png', 'svg', 'pdf').
        dpi: Resolution in dots per inch.
        figsize: Figure size (width, height) in inches.
        style: Matplotlib style name.
        show_grid: Whether to show grid lines.
        joint_indices: Which joints to plot (None = all).
        max_joints_per_plot: Maximum joints per subplot.
        title_prefix: Prefix for all plot titles.
    """

    plot_types: list[str] | None = None
    output_format: str = "png"
    dpi: int = 150
    figsize: tuple[float, float] = (12, 8)
    style: str = "default"
    show_grid: bool = True
    joint_indices: list[int] | None = None
    max_joints_per_plot: int = 8
    title_prefix: str = ""


@dataclass
class SimulationData:
    """Container for simulation data to be plotted.

    Attributes:
        times: Time array (n_steps,).
        positions: Joint positions (n_steps, n_q).
        velocities: Joint velocities (n_steps, n_v).
        accelerations: Joint accelerations (n_steps, n_v) or None.
        torques: Applied torques (n_steps, n_v) or None.
        energies: Dict of energy arrays (each n_steps,).
        contact_forces: Contact forces (n_steps, 3) or None.
        drift_accelerations: Drift accelerations (n_steps, n_v) or None.
        control_accelerations: Control accelerations (n_steps, n_v) or None.
        mass_matrices: Mass matrices (n_steps, n_v, n_v) or None.
        joint_names: Names for each joint.
        model_name: Name of the model.
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray | None = None
    torques: np.ndarray | None = None
    energies: dict[str, np.ndarray] = field(default_factory=dict)
    contact_forces: np.ndarray | None = None
    drift_accelerations: np.ndarray | None = None
    control_accelerations: np.ndarray | None = None
    mass_matrices: np.ndarray | None = None
    joint_names: list[str] = field(default_factory=list)
    model_name: str = "simulation"


class PlotGenerator:
    """Generates standardized plots from simulation data.

    Supports configurable plot types, styles, and output formats.
    Integrates with the data explorer for batch processing.

    Design by Contract:
        Preconditions:
            - matplotlib must be available for plot generation
            - Data must have consistent dimensions
        Postconditions:
            - All requested plots are saved to output directory
            - Plot manifest file lists all generated plots
    """

    def __init__(self, config: PlotConfig | None = None) -> None:
        """Initialize the plot generator.

        Args:
            config: Plot configuration. Uses defaults if None.
        """
        self.config = config or PlotConfig()

        if not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "matplotlib not available. Plot generation will be disabled."
            )

    def generate_standard_plots(
        self,
        data: SimulationData,
        output_dir: str | Path,
    ) -> list[Path]:
        """Generate all configured standard plots.

        Args:
            data: Simulation data to plot.
            output_dir: Directory for output plot files.

        Returns:
            List of paths to generated plot files.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping plot generation")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_types = self.config.plot_types or ALL_PLOT_TYPES
        generated: list[Path] = []

        for plot_type in plot_types:
            try:
                path = self._generate_plot(data, plot_type, output_dir)
                if path is not None:
                    generated.append(path)
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Failed to generate %s plot: %s", plot_type, e)

        logger.info(
            "Generated %d/%d plots in %s",
            len(generated),
            len(plot_types),
            output_dir,
        )
        return generated

    def generate_single_plot(
        self,
        data: SimulationData,
        plot_type: str,
        output_path: str | Path | None = None,
    ) -> Figure | None:
        """Generate a single plot.

        Args:
            data: Simulation data.
            plot_type: Type of plot to generate.
            output_path: Optional path to save the plot.

        Returns:
            Matplotlib Figure, or None if matplotlib unavailable.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig = self._create_plot(data, plot_type)

        if fig is not None and output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                str(output_path),
                dpi=self.config.dpi,
                bbox_inches="tight",
            )
            plt.close(fig)

        return fig

    def get_available_plot_types(self) -> list[dict[str, str]]:
        """Get list of available plot types with descriptions.

        Returns:
            List of dicts with 'type' and 'description'.
        """
        return [
            {
                "type": PlotType.JOINT_POSITIONS,
                "description": "Joint positions vs time",
            },
            {
                "type": PlotType.JOINT_VELOCITIES,
                "description": "Joint velocities vs time",
            },
            {
                "type": PlotType.JOINT_ACCELERATIONS,
                "description": "Joint accelerations vs time",
            },
            {
                "type": PlotType.JOINT_TORQUES,
                "description": "Applied joint torques vs time",
            },
            {
                "type": PlotType.ENERGY,
                "description": "Energy analysis (kinetic, potential, total)",
            },
            {
                "type": PlotType.PHASE_PORTRAIT,
                "description": "Phase portrait (position vs velocity)",
            },
            {
                "type": PlotType.CONTACT_FORCES,
                "description": "Contact / ground reaction forces",
            },
            {
                "type": PlotType.DRIFT_VS_CONTROL,
                "description": "Drift vs control acceleration decomposition",
            },
            {"type": PlotType.POWER, "description": "Joint power (torque × velocity)"},
            {
                "type": PlotType.MASS_MATRIX_CONDITION,
                "description": "Mass matrix condition number over time",
            },
        ]

    def _generate_plot(
        self,
        data: SimulationData,
        plot_type: str,
        output_dir: Path,
    ) -> Path | None:
        """Generate a single plot and save it.

        Args:
            data: Simulation data.
            plot_type: Plot type identifier.
            output_dir: Output directory.

        Returns:
            Path to the saved plot, or None.
        """
        fig = self._create_plot(data, plot_type)
        if fig is None:
            return None

        filename = f"{data.model_name}_{plot_type}.{self.config.output_format}"
        filepath = output_dir / filename
        fig.savefig(str(filepath), dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def _create_plot(self, data: SimulationData, plot_type: str) -> Figure | None:
        """Create a plot figure without saving.

        Args:
            data: Simulation data.
            plot_type: Plot type identifier.

        Returns:
            Matplotlib Figure, or None if data insufficient.
        """
        if plot_type == PlotType.JOINT_POSITIONS:
            return self._plot_joint_data(
                data.times,
                data.positions,
                data.joint_names,
                "Joint Positions",
                "Position (rad)",
            )
        elif plot_type == PlotType.JOINT_VELOCITIES:
            return self._plot_joint_data(
                data.times,
                data.velocities,
                data.joint_names,
                "Joint Velocities",
                "Velocity (rad/s)",
            )
        elif plot_type == PlotType.JOINT_ACCELERATIONS:
            if data.accelerations is None:
                return None
            return self._plot_joint_data(
                data.times,
                data.accelerations,
                data.joint_names,
                "Joint Accelerations",
                "Acceleration (rad/s²)",
            )
        elif plot_type == PlotType.JOINT_TORQUES:
            if data.torques is None:
                return None
            return self._plot_joint_data(
                data.times,
                data.torques,
                data.joint_names,
                "Joint Torques",
                "Torque (N·m)",
            )
        elif plot_type == PlotType.ENERGY:
            return self._plot_energy(data)
        elif plot_type == PlotType.PHASE_PORTRAIT:
            return self._plot_phase_portrait(data)
        elif plot_type == PlotType.CONTACT_FORCES:
            return self._plot_contact_forces(data)
        elif plot_type == PlotType.DRIFT_VS_CONTROL:
            return self._plot_drift_vs_control(data)
        elif plot_type == PlotType.POWER:
            return self._plot_power(data)
        elif plot_type == PlotType.MASS_MATRIX_CONDITION:
            return self._plot_mass_matrix_condition(data)
        else:
            logger.warning("Unknown plot type: %s", plot_type)
            return None

    def _plot_joint_data(
        self,
        times: np.ndarray,
        data: np.ndarray,
        joint_names: list[str],
        title: str,
        ylabel: str,
    ) -> Figure:
        """Plot per-joint time series data.

        Args:
            times: Time array.
            data: Joint data array (n_steps, n_joints).
            joint_names: Joint name labels.
            title: Plot title.
            ylabel: Y-axis label.

        Returns:
            Matplotlib Figure.
        """
        n_joints = data.shape[1]
        indices = self.config.joint_indices or list(range(n_joints))
        indices = indices[: self.config.max_joints_per_plot]

        fig, ax = plt.subplots(figsize=self.config.figsize)

        for idx in indices:
            name = joint_names[idx] if idx < len(joint_names) else f"Joint {idx}"
            ax.plot(times, data[:, idx], label=name, linewidth=0.8)

        prefix = f"{self.config.title_prefix} " if self.config.title_prefix else ""
        ax.set_title(f"{prefix}{title}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize="small")
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_energy(self, data: SimulationData) -> Figure | None:
        """Plot energy analysis."""
        if not data.energies:
            return None

        fig, ax = plt.subplots(figsize=self.config.figsize)

        for name, energy_arr in data.energies.items():
            if isinstance(energy_arr, np.ndarray) and len(energy_arr) == len(
                data.times
            ):
                ax.plot(data.times, energy_arr, label=name.capitalize(), linewidth=1.0)

        prefix = f"{self.config.title_prefix} " if self.config.title_prefix else ""
        ax.set_title(f"{prefix}Energy Analysis")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy (J)")
        ax.legend()
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_phase_portrait(self, data: SimulationData) -> Figure:
        """Plot phase portrait (position vs velocity) for each joint."""
        n_joints = min(data.positions.shape[1], data.velocities.shape[1])
        indices = self.config.joint_indices or list(range(n_joints))
        indices = indices[: min(4, len(indices))]  # Max 4 subplots

        n_plots = len(indices)
        ncols = min(2, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=self.config.figsize)
        if n_plots == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
            ax = axes[i]
            name = (
                data.joint_names[idx] if idx < len(data.joint_names) else f"Joint {idx}"
            )
            ax.plot(data.positions[:, idx], data.velocities[:, idx], linewidth=0.5)
            ax.set_title(f"{name}")
            ax.set_xlabel("Position (rad)")
            ax.set_ylabel("Velocity (rad/s)")
            if self.config.show_grid:
                ax.grid(True, alpha=0.3)
            # Mark start and end
            ax.plot(
                data.positions[0, idx],
                data.velocities[0, idx],
                "go",
                markersize=5,
                label="Start",
            )
            ax.plot(
                data.positions[-1, idx],
                data.velocities[-1, idx],
                "ro",
                markersize=5,
                label="End",
            )
            ax.legend(fontsize="x-small")

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        prefix = f"{self.config.title_prefix} " if self.config.title_prefix else ""
        fig.suptitle(f"{prefix}Phase Portraits")
        fig.tight_layout()
        return fig

    def _plot_contact_forces(self, data: SimulationData) -> Figure | None:
        """Plot contact / ground reaction forces."""
        if data.contact_forces is None:
            return None

        fig, ax = plt.subplots(figsize=self.config.figsize)
        labels = ["Fx", "Fy", "Fz"]
        for i in range(min(3, data.contact_forces.shape[1])):
            ax.plot(
                data.times, data.contact_forces[:, i], label=labels[i], linewidth=0.8
            )

        # Plot magnitude
        magnitude = np.linalg.norm(data.contact_forces[:, :3], axis=1)
        ax.plot(
            data.times,
            magnitude,
            label="|F|",
            linewidth=1.0,
            linestyle="--",
            color="black",
        )

        prefix = f"{self.config.title_prefix} " if self.config.title_prefix else ""
        ax.set_title(f"{prefix}Contact Forces (GRF)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N)")
        ax.legend()
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_drift_vs_control(self, data: SimulationData) -> Figure | None:
        """Plot drift vs control acceleration decomposition."""
        if data.drift_accelerations is None or data.control_accelerations is None:
            return None

        n_joints = data.drift_accelerations.shape[1]
        indices = self.config.joint_indices or list(range(n_joints))
        indices = indices[: min(4, len(indices))]

        n_plots = len(indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]
            name = (
                data.joint_names[idx] if idx < len(data.joint_names) else f"Joint {idx}"
            )
            ax.plot(
                data.times, data.drift_accelerations[:, idx], label="Drift", alpha=0.8
            )
            ax.plot(
                data.times,
                data.control_accelerations[:, idx],
                label="Control",
                alpha=0.8,
            )
            total = (
                data.drift_accelerations[:, idx] + data.control_accelerations[:, idx]
            )
            ax.plot(data.times, total, label="Total", linestyle="--", alpha=0.6)
            ax.set_ylabel(f"{name}\n(rad/s²)")
            ax.legend(loc="upper right", fontsize="x-small")
            if self.config.show_grid:
                ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        prefix = f"{self.config.title_prefix} " if self.config.title_prefix else ""
        fig.suptitle(f"{prefix}Drift vs Control Acceleration (Section F)")
        fig.tight_layout()
        return fig

    def _plot_power(self, data: SimulationData) -> Figure | None:
        """Plot joint power (torque × velocity)."""
        if data.torques is None:
            return None

        n_joints = min(data.torques.shape[1], data.velocities.shape[1])
        power = data.torques[:, :n_joints] * data.velocities[:, :n_joints]

        indices = self.config.joint_indices or list(range(n_joints))
        indices = indices[: self.config.max_joints_per_plot]

        fig, ax = plt.subplots(figsize=self.config.figsize)
        for idx in indices:
            name = (
                data.joint_names[idx] if idx < len(data.joint_names) else f"Joint {idx}"
            )
            ax.plot(data.times, power[:, idx], label=name, linewidth=0.8)

        prefix = f"{self.config.title_prefix} " if self.config.title_prefix else ""
        ax.set_title(f"{prefix}Joint Power (τ × ω)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="best", fontsize="small")
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)
        fig.tight_layout()
        return fig

    def _plot_mass_matrix_condition(self, data: SimulationData) -> Figure | None:
        """Plot mass matrix condition number over time."""
        if data.mass_matrices is None:
            return None

        n_steps = data.mass_matrices.shape[0]
        condition_numbers = np.zeros(n_steps)

        for i in range(n_steps):
            M = data.mass_matrices[i]
            if np.any(M != 0):
                try:
                    condition_numbers[i] = np.linalg.cond(M)
                except (ValueError, TypeError, RuntimeError):
                    condition_numbers[i] = np.nan

        fig, ax = plt.subplots(figsize=self.config.figsize)
        ax.semilogy(data.times, condition_numbers, linewidth=0.8)

        prefix = f"{self.config.title_prefix} " if self.config.title_prefix else ""
        ax.set_title(f"{prefix}Mass Matrix Condition Number")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Condition Number (log scale)")
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
