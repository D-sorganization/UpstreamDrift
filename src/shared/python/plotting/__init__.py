"""Plotting Package for Golf Swing Visualization.

This package provides modular plotting components organized by analysis type.

Modules:
    base: MplCanvas, RecorderInterface, color schemes
    config: ColorScheme, PlotConfig for consistent styling
    kinematics: Joint position/velocity/acceleration plots
    energy: Energy analysis and power plots
    (future) kinetics: Torque/force plots
    (future) coordination: Phase diagrams, coordination patterns
    (future) advanced: Wavelet, RQA, Lyapunov plots

For backward compatibility, the main GolfSwingPlotter class
and MplCanvas are still available via:
    from shared.python.plotting import GolfSwingPlotter, MplCanvas

For new code, prefer importing from specific modules:
    from shared.python.plotting.base import RecorderInterface, MplCanvas
    from shared.python.plotting.config import PlotConfig, ColorScheme
    from shared.python.plotting.kinematics import plot_joint_positions
    from shared.python.plotting.energy import plot_energy_overview
"""

# Import base components
from src.shared.python.plotting.base import (
    MplCanvas,
    RecorderInterface,
)

# Import configuration
from src.shared.python.plotting.config import (
    DARK_THEME,
    DEFAULT_CONFIG,
    LIGHT_THEME,
    ColorScheme,
    PlotConfig,
)

# Import energy functions
from src.shared.python.plotting.energy import (
    plot_energy_breakdown,
    plot_energy_overview,
    plot_power_analysis,
)

# Import kinematics functions
from src.shared.python.plotting.kinematics import (
    plot_club_head_speed,
    plot_joint_positions,
    plot_joint_velocities,
    plot_phase_diagram,
)

# Import main plotter class from core module
from src.shared.python.plotting_core import GolfSwingPlotter

__all__ = [
    # Core classes
    "GolfSwingPlotter",
    "MplCanvas",
    "RecorderInterface",
    # Configuration
    "ColorScheme",
    "PlotConfig",
    "DEFAULT_CONFIG",
    "DARK_THEME",
    "LIGHT_THEME",
    # Kinematic plots
    "plot_joint_positions",
    "plot_joint_velocities",
    "plot_club_head_speed",
    "plot_phase_diagram",
    # Energy plots
    "plot_energy_overview",
    "plot_energy_breakdown",
    "plot_power_analysis",
]
