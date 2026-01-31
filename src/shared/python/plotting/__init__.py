"""Plotting package for golf swing visualization.

This package provides modular plotting components:

- base: MplCanvas, RecorderInterface, color schemes
- kinematics: Joint angle/velocity plots
- kinetics: Torque/force plots
- energy: Energy analysis plots
- coordination: Phase diagrams, coordination patterns
- advanced: Wavelet, RQA, Lyapunov plots

For backward compatibility, the main GolfSwingPlotter class
and MplCanvas are still available via:
    from shared.python.plotting import GolfSwingPlotter, MplCanvas

For new code, prefer importing from specific modules:
    from shared.python.plotting.base import RecorderInterface, MplCanvas
"""

# Import base components
from src.shared.python.plotting.base import (
    MplCanvas,
    RecorderInterface,
)

# Import main plotter class from core module
from src.shared.python.plotting.core import GolfSwingPlotter

__all__ = [
    "GolfSwingPlotter",
    "MplCanvas",
    "RecorderInterface",
]
