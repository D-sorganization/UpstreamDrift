"""Plotting package for golf swing visualization.

This package provides modular plotting components:

- base: MplCanvas, RecorderInterface, color schemes
- (future) kinematics: Joint angle/velocity plots
- (future) kinetics: Torque/force plots
- (future) energy: Energy analysis plots
- (future) coordination: Phase diagrams, coordination patterns
- (future) advanced: Wavelet, RQA, Lyapunov plots

For backward compatibility, the main GolfSwingPlotter class
and MplCanvas are still available via:
    from shared.python.plotting import GolfSwingPlotter, MplCanvas

For new code, prefer importing from specific modules:
    from shared.python.plotting.base import RecorderInterface, MplCanvas
"""

# Import base components
from shared.python.plotting.base import (
    MplCanvas,
    RecorderInterface,
)

# Import main plotter class from core module
from shared.python.plotting_core import GolfSwingPlotter

__all__ = [
    "GolfSwingPlotter",
    "MplCanvas",
    "RecorderInterface",
]
