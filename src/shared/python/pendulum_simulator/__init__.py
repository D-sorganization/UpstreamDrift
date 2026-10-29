"""Double Pendulum Golf Swing Simulator.

A PyQt6-based visualization tool for exploring the dynamics of a driven
double pendulum using Lagrangian mechanics with relative (generalized)
coordinates. Demonstrates how off-diagonal mass matrix terms enable
passive energy transfer in kinematic chains like the golf swing.
"""

from src.shared.python.pendulum_simulator.tolerance_bands import (
    CrossEngineToleranceBand,
    compute_cross_engine_tolerance_band,
)

__version__ = "0.2.0"
__all__ = [
    "CrossEngineToleranceBand",
    "compute_cross_engine_tolerance_band",
]
