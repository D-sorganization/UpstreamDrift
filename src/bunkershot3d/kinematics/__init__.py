"""
Kinematics module for BunkerShot3D.
"""

from .coupling import CoSimulator, CoupledDoublePendulum
from .trajectory import SwingTrajectory, generate_reference_trajectory

__all__: list[str] = [
    "CoSimulator",
    "CoupledDoublePendulum",
    "SwingTrajectory",
    "generate_reference_trajectory",
]
