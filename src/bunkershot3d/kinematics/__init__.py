"""
Kinematics module for BunkerShot3D.
"""

from .coupling import CoSimulator, MockDoublePendulum
from .trajectory import SwingTrajectory, generate_reference_trajectory

__all__: list[str] = [
    "CoSimulator",
    "MockDoublePendulum",
    "SwingTrajectory",
    "generate_reference_trajectory",
]
