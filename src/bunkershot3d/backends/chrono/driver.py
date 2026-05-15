"""
Project Chrono backend driver for BunkerShot3D.
"""

from pathlib import Path
import numpy as np
from bunkershot3d.config import BunkerShotConfig


class ChronoDriver:
    """Driver for running the bunker shot simulation using Project Chrono."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config = BunkerShotConfig.from_yaml(self.config_path)

    def setup(self) -> None:
        """Setup the Chrono system (grains, clubhead, constraints)."""
        # TODO: import pychrono as chrono
        # Initialize chrono.ChSystemSMC()

    def run(self, output_path: Path | str) -> None:
        """Run the simulation and write HDF5 output."""
        # TODO: Advance timestep, interpolate trajectory, set clubhead kinematics
        # Query contact forces, extract grains, and use BunkerShotResultWriter
