"""
Project Chrono backend driver for BunkerShot3D.
"""

from pathlib import Path
import yaml
import numpy as np


class ChronoDriver:
    """Driver for running the bunker shot simulation using Project Chrono."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def setup(self) -> None:
        """Setup the Chrono system (grains, clubhead, constraints)."""
        # TODO: import pychrono as chrono
        # Initialize chrono.ChSystemSMC()

    def run(self, output_path: Path | str) -> None:
        """Run the simulation and write HDF5 output."""
        # TODO: Advance timestep, interpolate trajectory, set clubhead kinematics
        # Query contact forces, extract grains, and use BunkerShotResultWriter
