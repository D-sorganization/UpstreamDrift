"""
MuJoCo MPM backend driver for BunkerShot3D.
"""

from pathlib import Path
import yaml
import numpy as np


class MPMDriver:
    """Driver for running the bunker shot simulation using MuJoCo MPM continuum models."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def setup(self) -> None:
        """Setup the MuJoCo XML string with MPM sand plugin and clubhead."""
        # TODO: import mujoco
        # Construct the MJCF XML bridging discrete parameters to continuum properties

    def run(self, output_path: Path | str) -> None:
        """Run the simulation and write HDF5 output."""
        # TODO: Advance step using mujoco.mj_step
        # Read forces on clubhead from mjData
