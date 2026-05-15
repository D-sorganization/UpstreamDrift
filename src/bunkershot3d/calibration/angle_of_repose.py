"""
Angle of repose calibration experiment.
"""

from pathlib import Path
import numpy as np


class AngleOfReposeExperiment:
    """Simulates pouring particles from a lifted cylinder to measure final pile angle."""

    def __init__(self, backend: str = "chrono") -> None:
        """
        Initialize the experiment.
        Args:
            backend: The simulator backend to use (chrono, liggghts, mpm)
        """
        self.backend = backend
        self.target_angle = 32.0  # degrees

    def run_simulation(self, params: dict) -> float:
        """
        Run the calibration experiment for a given parameter set.
        Returns:
            The measured angle of repose in degrees.
        """
        # TODO: Construct cylinder, spawn grains, lift cylinder, settle, compute angle
        # For draft, return a mock value close to target if params are reasonable
        friction = params.get("friction_coefficient", 0.5)
        return 20.0 + (friction * 24.0)  # Mock mapping

    def calibrate(self) -> dict:
        """
        Run Bayesian optimization / CMA-ES to find optimal parameters.
        """
        # Mock calibration loop
        best_params = {"friction_coefficient": 0.5, "restitution_coefficient": 0.3}
        return best_params
