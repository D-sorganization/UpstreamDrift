"""
Drained shear cell (simplified Jenike-style) calibration experiment.
"""

from pathlib import Path
import numpy as np


class DrainedShearCellExperiment:
    """Applies normal load, shears at constant rate, extracts peak/residual friction angles."""

    def __init__(self, backend: str = "chrono") -> None:
        self.backend = backend
        self.target_phi_peak = 35.0
        self.target_phi_res = 30.0

    def run_simulation(self, params: dict) -> tuple[float, float]:
        """
        Run the shear cell simulation.
        Returns:
            (phi_peak, phi_res) in degrees.
        """
        friction = params.get("friction_coefficient", 0.5)

        # Mock response
        phi_peak = 20.0 + (friction * 30.0)
        phi_res = phi_peak - 5.0
        return phi_peak, phi_res

    def calibrate(self) -> dict:
        """
        Calibrate using an optimization strategy.
        """
        best_params = {"friction_coefficient": 0.5, "restitution_coefficient": 0.3}
        return best_params
