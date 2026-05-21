"""
Optimization loop for calibrating backend parameters to bulk properties.
"""

import numpy as np
from scipy.optimize import differential_evolution
from typing import Any
from collections.abc import Callable


class CalibrationOptimizer:
    """Optimizes backend contact model parameters to match macroscopic targets."""

    def __init__(self, experiment: "Any") -> None:  # type: ignore
        """
        Initialize with an experiment instance.
        The experiment must have:
        - `target_angle` or `target_phi_peak`
        - `run_simulation(params: dict) -> float`
        """
        self.experiment = experiment

    def _objective(self, x: np.ndarray) -> float:
        """Objective function to minimize."""
        friction, restitution = x[0], x[1]

        # Clip parameters to physically meaningful bounds during optimization
        friction = np.clip(friction, 0.01, 1.0)
        restitution = np.clip(restitution, 0.01, 1.0)

        params = {
            "friction_coefficient": float(friction),
            "restitution_coefficient": float(restitution),
        }

        # This handles both AngleOfRepose (returns float) and DrainedShearCell (returns tuple)
        result = self.experiment.run_simulation(params)

        if hasattr(self.experiment, "target_angle"):
            target = self.experiment.target_angle
            return (result - target) ** 2

        if hasattr(self.experiment, "target_phi_peak"):
            target_peak = self.experiment.target_phi_peak
            target_res = self.experiment.target_phi_res
            phi_peak, phi_res = result
            return (phi_peak - target_peak) ** 2 + (phi_res - target_res) ** 2

        raise ValueError("Experiment does not define known target properties.")

    def optimize(self) -> dict[str, float]:
        bounds = [(0.01, 1.0), (0.01, 1.0)]

        # differential_evolution is a stochastic population-based method suitable for noisy granular simulations
        res = differential_evolution(
            self._objective,
            bounds,
            strategy="best1bin",
            maxiter=50,
            popsize=5,
            tol=0.01,
        )

        best_fric, best_rest = res.x
        return {
            "friction_coefficient": float(np.clip(best_fric, 0.01, 1.0)),
            "restitution_coefficient": float(np.clip(best_rest, 0.01, 1.0)),
            "error": float(res.fun),
        }
