"""
Optimization loop for calibrating backend parameters to bulk properties.
"""

import numpy as np
from typing import Any


class CalibrationOptimizer:
    """Optimizes backend contact model parameters to match macroscopic targets."""

    #: Physically admissible ranges for (friction, restitution).
    #: ``optimize()`` hands these to the solver, which is the ONLY place the
    #: range is enforced. ``_objective`` deliberately does NOT clip: issue #6644
    #: F5 removed an internal clip because it created flat plateaus that stalled
    #: ``differential_evolution`` and let ``res.x`` sit outside the physical
    #: range while being silently re-clipped. See #8038.
    BOUNDS: tuple[tuple[float, float], tuple[float, float]] = (
        (0.01, 1.0),
        (0.01, 1.0),
    )

    def __init__(self, experiment: "Any") -> None:  # type: ignore
        """
        Initialize with an experiment instance.
        The experiment must have:
        - `target_angle` or `target_phi_peak`
        - `run_simulation(params: dict) -> float`
        """
        self.experiment = experiment

    def _objective(self, x: np.ndarray) -> float:
        """Objective function to minimize.

        Preconditions:
            - ``x`` provides at least two parameters (friction, restitution).

        Postconditions:
            - Parameters are forwarded to the experiment UNMODIFIED.

        The range is enforced by the ``bounds`` argument in :meth:`optimize`,
        not here. Clipping inside the objective was removed by #6644 F5: it is
        invisible to ``differential_evolution``, so it creates flat plateaus
        that stall the search and lets the returned ``res.x`` sit outside the
        physical range. Callers invoking ``_objective`` directly are responsible
        for supplying values within :attr:`BOUNDS`.
        """
        if x is None or len(x) < 2:
            raise ValueError(
                "x must provide two parameters (friction, restitution), "
                f"got {0 if x is None else len(x)}"
            )
        friction = float(x[0])
        restitution = float(x[1])

        params = {
            "friction_coefficient": friction,
            "restitution_coefficient": restitution,
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
        from scipy.optimize import differential_evolution

        bounds = [tuple(bound) for bound in self.BOUNDS]

        # differential_evolution is a stochastic population-based method suitable for noisy granular simulations
        res = differential_evolution(
            self._objective,
            bounds,  # type: ignore[arg-type]
            strategy="best1bin",
            maxiter=50,
            popsize=5,
            tol=0.01,
        )

        best_fric, best_rest = res.x
        return {
            "friction_coefficient": float(best_fric),
            "restitution_coefficient": float(best_rest),
            "error": float(res.fun),
        }
