"""Optimization loop for calibrating backend parameters to bulk properties.

Issue #7999: the optimizer used to search a fixed two-dimensional space
``[friction, restitution]`` even though no experiment reads
``restitution_coefficient``. The objective was exactly flat in that dimension,
so ``differential_evolution`` returned whichever member of its random
population happened to survive - and that number was written to disk as a
physical material property, differing between backends for no physical reason.

The optimizer now searches only the parameters the experiment declares in
``calibrated_parameters``, and verifies before optimising that each declared
parameter actually changes the objective.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Default search space when an experiment does not declare its own.
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "friction_coefficient": (0.01, 1.0),
    "restitution_coefficient": (0.01, 1.0),
}
_DEFAULT_PARAMETERS: tuple[str, ...] = ("friction_coefficient",)
#: Objective change below this counts as "the parameter has no effect".
_SENSITIVITY_TOLERANCE = 1e-12


class InertParameterError(ValueError):
    """Raised when a declared parameter does not affect the objective.

    Optimising such a parameter cannot identify it: the returned value is a
    property of the optimiser's random population, not of the material.
    """


class CalibrationOptimizer:
    """Optimizes backend contact model parameters to match macroscopic targets.

    Attributes:
        experiment: Experiment exposing ``run_simulation`` and a target.
        parameters: Names of the parameters being calibrated.
        bounds: Search bounds per parameter, in ``parameters`` order.
    """

    def __init__(self, experiment: Any) -> None:
        """Initialize with an experiment instance.

        The experiment must provide:

        - ``target_angle`` or ``target_phi_peak``/``target_phi_res``;
        - ``run_simulation(params: dict) -> float | tuple[float, float]``;
        - optionally ``calibrated_parameters`` and ``parameter_bounds``.

        Args:
            experiment: The experiment to calibrate against.

        Raises:
            ValueError: If ``experiment`` is None or declares no parameters.
        """
        if experiment is None:
            raise ValueError("experiment must be provided")
        self.experiment = experiment
        self.parameters: tuple[str, ...] = tuple(
            getattr(experiment, "calibrated_parameters", _DEFAULT_PARAMETERS)
        )
        if not self.parameters:
            raise ValueError("experiment declares no calibrated_parameters")
        declared_bounds = getattr(experiment, "parameter_bounds", {})
        self.bounds: list[tuple[float, float]] = [
            tuple(declared_bounds.get(name, _DEFAULT_BOUNDS.get(name, (0.01, 1.0))))  # type: ignore[misc]
            for name in self.parameters
        ]

    def _params_from_vector(self, x: np.ndarray) -> dict[str, float]:
        """Map an optimiser vector onto the experiment's keyword parameters."""
        return {name: float(x[i]) for i, name in enumerate(self.parameters)}

    def _objective(self, x: np.ndarray) -> float:
        """Squared residual between the simulated response and the target.

        Args:
            x: Parameter vector aligned with :attr:`parameters`.

        Returns:
            Squared error.

        Raises:
            ValueError: If the experiment declares no recognised target.
        """
        params = self._params_from_vector(np.atleast_1d(x))
        result = self.experiment.run_simulation(params)

        if hasattr(self.experiment, "target_angle"):
            return float((result - self.experiment.target_angle) ** 2)

        if hasattr(self.experiment, "target_phi_peak"):
            phi_peak, phi_res = result
            return float(
                (phi_peak - self.experiment.target_phi_peak) ** 2
                + (phi_res - self.experiment.target_phi_res) ** 2
            )

        raise ValueError("Experiment does not define known target properties.")

    def check_sensitivity(self) -> dict[str, float]:
        """Measure how much each declared parameter moves the objective.

        Returns:
            Mapping of parameter name to the absolute objective change observed
            when the parameter is swept across its bounds with the others held
            at their midpoint.

        Raises:
            InertParameterError: If any declared parameter has no effect.
        """
        midpoint = np.array([(lo + hi) / 2.0 for lo, hi in self.bounds])
        sensitivities: dict[str, float] = {}
        inert: list[str] = []
        for i, name in enumerate(self.parameters):
            low = midpoint.copy()
            high = midpoint.copy()
            low[i], high[i] = self.bounds[i]
            delta = abs(self._objective(high) - self._objective(low))
            sensitivities[name] = delta
            if delta <= _SENSITIVITY_TOLERANCE:
                inert.append(name)

        if inert:
            raise InertParameterError(
                f"{type(self.experiment).__name__} objective is flat in "
                f"{inert}; optimising over it would return optimiser noise, "
                "not a measurement (issue #7999). Remove it from "
                "calibrated_parameters or make the experiment read it."
            )
        return sensitivities

    def optimize(self) -> dict[str, float]:
        """Run global optimisation over the declared parameters.

        Returns:
            The identified parameter values plus the final ``error``. Only
            parameters the objective actually depends on appear.

        Raises:
            InertParameterError: If a declared parameter has no effect.
        """
        from scipy.optimize import differential_evolution

        self.check_sensitivity()

        # differential_evolution is a stochastic population-based method
        # suitable for noisy granular simulations.
        res = differential_evolution(
            self._objective,
            self.bounds,  # type: ignore[arg-type]
            strategy="best1bin",
            maxiter=50,
            popsize=5,
            tol=0.01,
        )

        identified = self._params_from_vector(np.atleast_1d(res.x))
        identified["error"] = float(res.fun)
        return identified
