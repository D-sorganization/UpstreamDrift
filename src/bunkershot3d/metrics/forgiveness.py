"""Forgiveness metrics: sensitivity of ball launch to input errors (issue #8614).

Quantifies how tolerant a wedge design is to golfer errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np

__all__ = [
    "ForgivenessMetrics",
    "SensitivityGradient",
    "compute_forgiveness_metrics",
]


@dataclass(frozen=True, slots=True)
class SensitivityGradient:
    """Gradient of an output with respect to an input.

    Attributes:
        value: The sensitivity value (d_output / d_input).
        units: Units string (e.g. "m/s per m").
        input_name: Name of the input parameter.
        output_name: Name of the output parameter.
    """

    value: float
    units: str
    input_name: str = ""
    output_name: str = ""

    @classmethod
    def from_finite_diff(
        cls,
        func: Callable[[float], float],
        nominal_value: float,
        perturbation: float,
        *,
        input_units: str = "m",
        output_units: str = "m/s",
    ) -> SensitivityGradient:
        """Compute gradient using central finite differences.

        Args:
            func: Function to differentiate (input -> output).
            nominal_value: Nominal input value.
            perturbation: Perturbation size for finite diff.
            input_units: Units of the input.
            output_units: Units of the output.

        Returns:
            SensitivityGradient with computed value.
        """
        f_plus = func(nominal_value + perturbation)
        f_minus = func(nominal_value - perturbation)
        gradient = (f_plus - f_minus) / (2 * perturbation)

        return cls(
            value=gradient,
            units=f"{output_units} per {input_units}",
        )


@dataclass
class ForgivenessMetrics:
    """Complete forgiveness analysis.

    Attributes:
        gradients: Nested dict of sensitivities [input][output].
        forgiveness_index: Normalized score (0-1, higher = more forgiving).
        error_budgets: Allowable input errors for given output variance.
    """

    gradients: dict[str, dict[str, SensitivityGradient]] = field(default_factory=dict)
    forgiveness_index: float = 0.5
    _reference_scale: dict[str, float] = field(default_factory=dict)

    def compute_error_budget(
        self,
        output_name: str,
        max_variance: float,
    ) -> dict[str, float]:
        """Compute allowable input error for given output variance.

        Args:
            output_name: Name of the output to bound.
            max_variance: Maximum acceptable output variance.

        Returns:
            Dict of input_name -> max_error.
        """
        budget: dict[str, float] = {}

        for input_name, output_grads in self.gradients.items():
            if output_name in output_grads:
                grad = output_grads[output_name]
                if abs(grad.value) > 1e-10:
                    budget[input_name] = max_variance / abs(grad.value)
                else:
                    budget[input_name] = float("inf")

        return budget


def compute_forgiveness_metrics(
    outcome_model: Callable[..., dict],
    nominal_inputs: dict[str, float],
    perturbations: dict[str, float],
    *,
    reference_scales: dict[str, float] | None = None,
) -> ForgivenessMetrics:
    """Compute forgiveness metrics via sensitivity analysis.

    Args:
        outcome_model: Function(input_params) -> dict of outputs.
        nominal_inputs: Nominal input values.
        perturbations: Perturbation size for each input.
        reference_scales: Reference scales for normalization (optional).

    Returns:
        ForgivenessMetrics with gradients and forgiveness index.
    """
    gradients: dict[str, dict[str, SensitivityGradient]] = {}

    nominal_outputs = outcome_model(**nominal_inputs)

    for input_name, perturbation in perturbations.items():
        if input_name not in nominal_inputs:
            continue

        inputs_plus = dict(nominal_inputs)
        inputs_plus[input_name] = nominal_inputs[input_name] + perturbation

        inputs_minus = dict(nominal_inputs)
        inputs_minus[input_name] = nominal_inputs[input_name] - perturbation

        outputs_plus = outcome_model(**inputs_plus)
        outputs_minus = outcome_model(**inputs_minus)

        gradients[input_name] = {}

        for output_name in nominal_outputs:
            if output_name in outputs_plus and output_name in outputs_minus:
                delta_output = outputs_plus[output_name] - outputs_minus[output_name]
                delta_input = 2 * perturbation
                grad_value = delta_output / delta_input

                gradients[input_name][output_name] = SensitivityGradient(
                    value=grad_value,
                    units="per unit",
                    input_name=input_name,
                    output_name=output_name,
                )

    forgiveness_index = _compute_forgiveness_index(gradients, reference_scales)

    return ForgivenessMetrics(
        gradients=gradients,
        forgiveness_index=forgiveness_index,
    )


def _compute_forgiveness_index(
    gradients: dict[str, dict[str, SensitivityGradient]],
    reference_scales: dict[str, float] | None,
) -> float:
    """Compute normalized forgiveness index (0-1, higher = more forgiving).

    Uses inverse of total gradient magnitude, normalized to [0, 1].
    """
    all_grads = []

    for input_grads in gradients.values():
        for grad in input_grads.values():
            all_grads.append(abs(grad.value))

    if not all_grads:
        return 0.5

    total_sensitivity = sum(all_grads)

    forgiveness = 1.0 / (1.0 + total_sensitivity / 100.0)

    return float(np.clip(forgiveness, 0.0, 1.0))
