"""Sensitivity-analysis helpers for the A3 data-fitting pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from .data_fitting_models import SensitivityResult

logger = get_logger(__name__)


class SensitivityAnalyzer:
    """Perform sensitivity analysis on model parameters."""

    def __init__(
        self,
        perturbation_size: float = 0.01,
    ) -> None:
        if not (perturbation_size is not None):
            raise ValueError("perturbation_size must be provided")
        self.perturbation_size = perturbation_size

    def compute_sensitivity(
        self,
        model_func: Any,
        parameter_name: str,
        nominal_value: float,
        output_metric: str,
    ) -> SensitivityResult:
        """Compute sensitivity of output to parameter."""
        if not (parameter_name is not None):
            raise ValueError("parameter_name must be provided")
        delta = nominal_value * self.perturbation_size

        try:
            output_up = model_func({parameter_name: nominal_value + delta})[
                output_metric
            ]
            output_down = model_func({parameter_name: nominal_value - delta})[
                output_metric
            ]
            output_nominal = model_func({parameter_name: nominal_value})[output_metric]
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Sensitivity computation failed: {e}")
            return SensitivityResult(
                parameter_name=parameter_name,
                nominal_value=nominal_value,
                sensitivity_index=0.0,
                partial_derivative=0.0,
                confidence_interval=(0.0, 0.0),
                elasticity=0.0,
            )

        partial = (output_up - output_down) / (2 * delta)
        elasticity = (
            (partial * nominal_value / output_nominal) if output_nominal != 0 else 0.0
        )
        output_range = abs(output_up - output_down)
        sensitivity_index = output_range / (2 * delta) if delta != 0 else 0.0
        ci_half_width = abs(partial) * delta * 2
        ci = (output_nominal - ci_half_width, output_nominal + ci_half_width)

        return SensitivityResult(
            parameter_name=parameter_name,
            nominal_value=nominal_value,
            sensitivity_index=float(sensitivity_index),
            partial_derivative=float(partial),
            confidence_interval=ci,
            elasticity=float(elasticity),
        )

    def sensitivity_report(
        self,
        sensitivities: list[SensitivityResult],
    ) -> dict[str, Any]:
        """Generate sensitivity analysis report."""
        if not (sensitivities is not None):
            raise ValueError("sensitivities must be provided")
        if not sensitivities:
            return {"error": "No sensitivity data"}

        sorted_sens = sorted(
            sensitivities,
            key=lambda s: abs(s.sensitivity_index),
            reverse=True,
        )

        return {
            "total_parameters": len(sensitivities),
            "most_sensitive": sorted_sens[0].parameter_name if sorted_sens else None,
            "least_sensitive": sorted_sens[-1].parameter_name if sorted_sens else None,
            "rankings": [
                {
                    "rank": i + 1,
                    "parameter": s.parameter_name,
                    "sensitivity_index": s.sensitivity_index,
                    "elasticity": s.elasticity,
                }
                for i, s in enumerate(sorted_sens)
            ],
            "summary_statistics": {
                "mean_sensitivity": float(
                    np.mean([s.sensitivity_index for s in sensitivities])
                ),
                "max_sensitivity": float(
                    max(s.sensitivity_index for s in sensitivities)
                ),
                "mean_elasticity": float(
                    np.mean([abs(s.elasticity) for s in sensitivities])
                ),
            },
        }
