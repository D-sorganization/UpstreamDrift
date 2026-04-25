"""Analysis utility functions shared by optimization and multi-parameter sweeps.

This module provides the ``evaluate_output`` helper that both
``optimization.py`` and ``multi_param_analysis.py`` depend on.
"""

from __future__ import annotations

import logging
from typing import Any

from .constants import (
    ATOL_ZERO,
    COMPRESSION_HIGH_POWER_HP,
    COMPRESSION_HIGH_PRESSURE_BAR,
    COMPRESSION_MIN_EFFICIENCY,
    COMPRESSION_TEMP_CRITICAL_K,
    COMPRESSION_TEMP_WARNING_K,
)

logger = logging.getLogger(__name__)


def evaluate_output(
    engine: Any,
    base_params: dict[str, float],
    manual_hhv: float,
    output_variable: str,
    overrides: dict[str, float] | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Run the engine with merged parameters and extract a named output.

    Parameters
    ----------
    engine:
        Calculation engine exposing a ``calculate(**params)`` method that
        returns a JSON-serialisable dictionary.
    base_params:
        Baseline parameter dictionary.
    manual_hhv:
        Higher-heating-value override supplied by the user.
    output_variable:
        Key to extract from the engine result dictionary.
    overrides:
        Parameter overrides applied on top of *base_params*.

    Returns
    -------
    tuple[float, dict, dict]
        ``(output_value, state_dict, composition_dict)`` where
        *state_dict* and *composition_dict* are sub-dicts from the engine
        result (empty dicts if not present).
    """
    if not (base_params is not None):
        raise ValueError("base_params must be provided")
    if not (base_params is not None):
        raise ValueError("base_params must be provided")
    params = {**base_params}
    if overrides:
        params.update(overrides)

    # Inject HHV if the engine expects it
    if manual_hhv > 0:
        params["manual_hhv"] = manual_hhv

    try:
        result = engine.calculate(**params)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        logger.warning("Engine calculation failed: %s", exc)
        return 0.0, {}, {}

    if not isinstance(result, dict):
        return 0.0, {}, {}

    output_value = float(result.get(output_variable, 0.0))

    state: dict[str, float] = result.get("state", {})
    composition: dict[str, float] = result.get("composition", {})

    return output_value, state, composition


def evaluate_compression_result(
    compression_result: dict[str, Any],
) -> dict[str, Any]:
    """Summarize compression-train concerns, warnings, and recommendations."""
    if compression_result is None:
        raise ValueError("compression_result must be provided")

    concerns: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    final_temp = compression_result["final_temperature"]
    final_pressure = compression_result["final_pressure"]
    total_power = compression_result["total_power_hp"]

    if final_temp > COMPRESSION_TEMP_WARNING_K:
        concerns.append("High final temperature may cause material degradation")
        recommendations.append("Consider additional intercooling or heat exchangers")

    if final_temp > COMPRESSION_TEMP_CRITICAL_K:
        warnings.append("CRITICAL: Temperature exceeds safe operating limits")

    if final_pressure > COMPRESSION_HIGH_PRESSURE_BAR:
        concerns.append("High pressure requires special equipment and safety measures")
        recommendations.append("Verify equipment pressure ratings and safety systems")

    if total_power > COMPRESSION_HIGH_POWER_HP:
        concerns.append("High power requirement - consider multiple compressors")
        recommendations.append("Evaluate economic feasibility of compression train")

    total_water_dropout = sum(
        stage["water_dropout"]["water_dropout"]
        for stage in compression_result["stages"]
    )
    if total_water_dropout > ATOL_ZERO:
        warnings.append(f"Water dropout detected: {total_water_dropout:.2f} mol%")
        recommendations.append("Install water knockout drums and drainage systems")

    isentropic_stages = [
        stage
        for stage in compression_result["stages"]
        if stage["work_isentropic"] is not None
    ]
    if isentropic_stages:
        efficiencies = [
            stage["work_actual"] / stage["work_isentropic"]
            for stage in isentropic_stages
        ]
        average_efficiency = sum(efficiencies) / len(efficiencies)
        if average_efficiency < COMPRESSION_MIN_EFFICIENCY:
            concerns.append("Low compression efficiency detected")
            recommendations.append("Consider compressor maintenance or replacement")
    else:
        average_efficiency = None

    return {
        "concerns": concerns,
        "warnings": warnings,
        "recommendations": recommendations,
        "total_water_dropout": total_water_dropout,
        "average_efficiency": average_efficiency,
    }


__all__ = ["evaluate_output", "evaluate_compression_result"]
