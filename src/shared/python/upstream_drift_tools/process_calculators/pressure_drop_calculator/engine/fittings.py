"""Fitting-loss calculations for the pressure drop engine."""

from __future__ import annotations

import logging

from ..models.pressure_drop_data_models import PipeFitting
from ..utils.fitting_loss_coefficients import (
    calculate_two_k_factor,
    get_fitting_k_factor,
)

logger = logging.getLogger(__name__)


def calculate_fitting_pressure_drop(
    fittings: list[PipeFitting],
    density: float,
    velocity: float,
    reynolds_number: float,
    diameter_inches: float,
) -> float:
    """Calculate total pressure drop across fittings and valves."""
    if fittings is None:
        raise ValueError("fittings must be provided")

    total_k = 0.0
    velocity_head = 0.5 * density * (velocity**2)

    for fitting in fittings:
        fitting_type_2k = fitting.fitting_type + "_2k"
        try:
            k_factor = calculate_two_k_factor(
                fitting_type_2k, reynolds_number, diameter_inches
            )
            logger.debug(
                "Using Two-K method for %s: K = %.3f",
                fitting.fitting_type,
                k_factor,
            )
        except (ValueError, KeyError):
            try:
                k_factor = get_fitting_k_factor(fitting.fitting_type)
                logger.debug(
                    "Using standard K for %s: K = %.3f",
                    fitting.fitting_type,
                    k_factor,
                )
            except ValueError:
                k_factor = fitting.k_factor
                logger.warning(
                    "Using provided K-factor for %s: K = %.3f",
                    fitting.fitting_type,
                    k_factor,
                )

        total_k += k_factor * fitting.quantity

    dp_fitting = total_k * velocity_head
    logger.info("Fitting losses: Total K = %.1f, ΔP = %.1f Pa", total_k, dp_fitting)
    return dp_fitting
