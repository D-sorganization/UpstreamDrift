"""Result formatting and presentation helpers for pressure drop calculations."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


PRESSURE_CONVERSIONS_TO_PA = {
    "Pa": 1.0,
    "kPa": 1000.0,
    "MPa": 1e6,
    "bar": 1e5,
    "mbar": 100.0,
    "atm": 101325.0,
    "psi": 6894.76,
    "psia": 6894.76,
    "psig": 6894.76,
}


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between units."""
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    if from_unit == "K":
        temp_k = value
    elif from_unit == "C":
        temp_k = value + 273.15
    elif from_unit == "F":
        temp_k = (value - 32) * 5 / 9 + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
    if to_unit == "K":
        return temp_k
    if to_unit == "C":
        return temp_k - 273.15
    if to_unit == "F":
        return (temp_k - 273.15) * 9 / 5 + 32
    raise ValueError(f"Unknown temperature unit: {to_unit}")


def convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """Convert pressure between units."""
    if from_unit not in PRESSURE_CONVERSIONS_TO_PA:
        raise ValueError(f"Unknown pressure unit: {from_unit}")
    if to_unit not in PRESSURE_CONVERSIONS_TO_PA:
        raise ValueError(f"Unknown pressure unit: {to_unit}")
    pressure_pa = value * PRESSURE_CONVERSIONS_TO_PA[from_unit]
    return pressure_pa / PRESSURE_CONVERSIONS_TO_PA[to_unit]


def format_results(results: Any) -> dict[str, Any]:
    """Format engine results into the public dictionary contract."""
    return {
        "pressure_drop_pa": results.total_pressure_drop,
        "pressure_drop_bar": results.total_pressure_drop / 1e5,
        "pressure_drop_psi": results.total_pressure_drop / 6894.76,
        "pressure_drop_kpa": results.total_pressure_drop / 1000.0,
        "friction_loss_pa": results.friction_pressure_drop,
        "friction_loss_bar": results.friction_pressure_drop / 1e5,
        "fitting_loss_pa": results.fitting_pressure_drop,
        "fitting_loss_bar": results.fitting_pressure_drop / 1e5,
        "elevation_loss_pa": results.elevation_pressure_drop,
        "outlet_pressure_pa": results.outlet_pressure,
        "outlet_pressure_bar": results.outlet_pressure / 1e5,
        "outlet_pressure_psi": results.outlet_pressure / 6894.76,
        "friction_factor": results.friction_factor,
        "reynolds_number": results.flow_properties.reynolds_number,
        "flow_velocity_m_s": results.flow_properties.velocity,
        "flow_velocity_ft_s": results.flow_properties.velocity * 3.28084,
        "mach_number": results.flow_properties.mach_number,
        "flow_regime": results.flow_regime,
        "density_kg_m3": results.flow_properties.density,
        "viscosity_pa_s": results.flow_properties.viscosity,
        "compressibility_factor": results.flow_properties.compressibility_factor,
        "molecular_weight": results.flow_properties.molecular_weight,
        "erosional_velocity_m_s": results.erosional_velocity,
        "erosion_ratio": results.erosion_ratio,
        "erosion_ratio_percent": results.erosion_ratio * 100,
        "pressure_drop_per_100ft_pa": results.pressure_drop_per_100ft,
        "velocity_pressure_pa": results.velocity_pressure,
        "warnings": results.warnings,
    }


def print_results(
    results: dict[str, Any],
    title: str = "PRESSURE DROP CALCULATION RESULTS",
    show_recommendations: bool = True,
) -> None:
    """Print results in a concise formatted log table."""
    logger.info("%s", title)
    logger.info("Pressure drop: %.4f bar", results["pressure_drop_bar"])
    logger.info("Outlet pressure: %.4f bar", results["outlet_pressure_bar"])
    logger.info("Flow regime: %s", results["flow_regime"])
    logger.info("Velocity: %.2f m/s", results["flow_velocity_m_s"])
    if results.get("warnings"):
        for warning in results["warnings"]:
            logger.warning("%s", warning)
    if show_recommendations:
        for recommendation in generate_recommendations(results):
            logger.info("Recommendation: %s", recommendation)


def generate_recommendations(results: dict[str, Any]) -> list[str]:
    """Generate engineering recommendations based on calculation results."""
    recommendations: list[str] = []
    dp_ratio = results["pressure_drop_pa"] / (
        results["outlet_pressure_pa"] + results["pressure_drop_pa"]
    )
    if dp_ratio > 0.20:
        recommendations.append(
            f"High pressure drop ({dp_ratio * 100:.0f}% of inlet). Consider: larger pipe diameter, shorter pipe run, or fewer fittings."
        )
    erosion_ratio = results["erosion_ratio"]
    if erosion_ratio > 0.8:
        recommendations.append(
            "Velocity exceeds 80% of erosional limit. Consider larger pipe diameter to reduce velocity and extend pipe life."
        )
    elif erosion_ratio > 0.5:
        recommendations.append(
            "Velocity is 50-80% of erosional limit. Monitor pipe condition and consider velocity reduction for longer service life."
        )
    if results["fitting_loss_pa"] > results["friction_loss_pa"]:
        recommendations.append(
            "Fitting losses exceed pipe friction. Consider using long-radius elbows, full-port valves, or reducing number of fittings."
        )
    if results["mach_number"] > 0.3:
        recommendations.append(
            f"High Mach number ({results['mach_number']:.3f}). Compressibility effects significant. Verify calculations and consider acoustic vibration analysis."
        )
    if results["reynolds_number"] < 4000:
        recommendations.append(
            f"Low Reynolds number ({results['reynolds_number']:.0f}). Flow may be transitional or laminar - friction factor has higher uncertainty in this regime."
        )
    if results["reynolds_number"] > 1e7:
        recommendations.append(
            f"Very high Reynolds number ({results['reynolds_number']:.0e}). Ensure turbulent flow correlations are valid. Consider CFD analysis for critical applications."
        )
    return recommendations
