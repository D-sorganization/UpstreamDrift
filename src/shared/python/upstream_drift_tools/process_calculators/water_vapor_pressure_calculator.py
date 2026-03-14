"""
Water Vapor Pressure Calculator
==============================
Wrapper for water vapor pressure calculations for backward compatibility.
"""

from .syngas_water_calculator import SyngasWaterCalculator


class WaterVaporPressureCalculator:
    """Calculates water vapor pressure using various methods."""

    def __init__(self) -> None:
        """Initialize the calculator."""
        self.calculator = SyngasWaterCalculator()

    def calculate_vapor_pressure(
        self, temperature_c: float, method: str = "auto"
    ) -> float:
        """
        Calculate vapor pressure in Pa.

        Args:
            temperature_c: Temperature in Celsius
            method: Calculation method ('auto', 'buck', 'antoine', etc.)

        Returns:
            Vapor pressure in Pa
        """
        assert temperature_c is not None, "temperature_c must be provided"
        pressure, _ = self.calculator.calculate_vapor_pressure(temperature_c, method)
        return pressure
