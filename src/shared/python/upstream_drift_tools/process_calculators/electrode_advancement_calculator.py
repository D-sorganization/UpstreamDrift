"""
Electrode Advancement Calculator
===============================
Calculates electrode consumption and slip rates for arc furnaces.
"""

import logging

logger = logging.getLogger(__name__)

# Default placeholder consumption rate [inches per kAh].
# This value is uncalibrated and should be overridden with furnace-specific
# calibration data for production use.
_DEFAULT_CONSUMPTION_RATE = 0.5


class ElectrodeAdvancementCalculator:
    """Calculates electrode consumption and advancement."""

    def __init__(
        self,
        consumption_rate: float | None = None,
    ) -> None:
        """Initialize parameters.

        Args:
            consumption_rate: Electrode consumption rate in inches per kAh.
                If None, uses the uncalibrated default (0.5) and logs a warning.
        """
        if consumption_rate is None:
            logger.warning(
                "Using uncalibrated default consumption rate "
                f"({_DEFAULT_CONSUMPTION_RATE} in/kAh). "
                "Provide furnace-specific calibration data for accurate results."
            )
            self.consumption_rate = _DEFAULT_CONSUMPTION_RATE
        else:
            self.consumption_rate = consumption_rate

    def calculate_consumption(self, current_ka: float, time_hrs: float) -> float:
        """
        Calculate electrode consumption.

        Args:
            current_ka: Current in kA
            time_hrs: Time in hours

        Returns:
            Consumption in inches
        """
        return self.consumption_rate * current_ka * time_hrs
