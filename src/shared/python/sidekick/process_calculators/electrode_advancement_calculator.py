"""
Electrode Advancement Calculator
===============================
Calculates electrode consumption and slip rates for arc furnaces.
"""

import warnings

from src.shared.python.contracts import check_non_negative, require_positive

try:
    from tools_core.electrode_advisor import (
        ElectrodeAdvancementCalculator as _RustCalculator,
    )
except ImportError:
    _RustCalculator = None


class _PyElectrodeAdvancementCalculator:
    """Calculates electrode consumption and advancement."""

    def __init__(self, consumption_rate: float = 0.5) -> None:
        """Initialize parameters.

        Args:
            consumption_rate: Electrode consumption rate in inches per kAh.
                Must be positive.
        """
        require_positive(consumption_rate, "consumption_rate")
        self.consumption_rate = consumption_rate
        warnings.warn(
            "Using pure-Python ElectrodeAdvancementCalculator",
            DeprecationWarning,
            stacklevel=2,
        )

    def calculate_consumption(self, current_ka: float, time_hrs: float) -> float:
        """
        Calculate electrode consumption.

        Args:
            current_ka: Current in kA (must be non-negative)
            time_hrs: Time in hours (must be non-negative)

        Returns:
            Consumption in inches
        """
        check_non_negative(current_ka, "current_ka")
        check_non_negative(time_hrs, "time_hrs")
        return self.consumption_rate * current_ka * time_hrs


if _RustCalculator is not None:
    ElectrodeAdvancementCalculator = _RustCalculator
else:
    ElectrodeAdvancementCalculator = _PyElectrodeAdvancementCalculator
