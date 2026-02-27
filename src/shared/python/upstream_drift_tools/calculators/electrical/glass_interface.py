"""Glass Properties Interface Module

Interface for external glass property calculators with default models.
Extracted from electrode_advisor.py for better organization.

Author: Chemical Equilibrium Calculator Team
Date: July 8, 2025
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Performance constants
_CACHE_MAX_SIZE = 1000  # Maximum cache entries before LRU eviction
_R_GAS = 8.314  # Gas constant J/(mol·K) - pre-computed to avoid repeated lookups


class GlassPropertiesInterface:
    """Interface for external glass property calculators

    Performance optimizations:
    - LRU cache with configurable max size to prevent unbounded memory growth
    - Pre-computed gas constant for Arrhenius calculations
    - Optimized cache key generation
    """

    def __init__(
        self,
        external_calculator: Callable | None = None,
        cache_max_size: int = _CACHE_MAX_SIZE,
    ) -> None:
        """Initialize the class.

        Args:
            external_calculator: Optional external calculator function
            cache_max_size: Maximum cache entries (default 1000)
        """
        self.external_calculator = external_calculator
        self._cache_max_size = cache_max_size
        self._default_properties = {
            "base_conductivity": 1.0,  # S/m at reference
            "activation_energy": 80000,  # J/mol
            "reference_temp": 1473.15,  # K (1200°C)
            "metal_conductivity": 10000.0,  # S/m - Very high for molten metal
        }
        # Use OrderedDict for LRU cache behavior
        self._temperature_dependent_data: OrderedDict[
            tuple[float, Any, float], float
        ] = OrderedDict()
        self._current_properties: dict[str, Any] = {}  # Store current glass properties
        # Pre-compute reference term for Arrhenius equation
        self._arrhenius_ref_term = (
            -self._default_properties["activation_energy"]
            / _R_GAS
            / self._default_properties["reference_temp"]
        )

    def get_conductivity(
        self,
        temperature_celsius: float,
        composition: dict[str, float] | None = None,
        power_density: float = 0,
        is_metal: bool = False,
    ) -> float:
        """Get electrical conductivity from external calculator or default model.

        Performance: Uses LRU cache with bounded size to prevent memory bloat.
        """
        if is_metal:
            # Metal has very high conductivity, relatively constant with temperature
            return self._default_properties["metal_conductivity"]

        # Build cache key - use frozenset for O(1) hashing instead of sorted tuple
        comp_key = frozenset(composition.items()) if composition else None
        cache_key = (temperature_celsius, comp_key, power_density)

        # Check cache with LRU promotion (move to end on access)
        if cache_key in self._temperature_dependent_data:
            # Move to end for LRU behavior
            self._temperature_dependent_data.move_to_end(cache_key)
            return self._temperature_dependent_data[cache_key]

        if self.external_calculator:
            # Use external calculator if available
            try:
                conductivity = self.external_calculator(
                    temperature_celsius,
                    composition,
                    power_density,
                )
            except (ValueError, TypeError, ArithmeticError) as e:
                logger.warning(f"External calculator failed: {e}. Using default model.")
                conductivity = self._default_conductivity_model(
                    temperature_celsius,
                    power_density,
                )
        else:
            # Use default Arrhenius model for glass
            conductivity = self._default_conductivity_model(
                temperature_celsius,
                power_density,
            )

        # Cache the result with LRU eviction
        self._temperature_dependent_data[cache_key] = conductivity
        # Evict oldest entries if cache exceeds max size
        while len(self._temperature_dependent_data) > self._cache_max_size:
            self._temperature_dependent_data.popitem(last=False)

        return conductivity

    def set_external_calculator(self, calculator: Callable) -> None:
        """Set external calculator function"""
        self.external_calculator = calculator
        # Clear cache when calculator changes
        self._temperature_dependent_data.clear()

    def update_properties(self, properties: dict) -> None:
        """Update current glass properties"""
        self._current_properties.update(properties)

    def get_current_properties(self) -> dict:
        """Get current glass properties"""
        return self._current_properties.copy()

    def _default_conductivity_model(
        self,
        temperature_celsius: float,
        power_density: float = 0,
    ) -> float:
        """Default conductivity model using Arrhenius equation.

        Performance optimizations:
        - Pre-computed reference term (_arrhenius_ref_term)
        - Minimized dict lookups by using local variables
        - Single exp() call path
        """
        # Apply power density heating effect upfront
        temp_kelvin = temperature_celsius + 273.15
        if power_density > 0:
            # Local heating from power density
            temp_kelvin += power_density * 0.0001

        # Arrhenius equation: σ = σ₀ * exp(-Ea/R * (1/T - 1/T_ref))
        # Pre-computed: _arrhenius_ref_term = -Ea/(R * T_ref)
        # So: exponent = -Ea/(R*T) - _arrhenius_ref_term = -Ea/R * (1/T - 1/T_ref)
        props = self._default_properties
        ea_over_r = props["activation_energy"] / _R_GAS
        exponent = -ea_over_r / temp_kelvin + self._arrhenius_ref_term

        return float(props["base_conductivity"] * np.exp(exponent))

    def get_resistivity(
        self,
        temperature_celsius: float,
        composition: dict[str, float] | None = None,
        power_density: float = 0,
        is_metal: bool = False,
    ) -> float:
        """Get electrical resistivity (1/conductivity)"""
        conductivity = self.get_conductivity(
            temperature_celsius,
            composition,
            power_density,
            is_metal,
        )
        return 1.0 / conductivity if conductivity > 0 else float("inf")

    def clear_cache(self) -> None:
        """Clear temperature-dependent data cache"""
        self._temperature_dependent_data.clear()
