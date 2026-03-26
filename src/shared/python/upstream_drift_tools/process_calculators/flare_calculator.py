"""Flare Calculator
===============

Core calculation engine for flare system design and analysis.
"""

import math
from dataclasses import dataclass
from typing import Final

from .constants import (
    BAR_TO_PA,
    FLARE_BASE_EFFICIENCY,
    FLARE_CO_EFFICIENCY_PENALTY,
    FLARE_CO_THRESHOLD,
    FLARE_COLD_TEMP_K,
    FLARE_COLD_TEMP_PENALTY,
    FLARE_FLAME_EMISSIVITY,
    FLARE_H2_EFFICIENCY_BOOST,
    FLARE_H2_THRESHOLD,
    FLARE_H2S_EFFICIENCY_PENALTY,
    FLARE_H2S_THRESHOLD,
    FLARE_HOT_TEMP_BOOST,
    FLARE_HOT_TEMP_K,
    FLARE_MAX_EFFICIENCY,
    FLARE_MAX_EXIT_VELOCITY,
    FLARE_MIN_EFFICIENCY,
    FLARE_MIN_HEIGHT,
    FLARE_SAFE_RADIATION_INTENSITY,
    G_MOL_TO_KG_MOL,
    R_UNIVERSAL,
    RADIATION_COMFORT,
    RADIATION_DAMAGE,
    RADIATION_LETHAL,
    RADIATION_SAFE,
    SECONDS_PER_HOUR,
)

# Standard Gas Properties (Molecular Weight [g/mol], Heating Value [kJ/kg], Cp [kJ/kg-K])
GAS_PROPERTIES: Final[dict[str, dict[str, float]]] = {
    "H2": {"mw": 2.016, "hv": 119930, "cp": 14.3},
    "CO": {"mw": 28.01, "hv": 10100, "cp": 1.04},
    "CH4": {"mw": 16.04, "hv": 50010, "cp": 2.22},
    "C2H6": {"mw": 30.07, "hv": 47520, "cp": 1.75},
    "C3H8": {"mw": 44.10, "hv": 46360, "cp": 1.67},
    "C4H10": {"mw": 58.12, "hv": 45720, "cp": 1.66},
    "H2S": {"mw": 34.08, "hv": 16500, "cp": 1.05},
    "N2": {"mw": 28.01, "hv": 0, "cp": 1.04},
    "CO2": {"mw": 44.01, "hv": 0, "cp": 0.84},
    "H2O": {"mw": 18.02, "hv": 0, "cp": 1.87},
}


@dataclass
class FlareDesign:
    """Flare design parameters."""

    height: float  # m
    diameter: float  # m
    exit_velocity: float  # m/s
    heat_release: float  # kW
    radiation_intensity: float  # kW/m²


class FlareCalculator:
    """Core flare calculation engine."""

    def __init__(self) -> None:
        """Initialize the flare calculator."""
        self.gas_properties = GAS_PROPERTIES

    def calculate_flare_size(
        self,
        total_flow: float,  # kg/hr
        gas_composition: dict[str, float],  # mol %
        temperature: float,  # K
        pressure: float,  # bar
    ) -> FlareDesign:
        """Calculate flare size based on flow conditions.

        Args:
            total_flow: Total gas flow rate (kg/hr)
            gas_composition: Gas composition (mol%)
            temperature: Gas temperature (K)
            pressure: Gas pressure (bar)

        Returns:
            FlareDesign object with calculated parameters
        """
        # DbC preconditions
        if not (total_flow > 0):
            raise ValueError(f"total_flow must be positive, got {total_flow}")
        if not (temperature > 0):
            raise ValueError(f"temperature must be positive (K), got {temperature}")
        if not (pressure > 0):
            raise ValueError(f"pressure must be positive (bar), got {pressure}")
        if not (len(gas_composition) > 0):
            raise ValueError("gas_composition must not be empty")

        # Normalize composition to fractions
        total_comp = sum(gas_composition.values())
        if total_comp == 0:
            comp_fractions = dict.fromkeys(gas_composition, 0.0)
        else:
            comp_fractions = {k: v / total_comp for k, v in gas_composition.items()}

        # Calculate mixture properties
        mix_mw = sum(
            comp_fractions[gas] * self.gas_properties[gas]["mw"]
            for gas in comp_fractions
        )
        mix_hv = sum(
            comp_fractions[gas] * self.gas_properties[gas]["hv"]
            for gas in comp_fractions
        )

        # Calculate heat release
        # kg/hr * kJ/kg * (1 hr / 3600 s) = kJ/s = kW
        heat_release = total_flow * mix_hv / SECONDS_PER_HOUR

        # Calculate gas density [kg/m³] using Ideal Gas Law
        # P = density * R_specific * T
        # density = P / (R_specific * T)
        # R_specific = R_universal / MW
        # Pressure in Pa, MW in kg/mol (g/mol / 1000)
        pressure_pa = pressure * BAR_TO_PA  # bar to Pa
        mix_mw_kg = mix_mw / G_MOL_TO_KG_MOL  # g/mol to kg/mol

        if temperature > 0:
            gas_density = pressure_pa / ((R_UNIVERSAL / mix_mw_kg) * temperature)
        else:
            gas_density = 1.0  # Fallback

        # Calculate flare diameter (simplified API 521 method)
        # Exit velocity target: 0.5 Mach for smokeless operation or max 170 m/s
        # Using simplified target velocity
        target_velocity = FLARE_MAX_EXIT_VELOCITY  # m/s

        # Calculate required cross-sectional area
        mass_flow_kg_s = total_flow / SECONDS_PER_HOUR
        if gas_density > 0 and target_velocity > 0:
            area = mass_flow_kg_s / (gas_density * target_velocity)
            diameter = math.sqrt(4 * area / math.pi)
        else:
            diameter = 0.0

        # Calculate flare height (simplified radiation method)
        # Target radiation intensity: 1.6 kW/m² at ground level for safe access
        target_radiation = FLARE_SAFE_RADIATION_INTENSITY  # kW/m²
        emissivity = FLARE_FLAME_EMISSIVITY  # Typical for clean hydrocarbon flames

        # Simplified height calculation from point source model
        # I = (tau * F * Q) / (4 * pi * R^2)
        # Assuming tau(transmissivity) integrated into emissivity or simplistic model
        # D = sqrt((tau * F * Q) / (4 * pi * K))
        # Here matching original logic: H = sqrt(em * Q / (4 * pi * I))
        if target_radiation > 0:
            height = math.sqrt(
                emissivity * heat_release / (4 * math.pi * target_radiation)
            )
        else:
            height = 0.0

        # Ensure minimum height
        height = max(height, FLARE_MIN_HEIGHT)

        result = FlareDesign(
            height=height,
            diameter=diameter,
            exit_velocity=target_velocity,
            heat_release=heat_release,
            radiation_intensity=target_radiation,
        )
        # DbC postconditions
        if not (result.height >= FLARE_MIN_HEIGHT):
            raise ValueError(()
            f"Flare height must be >= minimum ({FLARE_MIN_HEIGHT}), got {result.height}"
        )
        if not (result.diameter >= 0):
            raise ValueError(()
            f"Flare diameter must be non-negative, got {result.diameter}"
        )
        return result

    def calculate_radiation_zones(self, flare_design: FlareDesign) -> dict[str, float]:
        """Calculate radiation zones around the flare.

        Args:
            flare_design: Flare design parameters

        Returns:
            Dictionary with zone distances (m)
        """
        if not (flare_design is not None):
            raise ValueError("flare_design must be provided")
        if not (flare_design is not None):
            raise ValueError("flare_design must be provided")
        zones = {
            "lethal": 0.0,  # 37.5 kW/m²
            "damage": 0.0,  # 12.5 kW/m²
            "safe": 0.0,  # 1.6 kW/m²
            "comfort": 0.0,  # 0.5 kW/m²
        }

        emissivity = FLARE_FLAME_EMISSIVITY
        heat_release = flare_design.heat_release

        # Calculate distances for each radiation level
        radiation_levels = {
            "lethal": RADIATION_LETHAL,
            "damage": RADIATION_DAMAGE,
            "safe": RADIATION_SAFE,
            "comfort": RADIATION_COMFORT,
        }

        for zone, level in radiation_levels.items():
            if level > 0:
                # Distance based on point source model
                distance = math.sqrt(emissivity * heat_release / (4 * math.pi * level))
                zones[zone] = distance

        return zones

    def calculate_combustion_efficiency(
        self,
        gas_composition: dict[str, float],
        temperature: float,
        pressure: float,
    ) -> float:
        """Calculate combustion efficiency.

        Args:
            gas_composition: Gas composition (mol%)
            temperature: Gas temperature (K)
            pressure: Gas pressure (bar)

        Returns:
            Combustion efficiency (0-1)
        """
        # Simplified efficiency calculation
        if not (gas_composition is not None):
            raise ValueError("gas_composition must be provided")
        if not (gas_composition is not None):
            raise ValueError("gas_composition must be provided")
        efficiency = FLARE_BASE_EFFICIENCY  # Base efficiency

        # Normalize factors
        total = sum(gas_composition.values()) or 1.0

        # Factors based on mole fractions
        h2_frac = gas_composition.get("H2", 0) / total
        co_frac = gas_composition.get("CO", 0) / total
        h2s_frac = gas_composition.get("H2S", 0) / total

        if h2_frac > FLARE_H2_THRESHOLD:
            efficiency += FLARE_H2_EFFICIENCY_BOOST

        if co_frac > FLARE_CO_THRESHOLD:
            efficiency -= FLARE_CO_EFFICIENCY_PENALTY

        if h2s_frac > FLARE_H2S_THRESHOLD:
            efficiency -= FLARE_H2S_EFFICIENCY_PENALTY

        # Temperature effects
        if temperature < FLARE_COLD_TEMP_K:
            efficiency -= FLARE_COLD_TEMP_PENALTY
        elif temperature > FLARE_HOT_TEMP_K:
            efficiency += FLARE_HOT_TEMP_BOOST

        return max(FLARE_MIN_EFFICIENCY, min(FLARE_MAX_EFFICIENCY, efficiency))
