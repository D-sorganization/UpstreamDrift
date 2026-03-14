"""
Thermodynamic Properties Calculator
====================================

Standalone ideal gas mixture property calculator for common process gases.
Provides molecular weight, heat capacity, density, enthalpy, entropy,
and Gibbs energy calculations.

This is a lightweight module with no external dependencies beyond the
standard library. For high-accuracy calculations with real-gas effects,
use CoolProp or Cantera directly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Molecular weights (g/mol)
MOLECULAR_WEIGHTS: dict[str, float] = {
    "N2": 28.014,
    "O2": 31.998,
    "CO2": 44.009,
    "H2O": 18.015,
    "CO": 28.010,
    "H2": 2.016,
    "CH4": 16.043,
    "Ar": 39.948,
    "He": 4.003,
    "SO2": 64.066,
    "NO": 30.006,
    "NO2": 46.006,
    "H2S": 34.082,
    "NH3": 17.031,
    "C2H6": 30.070,
    "C3H8": 44.096,
}

# Molar heat capacities at 298 K, J/(mol*K) — ideal gas
MOLAR_CP_298: dict[str, float] = {
    "N2": 29.12,
    "O2": 29.38,
    "CO2": 37.13,
    "H2O": 33.59,
    "CO": 29.14,
    "H2": 28.84,
    "CH4": 35.69,
    "Ar": 20.79,
    "He": 20.79,
    "SO2": 39.87,
    "NO": 29.86,
    "NO2": 37.20,
    "H2S": 34.23,
    "NH3": 35.06,
    "C2H6": 52.49,
    "C3H8": 73.60,
}

R_GAS = 8.314  # J/(mol*K)


@dataclass
class ThermoResult:
    """Result of a thermodynamic property calculation."""

    temperature_k: float
    pressure_pa: float
    molecular_weight_g_mol: float
    molar_volume_m3_mol: float
    density_kg_m3: float
    enthalpy_j_mol: float
    entropy_j_molk: float
    gibbs_energy_j_mol: float
    cp_j_molk: float
    cv_j_molk: float
    gamma: float
    database_used: str = "ideal_gas"


class ThermoPropertiesCalculator:
    """Ideal gas mixture thermodynamic properties calculator.

    Usage::

        calc = ThermoPropertiesCalculator()
        result = calc.calculate(
            temperature_c=500.0,
            pressure_kpa=101.325,
            composition={"N2": 79, "O2": 21},
        )
        logger.info(result.density_kg_m3)
    """

    def calculate(
        self,
        temperature_c: float,
        pressure_kpa: float,
        composition: dict[str, float],
    ) -> ThermoResult:
        """Calculate mixture properties at given T, P, and composition.

        Args:
            temperature_c: Temperature in Celsius.
            pressure_kpa: Pressure in kPa.
            composition: Species mole fractions (need not sum to 1; will be normalized).

        Returns:
            ThermoResult with all computed properties.
        """
        assert temperature_c is not None, "temperature_c must be provided"
        temp_k = temperature_c + 273.15
        pressure_pa = pressure_kpa * 1000.0

        # Normalize composition
        total = sum(composition.values())
        if total <= 0:
            total = 1.0
        fractions = {k: v / total for k, v in composition.items()}

        # Mixture molecular weight
        mix_mw = sum(
            fractions.get(s, 0) * MOLECULAR_WEIGHTS.get(s, 28.0) for s in fractions
        )

        # Mixture molar Cp
        mix_cp = sum(fractions.get(s, 0) * MOLAR_CP_298.get(s, 29.0) for s in fractions)
        mix_cv = mix_cp - R_GAS
        gamma = mix_cp / mix_cv if mix_cv > 0 else 1.4

        # Ideal gas: PV = nRT
        molar_volume = R_GAS * temp_k / pressure_pa  # m^3/mol
        density = (mix_mw / 1000.0) / molar_volume  # kg/m^3

        # Enthalpy relative to 298.15 K reference
        enthalpy = mix_cp * (temp_k - 298.15)

        # Entropy relative to 298.15 K, 101325 Pa reference
        entropy = mix_cp * math.log(temp_k / 298.15) - R_GAS * math.log(
            pressure_pa / 101325.0
        )

        # Gibbs free energy
        gibbs = enthalpy - temp_k * entropy

        return ThermoResult(
            temperature_k=temp_k,
            pressure_pa=pressure_pa,
            molecular_weight_g_mol=mix_mw,
            molar_volume_m3_mol=molar_volume,
            density_kg_m3=density,
            enthalpy_j_mol=enthalpy,
            entropy_j_molk=entropy,
            gibbs_energy_j_mol=gibbs,
            cp_j_molk=mix_cp,
            cv_j_molk=mix_cv,
            gamma=gamma,
        )
