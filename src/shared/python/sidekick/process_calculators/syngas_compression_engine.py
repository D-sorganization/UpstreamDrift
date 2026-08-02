# ruff: noqa: E501
"""Syngas Compression Calculation Engine.

Pure calculation logic extracted from ``syngas_compression_calculator.py`` (issue #1806).
No GUI dependencies — can be used headlessly for batch calculations, testing, or API serving.

Classes
-------
CompressionStage
    Dataclass describing a single compression stage.
SyngasCompressionEngine
    Core calculation engine: mixture properties, water dropout, compression work,
    multistage analysis, and process-condition diagnostics.
"""  # noqa: E501

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from .constants import (
    ATOL_ZERO,
    BAR_TO_PA,
    CELSIUS_TO_KELVIN_OFFSET,
    COMPRESSION_HIGH_POWER_HP,
    COMPRESSION_HIGH_PRESSURE_BAR,
    COMPRESSION_MIN_EFFICIENCY,
    COMPRESSION_TEMP_CRITICAL_K,
    COMPRESSION_TEMP_WARNING_K,
    DEFAULT_GAMMA_DIATOMIC,
    INTERCOOLER_OUTLET_TEMP_K,
    R_GAS_J_MOL_K,
    SECONDS_PER_HOUR,
    WATTS_PER_HP,
)
from .syngas_water_calculator import SyngasWaterCalculator

__all__ = [
    "CompressionStage",
    "SyngasCompressionEngine",
]

# Validation utility — graceful fallback for standalone use
try:
    from integrated_process_simulator.utilities.validation import (
        validate_gas_composition,
    )
except ImportError:

    def validate_gas_composition(
        comp: dict[str, float],
        auto_normalize: bool = False,
    ) -> dict[str, float]:
        """Simple validation and normalization fallback."""
        if not comp:
            raise ValueError("Empty composition")
        total = sum(comp.values())
        if auto_normalize and total > 0:
            return {k: v / total for k, v in comp.items()}
        if abs(total - 1.0) > 0.01 and abs(total - 100.0) > 1.0:
            raise ValueError(f"Composition sum {total} not normalized")
        return comp


# Species database — graceful fallback for standalone use
try:
    from integrated_process_simulator.calculators.thermodynamic_properties.species_database import (  # noqa: E501
        get_species_database,
    )
except ImportError:

    @dataclass
    class _SpeciesData:
        molecular_weight: float  # kg/mol
        critical_temperature: float  # K
        critical_pressure: float  # Pa

    # Source: NIST / Perry's Chemical Engineers' Handbook (approximate values)
    _SPECIES_TABLE: dict[str, _SpeciesData] = {
        "CO": _SpeciesData(0.02801, 132.9, 3.499e6),
        "CO2": _SpeciesData(0.04401, 304.2, 7.376e6),
        "H2": _SpeciesData(0.00202, 33.2, 1.297e6),
        "H2O": _SpeciesData(0.01802, 647.1, 22.064e6),
        "CH4": _SpeciesData(0.01604, 190.6, 4.604e6),
        "N2": _SpeciesData(0.02801, 126.2, 3.390e6),
        "O2": _SpeciesData(0.03200, 154.6, 5.046e6),
        "H2S": _SpeciesData(0.03408, 373.5, 8.963e6),
        "Ar": _SpeciesData(0.03995, 150.8, 4.874e6),
    }

    class _MinimalSpeciesDB:
        def get_molecular_weight(self, species: str) -> float | None:
            s = _SPECIES_TABLE.get(species)
            return s.molecular_weight if s else None

        def get_species(self, species: str) -> _SpeciesData | None:
            return _SPECIES_TABLE.get(species)

    def get_species_database() -> _MinimalSpeciesDB:
        return _MinimalSpeciesDB()


try:
    from integrated_process_simulator.utilities.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CompressionStage:
    """Compression stage parameters."""

    inlet_pressure: float  # bar
    outlet_pressure: float  # bar
    inlet_temperature: float  # K
    efficiency: float  # isentropic efficiency
    compression_type: str  # 'isentropic', 'polytropic', 'isothermal'


# ---------------------------------------------------------------------------
# Calculation engine
# ---------------------------------------------------------------------------


class SyngasCompressionEngine:
    """Core syngas compression calculation engine."""

    # Approximate heat capacity ratios (gamma) for compressor sizing
    _APPROX_GAMMA = {
        "H2": 1.41,
        "CO": 1.40,
        "CO2": 1.30,
        "CH4": 1.32,
        "N2": 1.40,
        "H2O": 1.33,
        "Ar": 1.67,
    }

    def __init__(self) -> None:
        """Initialize the engine."""
        self.water_calculator = SyngasWaterCalculator()
        self.species_db = get_species_database()
        self.R = R_GAS_J_MOL_K

    def calculate_mixture_properties(
        self,
        composition: dict[str, float],
    ) -> dict[str, Any]:
        """Calculate mixture properties from component composition."""
        if composition is None:
            raise ValueError("composition must be provided")
        mole_fractions = validate_gas_composition(composition, auto_normalize=True)

        mix_mw = 0.0
        mix_tc = 0.0
        mix_pc = 0.0
        mix_gamma = 0.0

        for comp, frac in mole_fractions.items():
            species = self.species_db.get_species(comp)
            if not species:
                logger.warning("Species %s not found in database, using defaults", comp)
                continue

            mix_mw += frac * species.molecular_weight * 1000.0
            mix_tc += frac * species.critical_temperature
            mix_pc += frac * (species.critical_pressure / 100000.0)
            gamma = self._APPROX_GAMMA.get(comp, DEFAULT_GAMMA_DIATOMIC)
            mix_gamma += frac * gamma

        return {
            "molecular_weight": mix_mw,
            "critical_temperature": mix_tc,
            "critical_pressure": mix_pc,
            "heat_capacity_ratio": mix_gamma,
            "mole_fractions": mole_fractions,
        }

    def calculate_water_dropout(
        self,
        temperature: float,
        pressure: float,
        water_content: float,
    ) -> dict[str, float]:
        """Calculate water dropout during compression."""
        if pressure <= 0:
            raise ValueError(f"pressure must be > 0, got {pressure}")

        temperature_c = temperature - CELSIUS_TO_KELVIN_OFFSET
        water_vp_pa, _ = self.water_calculator.calculate_vapor_pressure(
            temperature_c, method="auto"
        )
        water_vp_bar = water_vp_pa / BAR_TO_PA

        if water_vp_bar <= 0:
            raise ValueError(
                f"water vapor pressure must be > 0 bar, got {water_vp_bar} "
                f"(at T={temperature} K)"
            )

        relative_humidity = (water_content / 100) * pressure / water_vp_bar

        if relative_humidity > 1.0:
            max_water_vapor = water_vp_bar / pressure * 100
            water_dropout = water_content - max_water_vapor
            condensation_rate = water_dropout / water_content * 100
        else:
            water_dropout = 0.0
            condensation_rate = 0.0

        return {
            "water_vapor_pressure": water_vp_bar,
            "relative_humidity": relative_humidity,
            "water_dropout": water_dropout,
            "condensation_rate": condensation_rate,
            "max_water_vapor": water_vp_bar / pressure * 100,
        }

    def calculate_compression_work(
        self,
        stage: CompressionStage,
        flow_rate: float,
        mixture_props: dict[str, float],
    ) -> dict[str, Any]:
        """Calculate compression work for different compression types."""
        if stage.inlet_pressure <= 0:
            raise ValueError(f"inlet_pressure must be > 0, got {stage.inlet_pressure}")
        if stage.outlet_pressure <= 0:
            raise ValueError(
                f"outlet_pressure must be > 0, got {stage.outlet_pressure}"
            )

        gamma = mixture_props["heat_capacity_ratio"]
        if gamma <= 0:
            raise ValueError(f"heat_capacity_ratio (gamma) must be > 0, got {gamma}")
        if gamma == 1.0:
            raise ValueError(
                "heat_capacity_ratio (gamma) must not be 1.0; "
                "gamma/(gamma-1) would cause division by zero"
            )

        pr = stage.outlet_pressure / stage.inlet_pressure

        work_isentropic = None
        temp_out_isentropic = None

        if stage.compression_type == "isentropic":
            temp_out_isentropic = stage.inlet_temperature * (
                pr ** ((gamma - 1) / gamma)
            )
            work_isentropic = (
                (gamma / (gamma - 1))
                * self.R
                * stage.inlet_temperature
                * (pr ** ((gamma - 1) / gamma) - 1)
            )
            work_actual = work_isentropic / stage.efficiency
            temp_out_actual = stage.inlet_temperature + (
                work_actual / (self.R * gamma / (gamma - 1))
            )

        elif stage.compression_type == "polytropic":
            n = gamma
            temp_out_actual = stage.inlet_temperature * (pr ** ((n - 1) / n))
            work_actual = (
                (n / (n - 1))
                * self.R
                * stage.inlet_temperature
                * (pr ** ((n - 1) / n) - 1)
                / stage.efficiency
            )

        elif stage.compression_type == "isothermal":
            work_actual = (
                self.R * stage.inlet_temperature * math.log(pr) / stage.efficiency
            )
            temp_out_actual = stage.inlet_temperature

        else:
            msg = f"Unknown compression type: {stage.compression_type}"
            raise ValueError(msg)

        power_hp = (flow_rate * 1000 / SECONDS_PER_HOUR) * work_actual / WATTS_PER_HP
        heat_rise = temp_out_actual - stage.inlet_temperature

        return {
            "work_isentropic": (
                work_isentropic if stage.compression_type == "isentropic" else None
            ),
            "work_actual": work_actual,
            "temp_out_isentropic": (
                temp_out_isentropic if stage.compression_type == "isentropic" else None
            ),
            "temp_out_actual": temp_out_actual,
            "power_hp": power_hp,
            "heat_rise": heat_rise,
            "pressure_ratio": pr,
        }

    def calculate_multistage_compression(
        self,
        stages: list[CompressionStage],
        flow_rate: float,
        composition: dict[str, float],
        intercooling: bool = True,
    ) -> dict[str, Any]:
        """Calculate multistage compression with optional intercooling."""
        if not stages:
            raise ValueError("stages list must not be empty")

        mixture_props = self.calculate_mixture_properties(composition)
        results = []
        total_power = 0
        current_temp = stages[0].inlet_temperature

        for i, stage in enumerate(stages):
            if i > 0 and intercooling:
                stage.inlet_temperature = INTERCOOLER_OUTLET_TEMP_K
            elif i > 0:
                stage.inlet_temperature = current_temp

            stage_result = self.calculate_compression_work(
                stage,
                flow_rate,
                mixture_props,
            )
            stage_result["stage_number"] = i + 1
            stage_result["inlet_temp"] = stage.inlet_temperature
            stage_result["outlet_temp"] = stage_result["temp_out_actual"]

            water_dropout = self.calculate_water_dropout(
                stage_result["outlet_temp"],
                stage.outlet_pressure,
                composition.get("H2O", 0),
            )
            stage_result["water_dropout"] = water_dropout

            results.append(stage_result)
            total_power += stage_result["power_hp"]
            current_temp = stage_result["outlet_temp"]

        return {
            "stages": results,
            "total_power_hp": total_power,
            "final_temperature": current_temp,
            "final_pressure": stages[-1].outlet_pressure,
            "mixture_properties": mixture_props,
        }

    def analyze_process_conditions(
        self,
        compression_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze process conditions and potential concerns."""
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
            recommendations.append(
                "Consider additional intercooling or heat exchangers",
            )

        if final_temp > COMPRESSION_TEMP_CRITICAL_K:
            warnings.append("CRITICAL: Temperature exceeds safe operating limits")

        if final_pressure > COMPRESSION_HIGH_PRESSURE_BAR:
            concerns.append(
                "High pressure requires special equipment and safety measures",
            )
            recommendations.append(
                "Verify equipment pressure ratings and safety systems",
            )

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
            avg_efficiency = sum(efficiencies) / len(efficiencies)
            if avg_efficiency < COMPRESSION_MIN_EFFICIENCY:
                concerns.append("Low compression efficiency detected")
                recommendations.append("Consider compressor maintenance or replacement")
        else:
            avg_efficiency = None

        return {
            "concerns": concerns,
            "warnings": warnings,
            "recommendations": recommendations,
            "total_water_dropout": total_water_dropout,
            "average_efficiency": avg_efficiency,
        }
