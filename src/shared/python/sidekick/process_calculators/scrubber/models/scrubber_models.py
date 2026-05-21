"""Data models for Scrubber Calculator."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScrubberInputs:
    """Inputs for the Scrubber Calculator."""

    gas_flow_kg_hr: float
    inlet_temp_c: float
    pressure_bar: float
    molecular_weight: float
    target_outlet_temp_c: float
    packing_name: str
    percent_of_flood: float
    height_safety_factor: float
    lg_ratio: float
    caustic_concentration_wt_pct: float
    cooling_water_inlet_temp_c: float
    kla_hr: float
    acid_gas_composition_ppmv: dict[str, float] = field(default_factory=dict)
    acid_gas_removal_pct: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScrubberResults:
    """Results from the Scrubber Calculator."""

    column_diameter_m: float
    packed_height_m: float
    pressure_drop_kpa: float
    naoh_pure_kg_hr: float
    naoh_solution_L_hr: float
    total_heat_duty_kw: float
    cooling_water_flow_L_min: float
    gas_density_kg_m3: float
    flooding_velocity_m_s: float
    htu_m: float
    max_ntu: float
    acid_gas_details: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
