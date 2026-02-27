"""Baghouse Calculator
===================

Core calculation engine for baghouse filter performance, solid removal, and drum sizing.

This is a standalone calculator that can work with or without the full thermodynamic
engine. When the thermo module is not available, it uses simplified ideal gas calculations.
"""

from dataclasses import dataclass
from typing import Any

from .constants import (
    ATM_PA,
    CELSIUS_TO_KELVIN_OFFSET,
    CP_AR_500K,
    CP_CH4_500K,
    CP_CO2_500K,
    CP_CO_500K,
    CP_DEFAULT_FALLBACK,
    CP_H2_500K,
    CP_H2O_500K,
    CP_MASS_DEFAULT_FALLBACK,
    CP_N2_500K,
    CP_O2_500K,
    HOURS_PER_DAY,
    M3_S_TO_CFM,
    MW_AR_KG,
    MW_CH4_KG,
    MW_CO2_KG,
    MW_CO_KG,
    MW_DEFAULT_KG,
    MW_H2_KG,
    MW_H2O_KG,
    MW_N2_KG,
    MW_O2_KG,
    R_UNIVERSAL,
    SECONDS_PER_HOUR,
    STP_TEMPERATURE_K,
)

# Try to import thermo module, use simplified calculations if not available
try:
    from tools.thermo import (
        FlowUnit,
        GasStream,
        ThermodynamicCalculator,
    )
    from tools.unit_converter import convert

    HAS_THERMO = True
except ImportError:
    HAS_THERMO = False
    FlowUnit = None
    GasStream = None
    ThermodynamicCalculator = None

    def convert(value: float, from_unit: str, to_unit: str) -> float:
        """Simple temperature conversion fallback."""
        if from_unit == "K" and to_unit == "C":
            return float(value - CELSIUS_TO_KELVIN_OFFSET)
        if from_unit == "C" and to_unit == "K":
            return float(value + CELSIUS_TO_KELVIN_OFFSET)
        return value


@dataclass
class BaghouseResult:
    """Result data for baghouse calculation."""

    carbon_removed_rate: float  # kg/hr
    ash_removed_rate: float  # kg/hr
    total_solids_removed_rate: float  # kg/hr

    drum_fill_time_hours: float
    drum_fill_time_days: float
    carbon_only_fill_time_hours: float
    ash_only_fill_time_hours: float

    clean_gas_flow_rate: float  # kg/hr
    flow_acfm: float
    flow_scfm: float
    air_to_cloth_ratio: float  # ft/min

    outlet_temperature_c: float
    ash_stream_composition: dict[str, float]

    removal_efficiency: dict[str, float]


class BaghouseCalculator:
    """Core baghouse calculation engine.

    This calculator determines baghouse filter performance including solid removal
    rates, drum fill times, and air-to-cloth ratios. It can operate in two modes:

    1. Full mode (with thermo module): Uses detailed thermodynamic calculations
    2. Simplified mode (standalone): Uses ideal gas approximations
    """

    def __init__(self, thermo_calc: Any | None = None) -> None:
        """Initialize the baghouse calculator.

        Args:
            thermo_calc: Optional thermodynamic calculator instance. If not provided
                        and thermo module is available, one will be created.
        """
        if thermo_calc is not None:
            self.thermo_calc = thermo_calc
        elif HAS_THERMO and ThermodynamicCalculator is not None:
            self.thermo_calc = ThermodynamicCalculator()
        else:
            self.thermo_calc = None

    def _estimate_cp_ideal(self, composition: dict[str, float]) -> float:
        """Estimate Cp for gas mixture using ideal gas approximation.

        Uses constant Cp values for common syngas species.

        Args:
            composition: Mole fraction composition

        Returns:
            Estimated Cp in J/(kg·K)
        """
        # Approximate Cp values at ~500K in J/(mol·K)
        cp_data = {
            "H2": CP_H2_500K,
            "CO": CP_CO_500K,
            "CO2": CP_CO2_500K,
            "H2O": CP_H2O_500K,
            "N2": CP_N2_500K,
            "CH4": CP_CH4_500K,
            "O2": CP_O2_500K,
            "Ar": CP_AR_500K,
        }
        # Molecular weights in kg/mol
        mw_data = {
            "H2": MW_H2_KG,
            "CO": MW_CO_KG,
            "CO2": MW_CO2_KG,
            "H2O": MW_H2O_KG,
            "N2": MW_N2_KG,
            "CH4": MW_CH4_KG,
            "O2": MW_O2_KG,
            "Ar": MW_AR_KG,
        }

        cp_mol = 0.0
        mw_avg = 0.0
        for species, frac in composition.items():
            cp_mol += frac * cp_data.get(species, CP_DEFAULT_FALLBACK)
            mw_avg += frac * mw_data.get(species, MW_DEFAULT_KG)

        if mw_avg > 0:
            return cp_mol / mw_avg  # J/(kg·K)
        return CP_MASS_DEFAULT_FALLBACK  # Default fallback

    def _estimate_volume_flow(
        self,
        mass_flow_kg_s: float,
        temperature_k: float,
        pressure_pa: float,
        composition: dict[str, float],
    ) -> tuple[float, float]:
        """Estimate volumetric flow rates using ideal gas law.

        Returns:
            (acfm, scfm) - Actual and standard cubic feet per minute
        """
        # Molecular weights in kg/mol
        mw_data = {
            "H2": MW_H2_KG,
            "CO": MW_CO_KG,
            "CO2": MW_CO2_KG,
            "H2O": MW_H2O_KG,
            "N2": MW_N2_KG,
            "CH4": MW_CH4_KG,
            "O2": MW_O2_KG,
            "Ar": MW_AR_KG,
        }

        mw_avg = sum(
            frac * mw_data.get(species, MW_DEFAULT_KG)
            for species, frac in composition.items()
        )

        if mw_avg > 0:
            molar_flow = mass_flow_kg_s / mw_avg  # mol/s
        else:
            molar_flow = mass_flow_kg_s / MW_N2_KG  # Assume N2-like

        # Actual volume flow (ideal gas)
        vol_actual_m3_s = molar_flow * R_UNIVERSAL * temperature_k / pressure_pa

        # Standard volume flow (at STP: 273.15 K, 101325 Pa)
        vol_std_m3_s = molar_flow * R_UNIVERSAL * STP_TEMPERATURE_K / ATM_PA

        # Convert to cfm (1 m³/s = 2118.88 cfm)
        acfm = vol_actual_m3_s * M3_S_TO_CFM
        scfm = vol_std_m3_s * M3_S_TO_CFM

        return acfm, scfm

    def _calculate_outlet_thermal(
        self,
        gas_flow_kg_s: float,
        inlet_temp_k: float,
        pressure_pa: float,
        composition: dict[str, float],
        heat_loss_w: float,
    ) -> tuple[float, float, float]:
        """Compute outlet temperature and volume flows.

        Tries full thermodynamic calculation first, falls back to simplified
        ideal-gas approach if the thermo module is unavailable or errors.

        Returns:
            (outlet_temp_c, flow_acfm, flow_scfm)
        """
        if self.thermo_calc is not None and HAS_THERMO:
            try:
                stream = GasStream(
                    flow_rate=gas_flow_kg_s,
                    flow_unit=FlowUnit.MASS,
                    temperature=inlet_temp_k,
                    pressure=pressure_pa,
                    composition=composition,
                )
                props = self.thermo_calc.calculate_stream_properties(stream)
                cp_mass = props.cp

                temp_drop = (
                    heat_loss_w / (gas_flow_kg_s * cp_mass)
                    if gas_flow_kg_s > 0 and cp_mass > 0
                    else 0.0
                )
                outlet_temp_k = max(inlet_temp_k - temp_drop, 0.0)

                outlet_stream = GasStream(
                    flow_rate=gas_flow_kg_s,
                    flow_unit=FlowUnit.MASS,
                    temperature=outlet_temp_k,
                    pressure=pressure_pa,
                    composition=composition,
                )
                outlet_props = self.thermo_calc.calculate_stream_properties(
                    outlet_stream
                )
                return (
                    convert(outlet_temp_k, "K", "C"),
                    outlet_props.acfm_flow,
                    outlet_props.scfm_flow,
                )
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                self.thermo_calc = None

        # Simplified path
        cp_mass = self._estimate_cp_ideal(composition)
        temp_drop = (
            heat_loss_w / (gas_flow_kg_s * cp_mass)
            if gas_flow_kg_s > 0 and cp_mass > 0
            else 0.0
        )
        outlet_temp_k = max(inlet_temp_k - temp_drop, 0.0)
        flow_acfm, flow_scfm = self._estimate_volume_flow(
            gas_flow_kg_s, outlet_temp_k, pressure_pa, composition
        )
        return outlet_temp_k - CELSIUS_TO_KELVIN_OFFSET, flow_acfm, flow_scfm

    @staticmethod
    def _calculate_drum_sizing(
        solid_carbon_in_kg_hr: float,
        ash_in_kg_hr: float,
        carbon_removal_efficiency: float,
        ash_removal_efficiency: float,
        drum_volume_m3: float,
        solid_density_kg_m3: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Compute solids removal rates and drum fill times.

        Returns:
            (carbon_removed, ash_removed, total_solids,
             fill_time_hours, fill_time_days,
             carbon_only_fill_hours, ash_only_fill_hours)
        """
        carbon_removed = solid_carbon_in_kg_hr * carbon_removal_efficiency
        ash_removed = ash_in_kg_hr * ash_removal_efficiency
        total_solids = carbon_removed + ash_removed
        drum_cap = solid_density_kg_m3 * drum_volume_m3

        fill_hrs = drum_cap / total_solids if total_solids > 0 else float("inf")
        fill_days = (
            fill_hrs / HOURS_PER_DAY if fill_hrs != float("inf") else float("inf")
        )
        c_fill = drum_cap / carbon_removed if carbon_removed > 0 else float("inf")
        a_fill = drum_cap / ash_removed if ash_removed > 0 else float("inf")

        return (
            carbon_removed,
            ash_removed,
            total_solids,
            fill_hrs,
            fill_days,
            c_fill,
            a_fill,
        )

    def calculate(
        self,
        gas_flow_kg_s: float,
        inlet_temp_k: float,
        pressure_pa: float,
        composition: dict[str, float],
        solid_carbon_in_kg_hr: float,
        ash_in_kg_hr: float,
        carbon_removal_efficiency: float,  # 0-1
        ash_removal_efficiency: float,  # 0-1
        heat_loss_w: float,
        drum_volume_m3: float,
        solid_density_kg_m3: float,
        bag_area_ft2: float,
    ) -> BaghouseResult:
        """Calculate baghouse performance.

        Args:
            gas_flow_kg_s: Gas mass flow rate [kg/s]
            inlet_temp_k: Inlet temperature [K]
            pressure_pa: Pressure [Pa]
            composition: Gas composition dictionary
            solid_carbon_in_kg_hr: Solid carbon input rate [kg/hr]
            ash_in_kg_hr: Ash input rate [kg/hr]
            carbon_removal_efficiency: Carbon removal efficiency (0-1)
            ash_removal_efficiency: Ash removal efficiency (0-1)
            heat_loss_w: Heat loss rate [W]
            drum_volume_m3: Collection drum volume [m³]
            solid_density_kg_m3: Density of collected solids [kg/m³]
            bag_area_ft2: Total bag filter area [ft²]

        Returns:
            BaghouseResult object
        """
        # DbC preconditions on physical quantities
        assert gas_flow_kg_s > 0, f"Gas flow must be positive, got {gas_flow_kg_s}"
        assert inlet_temp_k > 0, f"Temperature must be positive (K), got {inlet_temp_k}"
        assert pressure_pa > 0, f"Pressure must be positive, got {pressure_pa}"
        assert (
            0 <= carbon_removal_efficiency <= 1
        ), f"Carbon removal efficiency must be 0-1, got {carbon_removal_efficiency}"
        assert (
            0 <= ash_removal_efficiency <= 1
        ), f"Ash removal efficiency must be 0-1, got {ash_removal_efficiency}"
        assert drum_volume_m3 > 0, f"Drum volume must be positive, got {drum_volume_m3}"
        assert (
            solid_density_kg_m3 > 0
        ), f"Solid density must be positive, got {solid_density_kg_m3}"

        outlet_temp_c, flow_acfm, flow_scfm = self._calculate_outlet_thermal(
            gas_flow_kg_s,
            inlet_temp_k,
            pressure_pa,
            composition,
            heat_loss_w,
        )

        (
            carbon_removed,
            ash_removed,
            total_solids,
            fill_hrs,
            fill_days,
            c_fill,
            a_fill,
        ) = self._calculate_drum_sizing(
            solid_carbon_in_kg_hr,
            ash_in_kg_hr,
            carbon_removal_efficiency,
            ash_removal_efficiency,
            drum_volume_m3,
            solid_density_kg_m3,
        )

        air_to_cloth = flow_acfm / bag_area_ft2 if bag_area_ft2 > 0 else 0.0

        ash_stream_comp = {
            "carbon_fraction": (
                carbon_removed / total_solids if total_solids > 0 else 0.0
            ),
            "ash_fraction": (ash_removed / total_solids if total_solids > 0 else 0.0),
        }

        return BaghouseResult(
            carbon_removed_rate=carbon_removed,
            ash_removed_rate=ash_removed,
            total_solids_removed_rate=total_solids,
            drum_fill_time_hours=fill_hrs,
            drum_fill_time_days=fill_days,
            carbon_only_fill_time_hours=c_fill,
            ash_only_fill_time_hours=a_fill,
            clean_gas_flow_rate=gas_flow_kg_s * SECONDS_PER_HOUR,
            flow_acfm=flow_acfm,
            flow_scfm=flow_scfm,
            air_to_cloth_ratio=air_to_cloth,
            outlet_temperature_c=outlet_temp_c,
            ash_stream_composition=ash_stream_comp,
            removal_efficiency={
                "carbon": carbon_removal_efficiency * 100.0,
                "ash": ash_removal_efficiency * 100.0,
            },
        )
