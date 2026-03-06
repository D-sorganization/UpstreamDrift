"""Scrubber Calculation Engine."""

import logging
from typing import Any

from ...scrubber_calculator import (
    PACKING_DATABASE,
    calculate_caustic_requirement,
    calculate_column_diameter,
    calculate_cooling_water_requirement,
    calculate_flooding_velocity,
    calculate_gas_density,
    calculate_heat_transfer_duty,
    calculate_htu,
    calculate_ntu_removal,
    calculate_pressure_drop,
    calculate_required_packed_height,
)
from ..models.scrubber_models import ScrubberInputs, ScrubberResults

logger = logging.getLogger(__name__)


class ScrubberEngine:
    """Engine for packed bed scrubber design calculations."""

    @staticmethod
    def _calculate_column_sizing(
        inputs: ScrubberInputs,
        gas_density: float,
    ) -> tuple[float, float, float, float, float, list[str]]:
        """Column diameter and velocity sizing.

        Returns (diameter_m, actual_area, gas_mass_flux,
                 liquid_mass_flux, design_velocity, warnings).
        """
        warnings: list[str] = []
        liquid_flow_kg_hr = inputs.gas_flow_kg_hr * inputs.lg_ratio
        liquid_density = 1000.0 + 10.8 * inputs.caustic_concentration_wt_pct

        packing = PACKING_DATABASE.get(inputs.packing_name)
        if not packing:
            raise ValueError(f"Unknown packing type: {inputs.packing_name}")

        # Initial liquid flux estimate (assumed 2 m² area)
        estimated_area = 2.0
        liquid_mass_flux = (liquid_flow_kg_hr / 3600.0) / estimated_area

        flooding_velocity = calculate_flooding_velocity(
            liquid_mass_flux=liquid_mass_flux,
            gas_density=gas_density,
            liquid_density=liquid_density,
            packing=packing,
        )

        column_sizing = calculate_column_diameter(
            gas_flow_kg_hr=inputs.gas_flow_kg_hr,
            gas_density=gas_density,
            flooding_velocity=flooding_velocity,
            percent_of_flood=inputs.percent_of_flood,
        )
        if "warning" in column_sizing:
            warnings.append(str(column_sizing["warning"]))

        actual_area = float(column_sizing.get("cross_section_m2", 0.0))
        diameter_m = float(column_sizing.get("diameter_m", 0.0))
        gas_flow_kg_s = inputs.gas_flow_kg_hr / 3600.0

        if actual_area > 0:
            liquid_mass_flux = (liquid_flow_kg_hr / 3600.0) / actual_area
            gas_mass_flux = gas_flow_kg_s / actual_area
        else:
            liquid_mass_flux = 0.0
            gas_mass_flux = 0.0

        design_velocity = float(column_sizing.get("design_velocity_m_s", 0.0))
        return (
            diameter_m,
            actual_area,
            gas_mass_flux,
            liquid_mass_flux,
            design_velocity,
            warnings,
        )

    @staticmethod
    def _calculate_mass_transfer(
        inputs: ScrubberInputs,
        gas_density: float,
        gas_mass_flux: float,
        liquid_mass_flux: float,
    ) -> tuple[float, float, list[dict[str, Any]], dict[str, float]]:
        """NTU/HTU mass-transfer calculations.

        Returns (packed_height, max_ntu, acid_gas_details, acid_gas_removed).
        """
        mw_gases = {"HCl": 36.458, "SO2": 64.06, "H2S": 34.08, "HF": 20.01}
        acid_gas_details: list[dict[str, Any]] = []
        acid_gas_removed: dict[str, float] = {}
        max_ntu = 0.0

        for gas_name, inlet_ppmv in inputs.acid_gas_composition_ppmv.items():
            removal_pct = inputs.acid_gas_removal_pct.get(gas_name, 0.0)
            if inlet_ppmv > 0 and removal_pct > 0:
                inlet_frac = inlet_ppmv / 1e6
                outlet_ppmv = inlet_ppmv * (1 - removal_pct / 100.0)
                outlet_frac = outlet_ppmv / 1e6

                ntu = calculate_ntu_removal(inlet_frac, outlet_frac)
                max_ntu = max(max_ntu, ntu)

                gas_molar_flow = inputs.gas_flow_kg_hr / inputs.molecular_weight
                removed_kmol_hr = gas_molar_flow * (inlet_frac - outlet_frac)
                mw_gas = mw_gases.get(gas_name, 30.0)
                removed_kg_hr = removed_kmol_hr * mw_gas

                acid_gas_details.append(
                    {
                        "name": gas_name,
                        "inlet_ppmv": inlet_ppmv,
                        "outlet_ppmv": outlet_ppmv,
                        "removed_kg_hr": removed_kg_hr,
                        "ntu": ntu,
                    }
                )
                acid_gas_removed[gas_name] = removed_kg_hr

        packing = PACKING_DATABASE[inputs.packing_name]
        htu = calculate_htu(
            gas_mass_flux=gas_mass_flux,
            liquid_mass_flux=liquid_mass_flux,
            gas_density=gas_density,
            packing=packing,
            kla=inputs.kla_hr,
        )
        packed_height = calculate_required_packed_height(
            ntu=max_ntu, htu=htu, safety_factor=inputs.height_safety_factor
        )
        return packed_height, max_ntu, acid_gas_details, acid_gas_removed

    @staticmethod
    def _calculate_thermal(
        inputs: ScrubberInputs,
        acid_gas_removed: dict[str, float],
    ) -> tuple[float, float, float, float, list[str]]:
        """Caustic requirement, heat duty, and cooling water.

        Returns (naoh_pure, naoh_solution, heat_kw, cooling_L_min, warnings).
        """
        warnings: list[str] = []
        caustic_req = calculate_caustic_requirement(
            acid_gas_removed=acid_gas_removed,
            caustic_concentration=inputs.caustic_concentration_wt_pct,
        )

        water_condensed = (
            inputs.gas_flow_kg_hr
            * 0.0015
            * (inputs.inlet_temp_c - inputs.target_outlet_temp_c)
        )
        heat_duty = calculate_heat_transfer_duty(
            gas_flow_kg_hr=inputs.gas_flow_kg_hr,
            inlet_temp_c=inputs.inlet_temp_c,
            outlet_temp_c=inputs.target_outlet_temp_c,
            water_condensed_kg_hr=water_condensed,
        )

        cooling_water = calculate_cooling_water_requirement(
            heat_duty_kw=heat_duty["total_heat_kw"],
            water_inlet_temp_c=inputs.cooling_water_inlet_temp_c,
            outlet_gas_temp_c=inputs.target_outlet_temp_c,
        )
        if "warning" in cooling_water:
            warnings.append(str(cooling_water["warning"]))

        return (  # type: ignore
            caustic_req.get("naoh_pure_kg_hr", 0.0),
            caustic_req.get("naoh_solution_L_hr", 0.0),
            heat_duty["total_heat_kw"],
            cooling_water.get("water_flow_L_min", 0.0),
            warnings,
        )

    @staticmethod
    def calculate(inputs: ScrubberInputs) -> ScrubberResults:
        """Perform full scrubber design calculation."""
        if inputs.gas_flow_kg_hr <= 0:
            return ScrubberResults(
                column_diameter_m=0.0,
                packed_height_m=0.0,
                pressure_drop_kpa=0.0,
                naoh_pure_kg_hr=0.0,
                naoh_solution_L_hr=0.0,
                total_heat_duty_kw=0.0,
                cooling_water_flow_L_min=0.0,
                gas_density_kg_m3=0.0,
                flooding_velocity_m_s=0.0,
                htu_m=0.0,
                max_ntu=0.0,
                warnings=["Gas flow is zero or negative."],
            )

        # 1. Physical properties
        temp_k = inputs.inlet_temp_c + 273.15
        pressure_pa = inputs.pressure_bar * 1e5
        gas_density = calculate_gas_density(
            temp_k, pressure_pa, inputs.molecular_weight
        )

        # 2. Column sizing
        (
            diameter_m,
            _actual_area,
            gas_mass_flux,
            liquid_mass_flux,
            design_velocity,
            col_warnings,
        ) = ScrubberEngine._calculate_column_sizing(inputs, gas_density)

        # 3. Mass transfer (NTU / HTU)
        packed_height, max_ntu, acid_gas_details, acid_gas_removed = (
            ScrubberEngine._calculate_mass_transfer(
                inputs,
                gas_density,
                gas_mass_flux,
                liquid_mass_flux,
            )
        )

        # 4. Pressure drop
        liquid_density = 1000.0 + 10.8 * inputs.caustic_concentration_wt_pct
        packing = PACKING_DATABASE[inputs.packing_name]
        pressure_drop_pa = 0.0
        if design_velocity > 0 and packed_height > 0:
            pressure_drop_pa = calculate_pressure_drop(
                gas_velocity=design_velocity,
                gas_density=gas_density,
                liquid_mass_flux=liquid_mass_flux,
                liquid_density=liquid_density,
                packing=packing,
                packed_height=packed_height,
            )

        # 5. Thermal & caustic
        naoh_pure, naoh_sol, heat_kw, cw_flow, therm_warnings = (
            ScrubberEngine._calculate_thermal(inputs, acid_gas_removed)
        )

        packing_obj = PACKING_DATABASE.get(inputs.packing_name)
        flooding_vel = 0.0
        if packing_obj:
            flooding_vel = calculate_flooding_velocity(
                liquid_mass_flux=liquid_mass_flux,
                gas_density=gas_density,
                liquid_density=liquid_density,
                packing=packing_obj,
            )

        # 6. HTU (for result)
        htu = calculate_htu(
            gas_mass_flux=gas_mass_flux,
            liquid_mass_flux=liquid_mass_flux,
            gas_density=gas_density,
            packing=packing,
            kla=inputs.kla_hr,
        )

        return ScrubberResults(
            column_diameter_m=diameter_m,
            packed_height_m=packed_height,
            pressure_drop_kpa=pressure_drop_pa / 1000.0,
            naoh_pure_kg_hr=naoh_pure,
            naoh_solution_L_hr=naoh_sol,
            total_heat_duty_kw=heat_kw,
            cooling_water_flow_L_min=cw_flow,
            gas_density_kg_m3=gas_density,
            flooding_velocity_m_s=flooding_vel,
            htu_m=htu,
            max_ntu=max_ntu,
            acid_gas_details=acid_gas_details,
            warnings=col_warnings + therm_warnings,
        )
