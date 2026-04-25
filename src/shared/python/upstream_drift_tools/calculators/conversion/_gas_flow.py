from __future__ import annotations

from .core import (
    actual_to_standard_flow,
    scfm_to_standard_m3_per_hour,
    standard_m3_per_hour_to_scfm,
    standard_to_actual_flow,
)
from .tables import GAS_DATABASE, StandardCondition


class GasFlowMixin:
    def _convert_gas_flow(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        temperature: float | None = None,
        pressure: float | None = None,
        gas_type: str = "air",
        standard_condition: StandardCondition = StandardCondition.SCFM_60F,
    ) -> float:
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        gas_props = GAS_DATABASE.get(gas_type.lower(), GAS_DATABASE["air"])
        self._ensure_acfm_inputs(from_unit, to_unit, temperature, pressure)
        m3_hr_std = self._gas_flow_to_standard_m3h(
            value,
            from_unit,
            gas_props.density_stp,
            temperature,
            pressure,
            standard_condition,
        )
        return self._standard_m3h_to_gas_flow(
            m3_hr_std,
            to_unit,
            gas_props.density_stp,
            temperature,
            pressure,
            standard_condition,
        )

    def _ensure_acfm_inputs(
        self,
        from_unit: str,
        to_unit: str,
        temperature: float | None,
        pressure: float | None,
    ) -> None:
        if (from_unit == "ACFM" or to_unit == "ACFM") and (
            temperature is None or pressure is None
        ):
            msg = "Temperature and pressure are required for ACFM conversions"
            raise ValueError(msg)

    def _gas_flow_to_standard_m3h(
        self,
        value: float,
        from_unit: str,
        density_stp: float,
        temperature: float | None,
        pressure: float | None,
        standard_condition: StandardCondition,
    ) -> float:
        from ._errors import UnknownUnitError

        if from_unit == "SCFM":
            return scfm_to_standard_m3_per_hour(
                value, standard_condition, StandardCondition.STP
            )
        if from_unit == "ACFM":
            if not (temperature is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            if not (pressure is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            scfm = actual_to_standard_flow(
                value, temperature, pressure, standard_condition
            )
            return scfm_to_standard_m3_per_hour(
                scfm, standard_condition, StandardCondition.STP
            )
        if from_unit in {"Nm3/hr", "Nm³/hr"}:
            return value
        if from_unit in self.mass_flow_factors:  # type: ignore[attr-defined]
            kg_s = value * self.mass_flow_factors[from_unit]  # type: ignore[attr-defined]
            return (kg_s * 3600.0) / density_stp
        msg = f"Unknown gas flow unit: {from_unit}"
        raise UnknownUnitError(msg)

    def _standard_m3h_to_gas_flow(
        self,
        m3_hr_std: float,
        to_unit: str,
        density_stp: float,
        temperature: float | None,
        pressure: float | None,
        standard_condition: StandardCondition,
    ) -> float:
        from ._errors import UnknownUnitError

        if to_unit == "SCFM":
            return standard_m3_per_hour_to_scfm(
                m3_hr_std, StandardCondition.STP, standard_condition
            )
        if to_unit == "ACFM":
            if not (temperature is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            if not (pressure is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            scfm = standard_m3_per_hour_to_scfm(
                m3_hr_std, StandardCondition.STP, standard_condition
            )
            return standard_to_actual_flow(
                scfm, temperature, pressure, standard_condition
            )
        if to_unit in {"Nm3/hr", "Nm³/hr"}:
            return m3_hr_std
        if to_unit in self.mass_flow_factors:  # type: ignore[attr-defined]
            kg_s = (m3_hr_std * density_stp) / 3600.0
            return kg_s / self.mass_flow_factors[to_unit]  # type: ignore[attr-defined]
        msg = f"Unknown gas flow unit: {to_unit}"
        raise UnknownUnitError(msg)

    def convert_gas_flow_scfm_acfm(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        gas_type: str = "air",
        actual_temp_K: float | None = None,
        actual_pressure_kPa: float | None = None,
        standard_condition: StandardCondition = StandardCondition.SCFM_60F,
        compressibility_factor: float = 1.0,
    ) -> float:
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        std_temp, std_pressure_pa, _ = standard_condition.value
        temperature = actual_temp_K or std_temp
        pressure_pa = (
            actual_pressure_kPa * 1000.0
            if actual_pressure_kPa is not None
            else std_pressure_pa
        )

        result = self._convert_gas_flow(
            value,
            from_unit.upper(),
            to_unit.upper(),
            temperature=temperature,
            pressure=pressure_pa,
            gas_type=gas_type,
            standard_condition=standard_condition,
        )

        if from_unit.upper() == "SCFM" and to_unit.upper() == "ACFM":
            return result * compressibility_factor
        if from_unit.upper() == "ACFM" and to_unit.upper() == "SCFM":
            if compressibility_factor <= 0:
                return result
            return result / compressibility_factor
        return result

    def compressibility_factor(
        self,
        gas_type: str,
        temperature: float,
        pressure: float,
    ) -> float:
        if not (gas_type is not None):
            raise ValueError("gas_type must be provided")
        if not (gas_type is not None):
            raise ValueError("gas_type must be provided")
        self._require_positive_finite(temperature, "temperature")  # type: ignore[attr-defined]
        self._require_positive_finite(pressure, "pressure")  # type: ignore[attr-defined]
        gas_props = GAS_DATABASE.get(gas_type.lower(), GAS_DATABASE["air"])
        Tr = temperature / gas_props.critical_temp
        Pr = pressure / gas_props.critical_pressure

        if 0.7 < Tr < 4 and Pr < 10:
            Z = 1 + (0.083 - 0.422 / Tr**1.6) * Pr + (0.139 - 0.172 / Tr**4.2) * Pr**2
            return float(max(Z, 0.1))
        return 1.0
