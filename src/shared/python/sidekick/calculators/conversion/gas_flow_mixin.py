"""Gas flow conversion mixin for UnitConversionService.

Extracts SCFM/ACFM/Nm3 gas flow conversion logic from the main service class
to improve single-responsibility adherence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .core import (
    actual_to_standard_flow,
    scfm_to_standard_m3_per_hour,
    standard_m3_per_hour_to_scfm,
    standard_to_actual_flow,
)
from .tables import (
    GAS_DATABASE,
    NORMAL_REFERENCE_CONDITION,
    GasProperties,
    StandardCondition,
)

__all__ = [
    "GasFlowConversionMixin",
]

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


class GasFlowConversionMixin:
    """Gas flow (SCFM/ACFM/Nm³/hr) conversion mixin for UnitConversionService."""

    mass_flow_factors: dict[str, float]  # provided by UnitConversionService

    @staticmethod
    def _resolve_gas_type(gas_type: str | None) -> GasProperties:
        """Return gas properties for a supported gas or fail loudly."""
        if gas_type is None:
            raise ValueError("gas_type must be provided")
        key = gas_type.lower()
        try:
            return GAS_DATABASE[key]
        except KeyError as exc:
            supported = ", ".join(sorted(GAS_DATABASE))
            msg = f"Unknown gas type: {gas_type}. Supported gas types: {supported}"
            raise ValueError(msg) from exc

    @staticmethod
    def _gas_acentric_factor(gas_type: str) -> float:
        """Return available acentric factor data for the Pitzer correction."""
        acentric_factors = {
            "air": 0.04,
            "co": 0.049,
            "co2": 0.225,
            "h2o": 0.344,
            "hydrogen": -0.22,
            "methane": 0.011,
            "nitrogen": 0.037,
            "oxygen": 0.021,
        }
        return acentric_factors.get(gas_type.lower(), 0.0)

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
        """Convert gas flow rate.

        Raises:
            ValueError: If ``value`` is ``None``.
        """
        # Explicit ValueError guard (not bare ``assert``) so it survives
        # ``python -O`` (issue #3182 / #3344).
        if value is None:
            raise ValueError("value must be provided")
        gas_props = self._resolve_gas_type(gas_type)
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
        """Validate required inputs when ACFM is involved."""
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
        """Convert gas flow value to STP-normalized m³/hr."""
        from .service import UnknownUnitError

        # The "standard m³/hr" pivot here IS the Nm³ basis. Anchor it on the
        # single authoritative normal state (DIN 1343: 0 °C, 101.325 kPa) so
        # SCFM/ACFM ↔ Nm³ agrees with the tar-concentration mixin (issue #3389).
        if from_unit == "SCFM":
            return float(
                scfm_to_standard_m3_per_hour(
                    value, standard_condition, NORMAL_REFERENCE_CONDITION
                )
            )
        if from_unit == "ACFM":
            if temperature is None or pressure is None:
                msg = (
                    "temperature and pressure are required after ACFM input validation"
                )
                raise RuntimeError(msg)
            scfm = actual_to_standard_flow(
                value, temperature, pressure, standard_condition
            )
            return float(
                scfm_to_standard_m3_per_hour(
                    scfm, standard_condition, NORMAL_REFERENCE_CONDITION
                )
            )
        if from_unit in {"Nm3/hr", "Nm³/hr"}:
            return value
        if from_unit in self.mass_flow_factors:
            kg_s = value * self.mass_flow_factors[from_unit]
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
        """Convert STP-normalized m³/hr to destination gas flow unit."""
        from .service import UnknownUnitError

        if to_unit == "SCFM":
            return float(
                standard_m3_per_hour_to_scfm(
                    m3_hr_std, NORMAL_REFERENCE_CONDITION, standard_condition
                )
            )
        if to_unit == "ACFM":
            if temperature is None or pressure is None:
                msg = (
                    "temperature and pressure are required after ACFM input validation"
                )
                raise RuntimeError(msg)
            scfm = standard_m3_per_hour_to_scfm(
                m3_hr_std, NORMAL_REFERENCE_CONDITION, standard_condition
            )
            return float(
                standard_to_actual_flow(scfm, temperature, pressure, standard_condition)
            )
        if to_unit in {"Nm3/hr", "Nm³/hr"}:
            return m3_hr_std
        if to_unit in self.mass_flow_factors:
            kg_s = (m3_hr_std * density_stp) / 3600.0
            return kg_s / self.mass_flow_factors[to_unit]
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
        """Convert gas flow between SCFM and ACFM.

        Raises:
            ValueError: If ``value`` is ``None``; if ``compressibility_factor``
                is not positive and finite; or if an explicitly supplied
                ``actual_temp_K`` / ``actual_pressure_kPa`` is not positive and
                finite.
        """
        import math

        # Explicit ValueError guards (not bare ``assert``) so they survive
        # ``python -O`` (issue #3182 / #3344).
        if value is None:
            raise ValueError("value must be provided")
        if not math.isfinite(compressibility_factor) or compressibility_factor <= 0:
            msg = (
                "compressibility_factor must be positive and finite, "
                f"got {compressibility_factor}"
            )
            raise ValueError(msg)
        # An explicitly supplied actual temperature/pressure must be physical.
        # ``None`` means "use the standard condition default"; ``0.0`` (or any
        # non-positive/non-finite value) is an invalid explicit input and must
        # be rejected rather than silently coerced to the default (#3342/#3367).
        if actual_temp_K is not None and (
            not math.isfinite(actual_temp_K) or actual_temp_K <= 0
        ):
            msg = f"actual_temp_K must be positive and finite, got {actual_temp_K}"
            raise ValueError(msg)
        if actual_pressure_kPa is not None and (
            not math.isfinite(actual_pressure_kPa) or actual_pressure_kPa <= 0
        ):
            msg = (
                "actual_pressure_kPa must be positive and finite, "
                f"got {actual_pressure_kPa}"
            )
            raise ValueError(msg)

        std_temp, std_pressure_pa, _ = standard_condition.value
        temperature = actual_temp_K if actual_temp_K is not None else std_temp
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
            return result / compressibility_factor
        return result

    def compressibility_factor(
        self,
        gas_type: str,
        temperature: float,
        pressure: float,
    ) -> float:
        """Calculate compressibility factor.

        Raises:
            ValueError: If ``gas_type`` is ``None``, if ``temperature`` is not
                positive and finite, or if ``pressure`` is not positive and finite.
        """
        import math

        # Explicit ValueError guard (not bare ``assert``) so it survives
        # ``python -O`` (issue #3182 / #3344).
        if gas_type is None:
            raise ValueError("gas_type must be provided")
        if not math.isfinite(temperature) or temperature <= 0:
            msg = f"temperature must be positive and finite, got {temperature}"
            raise ValueError(msg)
        if not math.isfinite(pressure) or pressure <= 0:
            msg = f"pressure must be positive and finite, got {pressure}"
            raise ValueError(msg)
        gas_props = self._resolve_gas_type(gas_type)
        Tr = temperature / gas_props.critical_temp
        Pr = pressure / gas_props.critical_pressure

        if 0.7 < Tr < 4 and Pr < 10:
            B0 = 0.083 - 0.422 / Tr**1.6
            B1 = 0.139 - 0.172 / Tr**4.2
            Z = 1.0 + (B0 + self._gas_acentric_factor(gas_type) * B1) * (Pr / Tr)
            return float(max(0.1, min(Z, 1.5)))
        return 1.0
