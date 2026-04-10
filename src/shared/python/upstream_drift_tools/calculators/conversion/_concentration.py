from __future__ import annotations


class ConcentrationMixin:
    def tar_concentration(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        temperature: float = 273.15,
        pressure: float = 101.325,
        molecular_weight: float | None = None,
    ) -> float:
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        self._validate_tar_inputs(temperature, pressure)
        from_key = from_unit.lower()
        to_key = to_unit.lower()
        if from_key == to_key:
            return value
        molecular_weight = self._resolve_molecular_weight(
            from_key, to_key, molecular_weight
        )
        self._ensure_known_concentration_unit(from_key, from_unit)
        self._ensure_known_concentration_unit(to_key, to_unit)
        mg_nm3_value = self._tar_to_mg_nm3(
            value, from_key, from_unit, temperature, pressure, molecular_weight
        )
        return self._tar_from_mg_nm3(
            mg_nm3_value, to_key, to_unit, temperature, pressure, molecular_weight
        )

    def _validate_tar_inputs(self, temperature: float, pressure: float) -> None:
        if pressure <= 0:
            msg = f"pressure must be positive, got {pressure}"
            raise ValueError(msg)
        if temperature <= 0:
            msg = f"temperature must be positive, got {temperature}"
            raise ValueError(msg)

    def _resolve_molecular_weight(
        self, from_key: str, to_key: str, molecular_weight: float | None
    ) -> float | None:
        requires_molecular_weight = from_key == "ppm_mass" or to_key == "ppm_mass"
        if not requires_molecular_weight:
            return molecular_weight
        if molecular_weight is None:
            msg = "Molecular weight required for ppm conversion"
            raise ValueError(msg)
        self._require_positive_finite(molecular_weight, "Molecular weight")  # type: ignore[attr-defined]
        return molecular_weight

    def _ensure_known_concentration_unit(self, unit_key: str, raw_unit: str) -> None:
        if unit_key not in self.concentration_conversions:  # type: ignore[attr-defined]
            msg = f"Unknown concentration unit: {raw_unit}"
            raise ValueError(msg)

    def _tar_to_mg_nm3(
        self,
        value: float,
        from_key: str,
        from_unit: str,
        temperature: float,
        pressure: float,
        molecular_weight: float | None,
    ) -> float:
        factor = self.concentration_conversions[from_key]  # type: ignore[attr-defined]
        if factor is not None:
            return value * factor
        if from_key in {"mg/m³", "mg/m3"}:
            return value * (temperature / 273.15) * (101.325 / pressure)
        if from_key in {"g/m³", "g/m3"}:
            return value * 1000.0 * (temperature / 273.15) * (101.325 / pressure)
        if from_key == "ppm_mass":
            if not (molecular_weight is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            return value * molecular_weight / 24.45
        msg = f"Conversion from {from_unit} not implemented"
        raise ValueError(msg)

    def _tar_from_mg_nm3(
        self,
        mg_nm3_value: float,
        to_key: str,
        to_unit: str,
        temperature: float,
        pressure: float,
        molecular_weight: float | None,
    ) -> float:
        factor = self.concentration_conversions[to_key]  # type: ignore[attr-defined]
        if factor is not None:
            return mg_nm3_value / factor
        if to_key in {"mg/m³", "mg/m3"}:
            return mg_nm3_value * (273.15 / temperature) * (pressure / 101.325)
        if to_key in {"g/m³", "g/m3"}:
            return mg_nm3_value / 1000.0 * (273.15 / temperature) * (pressure / 101.325)
        if to_key == "ppm_mass":
            if not (molecular_weight is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            return mg_nm3_value * 24.45 / molecular_weight
        msg = f"Conversion to {to_unit} not implemented"
        raise ValueError(msg)

    def syngas_composition(self, value: float, from_unit: str, to_unit: str) -> float:
        from_key = from_unit.lower()
        to_key = to_unit.lower()

        if from_key == to_key:
            return value

        if {from_key, to_key} <= {"mol%", "vol%"}:
            return value

        conversions = {
            ("ppm", "ppb"): 1000.0,
            ("ppb", "ppm"): 0.001,
            ("ppm", "%"): 0.0001,
            ("%", "ppm"): 10000.0,
            ("ppb", "%"): 0.0000001,
            ("%", "ppb"): 10000000.0,
            ("ppm", "mol%"): 0.0001,
            ("mol%", "ppm"): 10000.0,
            ("ppm", "vol%"): 0.0001,
            ("vol%", "ppm"): 10000.0,
        }

        key = (from_key, to_key)
        if key in conversions:
            return value * conversions[key]

        msg = f"Conversion from {from_unit} to {to_unit} not supported"
        raise ValueError(msg)
