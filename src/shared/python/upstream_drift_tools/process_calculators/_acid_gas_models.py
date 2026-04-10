from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AcidGasComposition:
    """Composition of acid gases and water vapor in syngas"""

    h2o: float = 0.0  # Water vapor mole fraction
    hf: float = 0.0  # Hydrogen fluoride mole fraction
    hcl: float = 0.0  # Hydrogen chloride mole fraction
    h2s: float = 0.0  # Hydrogen sulfide mole fraction
    other: float = 0.0  # Other components (H2, CO, CO2, etc.)
    name: str = ""

    def normalize(self) -> AcidGasComposition:
        """Normalize composition to sum to 1.0"""
        total = self.h2o + self.hf + self.hcl + self.h2s + self.other
        if total > 0:
            return AcidGasComposition(
                h2o=self.h2o / total,
                hf=self.hf / total,
                hcl=self.hcl / total,
                h2s=self.h2s / total,
                other=self.other / total,
                name=self.name,
            )
        return self

    def to_dict(self) -> dict[str, float]:
        """Convert composition to dictionary format.

        Returns:
            Dictionary with component names as keys and mole fractions as values.
        """
        return {
            "H2O": self.h2o,
            "HF": self.hf,
            "HCl": self.hcl,
            "H2S": self.h2s,
            "Other": self.other,
        }

    @property
    def total(self) -> float:
        """Total mole fraction (should be 1.0 for normalized composition).

        Returns:
            Sum of all component mole fractions.
        """
        return self.h2o + self.hf + self.hcl + self.h2s + self.other


@dataclass
class DewpointResult:
    """Comprehensive dewpoint calculation results"""

    # Input conditions
    temperature_c: float
    temperature_k: float
    pressure_bar: float
    pressure_pa: float
    composition: AcidGasComposition

    # Individual acid gas dewpoints
    h2o_dewpoint_c: float
    hf_dewpoint_c: float
    hcl_dewpoint_c: float
    h2s_dewpoint_c: float

    # Overall dewpoint (highest among all components)
    overall_dewpoint_c: float
    limiting_component: str

    # Vapor pressures at current conditions
    h2o_vapor_pressure_pa: float
    hf_vapor_pressure_pa: float
    hcl_vapor_pressure_pa: float
    h2s_vapor_pressure_pa: float

    # Partial pressures
    h2o_partial_pressure_pa: float
    hf_partial_pressure_pa: float
    hcl_partial_pressure_pa: float
    h2s_partial_pressure_pa: float

    # Safety margins
    dewpoint_margin_c: float
    condensation_risk: str

    # Additional info
    calculation_method: str
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export.

        Returns:
            Dictionary containing all result data in exportable format.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "input": {
                "temperature_c": self.temperature_c,
                "pressure_bar": self.pressure_bar,
                "composition": self.composition.to_dict(),
            },
            "dewpoints": {
                "H2O": self.h2o_dewpoint_c,
                "HF": self.hf_dewpoint_c,
                "HCl": self.hcl_dewpoint_c,
                "H2S": self.h2s_dewpoint_c,
                "overall": self.overall_dewpoint_c,
                "limiting_component": self.limiting_component,
            },
            "vapor_pressures_pa": {
                "H2O": self.h2o_vapor_pressure_pa,
                "HF": self.hf_vapor_pressure_pa,
                "HCl": self.hcl_vapor_pressure_pa,
                "H2S": self.h2s_vapor_pressure_pa,
            },
            "safety": {
                "dewpoint_margin_c": self.dewpoint_margin_c,
                "condensation_risk": self.condensation_risk,
            },
            "method": self.calculation_method,
            "sources": self.sources,
            "warnings": self.warnings,
        }
