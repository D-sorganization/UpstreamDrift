"""
UpstreamDrift Shared Tools Library
==================================

Shared components for Gasification Model and UpstreamDrift (Golf Suite).

Subpackages (import by domain):
    calculators           - Calculation engines (conversion, electrical, mechanical, thermo)
    data_processing       - DataProcessorEngine, readers/writers, typed exceptions
    lab                   - Laboratory tools (bio/C3D reader)
    process_calculators   - Standalone process engineering calculators
    theme                 - Fleet-wide color theme system (13+ themes, PyQt6 integration)
    ui                    - PyQt6 widgets, themes, managers, mixins
    utils                 - Logging, paths, state management, physical constants
"""

from .protocols import (
    CalculationResult,
    Calculator,
    DataTransformer,
    InputValidator,
    ProcessCalculator,
    StateSerializable,
    UnitConverter,
    ValidationResult,
)

__version__ = "0.1.0"

__all__ = [
    # Protocols
    "Calculator",
    "DataTransformer",
    "ProcessCalculator",
    "StateSerializable",
    "UnitConverter",
    # Data classes
    "CalculationResult",
    "ValidationResult",
    # Validation
    "InputValidator",
    # Subpackages (explicit for discovery)
    "calculators",
    "data_processing",
    "lab",
    "process_calculators",
    "theme",
    "ui",
    "utils",
]
