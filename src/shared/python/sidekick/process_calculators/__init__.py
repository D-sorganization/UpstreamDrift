"""Process Engineering Calculators - Standalone Utility Tools.

This package contains standalone process engineering calculators that can be used
independently of any specific application. These tools are designed to be reusable
across multiple projects.

Categories:
    - Equipment Sizing: Pressure drop, scrubbers, baghouse, flare
    - Thermodynamic: Acid gas dewpoint, syngas water, WGS reactor
    - Process Analysis: PSA, syngas compression, optimization
    - Financial: NPV/IRR analysis, project economics
    - Mathematical: ODE solver, thermal profiles, multi-parameter analysis

Usage:
    from sidekick.process_calculators import (
        AcidGasDewpointCalculator,
        ScrubberCalculator,
        FlareCalculator,
        FinancialCalculator,
    )

Note:
    Some calculators have optional dependencies (PyQt6, scipy, etc.).
    Core calculation functionality works without UI dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .acid_gas_dewpoint_calculator import AcidGasDewpointCalculator
from .baghouse_calculator import BaghouseCalculator, BaghouseResult
from .constants import (
    ATOL_ZERO,
    R_GAS_J_MOL_K,
    R_UNIVERSAL,
    STANDARD_GRAVITY,
    celsius_to_kelvin,
    kelvin_to_celsius,
)
from .electrode_advancement_calculator import ElectrodeAdvancementCalculator
from .financial_calculator import FinancialModelCalculator as FinancialCalculator
from .flare_calculator import FlareCalculator, FlareDesign
from .ode_solver import ODESolver
from .thermal_profile_predictor import (
    fit_heating_parameters,
    predict_temperature_profile,
)

if TYPE_CHECKING:
    from .pressure_drop_calculator import (
        PressureDropCalculator as PressureDropCalculatorType,
    )
    from .syngas_water_calculator import (
        SyngasWaterCalculator as SyngasWaterCalculatorType,
    )
    from .water_vapor_pressure_calculator import (
        WaterVaporPressureCalculator as WaterVaporPressureCalculatorType,
    )
    from .wgs_reactor_calculator import WGSReactorEngine as WGSReactorEngineType

_logger = logging.getLogger(__name__)

# Track import errors for optional modules
_import_errors: list[str] = []

# Calculators with numpy/scipy dependencies
try:
    ScrubberCalculator = None  # Module has functions, no class
except ImportError as e:
    _import_errors.append(f"ScrubberCalculator not available: {e}")
    ScrubberCalculator = None

# Pre-declare with type annotations to avoid no-redef errors in except branches
SyngasWaterCalculator: type[SyngasWaterCalculatorType] | None
WaterVaporPressureCalculator: type[WaterVaporPressureCalculatorType] | None
try:
    from .syngas_water_calculator import SyngasWaterCalculator
    from .water_vapor_pressure_calculator import (
        WaterVaporPressureCalculator,
    )
except ImportError as e:
    _import_errors.append(f"Water calculators not available: {e}")
    SyngasWaterCalculator = None
    WaterVaporPressureCalculator = None

# Pre-declare with type annotations to avoid no-redef errors in except branches
WGSReactorCalculator: type[WGSReactorEngineType] | None
WGSReactorEngine: type[WGSReactorEngineType] | None
try:
    from .wgs_reactor_calculator import WGSReactorEngine

    WGSReactorCalculator = WGSReactorEngine  # Alias
except ImportError as e:
    _import_errors.append(f"WGSReactorCalculator not available: {e}")
    WGSReactorEngine = None
    WGSReactorCalculator = None

# Pre-declare callable aliases to avoid assignment type errors in except branches
Optimizer: Callable[..., Any] | None = None
AdamOptimizer: Callable[..., Any] | None = None
try:
    from .optimization import (
        find_optimal_on_surface,
        run_adam_optimization,
    )

    AdamOptimizer = run_adam_optimization  # Alias for backwards compatibility
    Optimizer = find_optimal_on_surface  # Alias for backwards compatibility
except ImportError as e:
    _import_errors.append(f"Optimization not available: {e}")

# Pre-declare callable alias to avoid assignment type error in except branch
MultiParameterAnalysis: Callable[..., Any] | None = None
try:
    from .multi_param_analysis import (
        run_multi_parameter_analysis,
    )

    MultiParameterAnalysis = (
        run_multi_parameter_analysis  # Alias for backwards compatibility
    )
except ImportError as e:
    _import_errors.append(f"MultiParameterAnalysis not available: {e}")

# UI-dependent calculators (require PyQt6)
# Pre-declare to avoid type errors in except branch
SyngasCompressionCalculator: Any = None
try:
    from .syngas_compression_calculator import (
        SyngasCompressionCalculatorWidget,
    )

    SyngasCompressionCalculator = SyngasCompressionCalculatorWidget  # Alias
except ImportError as e:
    _import_errors.append(f"SyngasCompressionCalculator not available: {e}")

# Modular packages — pre-declare to avoid no-redef in except branch
PressureDropCalculator: type[PressureDropCalculatorType] | None
try:
    from .pressure_drop_calculator import (
        PressureDropCalculator,
    )
except ImportError as e:
    _import_errors.append(f"PressureDropCalculator not available: {e}")
    PressureDropCalculator = None

# Log any import errors that occurred
for error in _import_errors:
    _logger.debug(error)

__all__ = [
    "ATOL_ZERO",
    "AcidGasDewpointCalculator",
    "AdamOptimizer",
    "BaghouseCalculator",
    "BaghouseResult",
    "ElectrodeAdvancementCalculator",
    "FinancialCalculator",
    "FlareCalculator",
    "FlareDesign",
    "MultiParameterAnalysis",
    "ODESolver",
    "Optimizer",
    "PressureDropCalculator",
    "R_GAS_J_MOL_K",
    "R_UNIVERSAL",
    "STANDARD_GRAVITY",
    "ScrubberCalculator",
    "SyngasCompressionCalculator",
    "SyngasWaterCalculator",
    "WGSReactorCalculator",
    "WGSReactorEngine",
    "WaterVaporPressureCalculator",
    "celsius_to_kelvin",
    "fit_heating_parameters",
    "kelvin_to_celsius",
    "predict_temperature_profile",
]
