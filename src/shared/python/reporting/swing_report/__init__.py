"""Cross-engine swing report contract (epic #11419, slice #12324)."""

from .adapters_base import EngineResult, QuantityData, result_from_reporting_mixin
from .builder import build_swing_report
from .contract import SwingReport
from .policy import (
    ADVERTISED,
    POLICY,
    WITHHELD,
    QuantityClass,
    classify,
)
from .plots import PLOT_REGISTRY, PlotSpec, make_registry
from .selection import PlotSelection, UnknownPlotError, resolve_plots, select_plots
from .validator import (
    SwingReportContractError,
    Violation,
    validate_swing_report,
)

__all__ = [
    "EngineResult",
    "QuantityData",
    "ADVERTISED",
    "PLOT_REGISTRY",
    "POLICY",
    "PlotSelection",
    "PlotSpec",
    "WITHHELD",
    "QuantityClass",
    "SwingReport",
    "SwingReportContractError",
    "UnknownPlotError",
    "Violation",
    "build_swing_report",
    "classify",
    "make_registry",
    "resolve_plots",
    "select_plots",
    "result_from_reporting_mixin",
    "validate_swing_report",
]
