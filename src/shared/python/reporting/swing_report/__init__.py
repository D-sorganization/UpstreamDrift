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
from .validator import (
    SwingReportContractError,
    Violation,
    validate_swing_report,
)

__all__ = [
    "EngineResult",
    "QuantityData",
    "ADVERTISED",
    "POLICY",
    "WITHHELD",
    "QuantityClass",
    "SwingReport",
    "SwingReportContractError",
    "Violation",
    "build_swing_report",
    "classify",
    "result_from_reporting_mixin",
    "validate_swing_report",
]
