"""Statistical analysis module for golf swing biomechanics.

This module re-exports the shared statistical analysis functionality.
"""

from __future__ import annotations

from shared.python.statistical_analysis import (
    KinematicSequenceInfo,
    PeakInfo,
    StatisticalAnalyzer,
    SummaryStatistics,
    SwingPhase,
)

__all__ = [
    "KinematicSequenceInfo",
    "PeakInfo",
    "StatisticalAnalyzer",
    "SummaryStatistics",
    "SwingPhase",
]
