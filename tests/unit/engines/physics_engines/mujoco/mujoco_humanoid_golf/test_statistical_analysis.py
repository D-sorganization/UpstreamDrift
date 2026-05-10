"""Unit tests for statistical_analysis.py re-exports."""

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.statistical_analysis import (
    KinematicSequenceInfo,
    PeakInfo,
    StatisticalAnalyzer,
    SummaryStatistics,
    SwingPhase,
)


def test_re_exports():
    """Test that all required classes are re-exported."""
    assert KinematicSequenceInfo is not None
    assert PeakInfo is not None
    assert StatisticalAnalyzer is not None
    assert SummaryStatistics is not None
    assert SwingPhase is not None
