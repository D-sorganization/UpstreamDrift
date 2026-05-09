"""Tests for src.shared.python.injury.spinal_load_analysis (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.injury.spinal_load_analysis import (
    SpinalLoadAnalyzer,
    SpinalLoadResult,
    SpinalRiskLevel,
)


def _make_analyzer(body_weight: float = 80.0) -> SpinalLoadAnalyzer:
    return SpinalLoadAnalyzer(body_weight=body_weight, height=1.75)


def _make_inputs(n: int = 100) -> dict:
    t = np.linspace(0.0, 1.5, n)
    angles = {
        "lumbar_flexion": 30.0 * np.sin(np.linspace(0, np.pi, n)),
        "lumbar_rotation": 20.0 * np.sin(np.linspace(0, 2 * np.pi, n)),
        "lumbar_lateral_bend": 10.0 * np.sin(np.linspace(0, 2 * np.pi, n)),
    }
    velocities = {k: np.gradient(v, t[1] - t[0]) for k, v in angles.items()}
    torques = {k: v * 0.5 for k, v in velocities.items()}
    return {
        "joint_angles": angles,
        "joint_velocities": velocities,
        "joint_torques": torques,
        "time": t,
    }


class TestSpinalLoadAnalyzer:
    def test_spinal_load_analysis_construction(self) -> None:
        analyzer = _make_analyzer()
        assert analyzer.body_weight == pytest.approx(80.0)

    def test_default_segments(self) -> None:
        analyzer = _make_analyzer()
        assert "L4-L5" in analyzer.lumbar_segments
        assert "L5-S1" in analyzer.lumbar_segments

    def test_custom_segments(self) -> None:
        analyzer = SpinalLoadAnalyzer(body_weight=70.0, lumbar_segments=["L4-L5"])
        assert analyzer.lumbar_segments == ["L4-L5"]

    def test_analyze_returns_result(self) -> None:
        analyzer = _make_analyzer()
        inputs = _make_inputs()
        result = analyzer.analyze(**inputs)
        assert isinstance(result, SpinalLoadResult)

    def test_result_has_time(self) -> None:
        analyzer = _make_analyzer()
        inputs = _make_inputs()
        result = analyzer.analyze(**inputs)
        assert len(result.time) > 0

    def test_result_peak_compression_non_negative(self) -> None:
        analyzer = _make_analyzer()
        inputs = _make_inputs()
        result = analyzer.analyze(**inputs)
        assert result.peak_compression_bw >= 0.0

    def test_result_has_overall_risk(self) -> None:
        analyzer = _make_analyzer()
        inputs = _make_inputs()
        result = analyzer.analyze(**inputs)
        assert isinstance(result.overall_risk, SpinalRiskLevel)

    def test_result_has_segments(self) -> None:
        analyzer = _make_analyzer()
        inputs = _make_inputs()
        result = analyzer.analyze(**inputs)
        assert isinstance(result.segments, dict)

    def test_risk_levels_are_valid(self) -> None:
        analyzer = _make_analyzer()
        inputs = _make_inputs()
        result = analyzer.analyze(**inputs)
        valid_levels = set(SpinalRiskLevel)
        assert result.overall_risk in valid_levels
        assert result.compression_risk in valid_levels
        assert result.shear_risk in valid_levels


class TestSpinalRiskLevel:
    def test_safe_exists(self) -> None:
        assert SpinalRiskLevel.SAFE is not None

    def test_caution_exists(self) -> None:
        assert SpinalRiskLevel.CAUTION is not None

    def test_high_risk_exists(self) -> None:
        assert SpinalRiskLevel.HIGH_RISK is not None

    def test_critical_exists(self) -> None:
        assert SpinalRiskLevel.CRITICAL is not None


class TestClassConstants:
    def test_compression_thresholds(self) -> None:
        assert (
            SpinalLoadAnalyzer.COMPRESSION_SAFE < SpinalLoadAnalyzer.COMPRESSION_CAUTION
        )
        assert (
            SpinalLoadAnalyzer.COMPRESSION_CAUTION < SpinalLoadAnalyzer.COMPRESSION_HIGH
        )

    def test_shear_thresholds(self) -> None:
        assert SpinalLoadAnalyzer.SHEAR_SAFE < SpinalLoadAnalyzer.SHEAR_CAUTION
        assert SpinalLoadAnalyzer.SHEAR_CAUTION < SpinalLoadAnalyzer.SHEAR_HIGH

    def test_x_factor_thresholds_positive(self) -> None:
        assert SpinalLoadAnalyzer.X_FACTOR_SAFE > 0.0
        assert SpinalLoadAnalyzer.X_FACTOR_CAUTION > SpinalLoadAnalyzer.X_FACTOR_SAFE
