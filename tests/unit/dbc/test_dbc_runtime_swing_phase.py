"""Runtime DbC tests for swing_metrics and phase_detection contracts.

Tests contracts added to:
- SwingMetricsMixin: ROM non-negative, tempo durations non-negative,
  ratio non-negative, X-factor stretch peak non-negative
- PhaseDetectionMixin: phase durations non-negative, start <= end
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

from src.shared.python.analysis.phase_detection import PhaseDetectionMixin
from src.shared.python.analysis.swing_metrics import SwingMetricsMixin

pytestmark = pytest.mark.unit


class _SwingMetricsHarness(SwingMetricsMixin):
    """Concrete SwingMetricsMixin harness preserving real MRO delegation."""


class _PhaseDetectionHarness(PhaseDetectionMixin):
    """Concrete PhaseDetectionMixin harness preserving real MRO delegation."""

    times: np.ndarray
    duration: float
    club_head_speed: np.ndarray | None


def _make_swing_mixin(n: int = 200, n_joints: int = 4) -> _SwingMetricsHarness:
    """Create a concrete SwingMetricsMixin instance with realistic swing data."""
    rng = np.random.default_rng(42)
    obj = _SwingMetricsHarness()
    obj.times = np.linspace(0, 2, n)
    obj.dt = float(obj.times[1] - obj.times[0])
    obj.joint_positions = rng.standard_normal((n, n_joints)).cumsum(axis=0)

    # Create realistic club head speed: starts slow, ramps up, peaks near end
    t_norm = np.linspace(0, 1, n)
    obj.club_head_speed = np.abs(30.0 * t_norm**2 * np.sin(np.pi * t_norm))
    return obj


def _make_phase_mixin(n: int = 200) -> _PhaseDetectionHarness:
    """Create a concrete PhaseDetectionMixin instance with realistic data."""
    obj = _PhaseDetectionHarness()
    obj.times = np.linspace(0, 2, n)
    obj.duration = 2.0
    # Create realistic speed profile
    t_norm = np.linspace(0, 1, n)
    obj.club_head_speed = np.abs(30.0 * t_norm**2 * np.sin(np.pi * t_norm))
    return obj


# ==================== Swing Metrics Tests ====================


class TestROMPostconditions(unittest.TestCase):
    """Verify ensure() on compute_range_of_motion."""

    def test_rom_non_negative(self) -> None:
        obj = _make_swing_mixin()
        min_a, max_a, rom = obj.compute_range_of_motion(0)
        self.assertGreaterEqual(rom, 0)
        self.assertGreaterEqual(max_a, min_a)

    def test_zero_rom_for_constant(self) -> None:
        obj = _make_swing_mixin()
        obj.joint_positions = np.ones((200, 4))
        min_a, max_a, rom = obj.compute_range_of_motion(0)
        self.assertAlmostEqual(rom, 0.0)

    def test_out_of_range_returns_zeros(self) -> None:
        obj = _make_swing_mixin()
        min_a, max_a, rom = obj.compute_range_of_motion(99)
        self.assertEqual(rom, 0.0)

    def test_all_joints_have_non_negative_rom(self) -> None:
        obj = _make_swing_mixin()
        for j in range(4):
            _, _, rom = obj.compute_range_of_motion(j)
            self.assertGreaterEqual(rom, 0)


class TestTempoPostconditions(unittest.TestCase):
    """Verify ensure() on compute_tempo."""

    def test_valid_tempo(self) -> None:
        obj = _make_swing_mixin()
        result = obj.compute_tempo()
        if result is not None:
            backswing, downswing, ratio = result
            self.assertGreaterEqual(backswing, 0)
            self.assertGreaterEqual(downswing, 0)
            self.assertGreaterEqual(ratio, 0)

    def test_no_club_head_speed_returns_none(self) -> None:
        obj = _make_swing_mixin()
        obj.club_head_speed = None
        result = obj.compute_tempo()
        self.assertIsNone(result)

    def test_short_data_returns_none(self) -> None:
        obj = _make_swing_mixin()
        obj.club_head_speed = np.array([1.0, 2.0, 3.0])
        result = obj.compute_tempo()
        self.assertIsNone(result)


class TestXFactorStretchPostconditions(unittest.TestCase):
    """Verify ensure() and deprecation behavior on compute_x_factor_stretch."""

    def test_peak_stretch_rate_non_negative(self) -> None:
        obj = _make_swing_mixin()
        with self.assertWarns(DeprecationWarning):
            result = obj.compute_x_factor_stretch(0, 1)
        self.assertIsNotNone(result)
        assert result is not None
        _, peak_rate = result
        self.assertGreaterEqual(peak_rate, 0.0)

    def test_dbc_runtime_swing_phase_out_of_range_returns_none(self) -> None:
        obj = _make_swing_mixin()
        with self.assertWarns(DeprecationWarning):
            result = obj.compute_x_factor_stretch(0, 99)
        self.assertIsNone(result)

    def test_constant_joints_zero_stretch(self) -> None:
        obj = _make_swing_mixin()
        obj.joint_positions = np.ones((200, 4))
        with self.assertWarns(DeprecationWarning):
            result = obj.compute_x_factor_stretch(0, 1)
        self.assertIsNotNone(result)
        assert result is not None
        _, peak_rate = result
        self.assertAlmostEqual(peak_rate, 0.0)


class TestXFactorVelocityPostconditions(unittest.TestCase):
    """Verify ensure() on compute_x_factor_velocity directly."""

    def test_peak_velocity_non_negative(self) -> None:
        obj = _make_swing_mixin()
        result = obj.compute_x_factor_velocity(0, 1)
        self.assertIsNotNone(result)
        assert result is not None
        velocity_arr, peak_rate = result
        self.assertEqual(velocity_arr.shape[0], 200)
        self.assertGreaterEqual(peak_rate, 0.0)

    def test_out_of_range_returns_none(self) -> None:
        obj = _make_swing_mixin()
        result = obj.compute_x_factor_velocity(0, 99)
        self.assertIsNone(result)

    def test_constant_joints_zero_velocity(self) -> None:
        obj = _make_swing_mixin()
        obj.joint_positions = np.ones((200, 4))
        result = obj.compute_x_factor_velocity(0, 1)
        self.assertIsNotNone(result)
        assert result is not None
        _, peak_rate = result
        self.assertAlmostEqual(peak_rate, 0.0)


# ==================== Phase Detection Tests ====================


class TestPhaseDetectionPostconditions(unittest.TestCase):
    """Verify ensure() on detect_swing_phases."""

    def test_all_durations_non_negative(self) -> None:
        obj = _make_phase_mixin()
        phases = obj.detect_swing_phases()
        self.assertGreater(len(phases), 0)
        for phase in phases:
            self.assertGreaterEqual(
                phase.duration,
                0,
                f"Phase '{phase.name}' has negative duration: {phase.duration}",
            )

    def test_start_lte_end(self) -> None:
        obj = _make_phase_mixin()
        phases = obj.detect_swing_phases()
        for phase in phases:
            self.assertLessEqual(
                phase.start_time,
                phase.end_time,
                f"Phase '{phase.name}' start > end",
            )
            self.assertLessEqual(
                phase.start_index,
                phase.end_index,
                f"Phase '{phase.name}' start_index > end_index",
            )

    def test_fallback_single_phase(self) -> None:
        obj = _make_phase_mixin()
        obj.club_head_speed = None  # Trigger fallback
        phases = obj.detect_swing_phases()
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0].name, "Complete Swing")
        self.assertGreaterEqual(phases[0].duration, 0)

    def test_short_data_fallback(self) -> None:
        obj = _make_phase_mixin()
        obj.club_head_speed = np.array([1.0, 2.0, 3.0])
        phases = obj.detect_swing_phases()
        self.assertEqual(len(phases), 1)

    def test_standard_phases_present(self) -> None:
        obj = _make_phase_mixin()
        phases = obj.detect_swing_phases()
        phase_names = [p.name for p in phases]
        # Standard golf swing phases
        for expected in [
            "Address",
            "Backswing",
            "Downswing",
            "Impact",
            "Follow-through",
        ]:
            self.assertIn(expected, phase_names, f"Missing phase: {expected}")


if __name__ == "__main__":
    unittest.main()
