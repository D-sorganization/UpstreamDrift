"""Tests for src.shared.python.analysis.phase_detection (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.analysis.dataclasses import SwingPhase
from src.shared.python.analysis.phase_detection import PhaseDetectionMixin


class _Concrete(PhaseDetectionMixin):
    def __init__(self, n: int = 100) -> None:
        t = np.linspace(0.0, 1.0, n)
        # Simulate club head speed that peaks at ~0.7 seconds
        speed = np.zeros(n)
        speed += 5.0 * np.exp(
            -((t - 0.3) ** 2) / 0.005
        )  # small bump (top of backswing)
        speed += 30.0 * np.exp(-((t - 0.7) ** 2) / 0.003)  # large peak (impact)
        self.times = t
        self.club_head_speed = speed
        self.duration = float(t[-1] - t[0])


class TestPhaseDetectionMixin:
    def setup_method(self) -> None:
        self.obj = _Concrete(n=100)

    def test_phase_detection_returns_list(self) -> None:
        phases = self.obj.detect_swing_phases()
        assert isinstance(phases, list)

    def test_returns_non_empty_list(self) -> None:
        phases = self.obj.detect_swing_phases()
        assert len(phases) > 0

    def test_phases_are_swing_phase_objects(self) -> None:
        phases = self.obj.detect_swing_phases()
        for p in phases:
            assert isinstance(p, SwingPhase)

    def test_each_phase_has_positive_duration(self) -> None:
        phases = self.obj.detect_swing_phases()
        for p in phases:
            assert p.duration >= 0.0

    def test_start_time_le_end_time(self) -> None:
        phases = self.obj.detect_swing_phases()
        for p in phases:
            assert p.start_time <= p.end_time

    def test_start_index_le_end_index(self) -> None:
        phases = self.obj.detect_swing_phases()
        for p in phases:
            assert p.start_index <= p.end_index

    def test_fallback_single_phase_when_no_speed(self) -> None:
        obj = _Concrete(n=5)
        obj.club_head_speed = None
        phases = obj.detect_swing_phases()
        assert len(phases) == 1
        assert phases[0].name == "Complete Swing"

    def test_fallback_single_phase_when_short_data(self) -> None:
        obj = _Concrete(n=10)  # < 20 → fallback
        phases = obj.detect_swing_phases()
        assert len(phases) == 1
        assert phases[0].name == "Complete Swing"

    def test_phase_names_are_strings(self) -> None:
        phases = self.obj.detect_swing_phases()
        for p in phases:
            assert isinstance(p.name, str)
            assert len(p.name) > 0
