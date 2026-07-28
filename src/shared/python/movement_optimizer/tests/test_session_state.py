# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for GUI session-state helpers without requiring Qt widgets."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import ClassVar

from movement_optimizer.gui.exercise_state import ExerciseRuntimeState
from movement_optimizer.gui.session_state import (
    collect_results,
    collect_slider_values,
    restore_slider_values,
)
from movement_optimizer.tests._helpers import make_test_result


@dataclass
class _FakeSlider:
    current: float

    def value(self) -> float:
        return self.current

    def set_value(self, value: float) -> None:
        self.current = value


class _FakeSidebar:
    def __init__(self) -> None:
        self.mass_slider = _FakeSlider(75.0)
        self.height_slider = _FakeSlider(1.75)
        self.ll_slider = _FakeSlider(1.0)
        self.ul_slider = _FakeSlider(1.0)
        self.to_slider = _FakeSlider(1.0)
        self.bar_slider = _FakeSlider(60.0)
        self.dur_slider = _FakeSlider(2.0)
        self.smooth_slider = _FakeSlider(1.0)


class _FakeWindow:
    EXERCISE_CONFIGS: ClassVar = [
        ("Squat", "squat"),
        ("Deadlift", "deadlift"),
        ("Bench", "bench_press"),
    ]

    def __init__(self) -> None:
        self.exercise_states = [
            ExerciseRuntimeState(result=make_test_result(cost=10.0)),
            ExerciseRuntimeState(),
            ExerciseRuntimeState(result=make_test_result(cost=20.0)),
        ]
        # collect_results acquires this lock to take a consistent snapshot.
        self._opt_lock = threading.RLock()


class TestSessionState:
    def test_collect_results_keeps_only_non_null_entries(self):
        results = collect_results(_FakeWindow())
        assert list(results) == ["squat", "bench_press"]
        assert results["squat"].cost == 10.0
        assert results["bench_press"].cost == 20.0

    def test_collect_and_restore_slider_values(self):
        sidebar = _FakeSidebar()
        values = collect_slider_values(sidebar)
        assert values == {
            "body_mass": 75.0,
            "height": 1.75,
            "lower_leg": 1.0,
            "upper_leg": 1.0,
            "torso": 1.0,
            "bar_mass": 60.0,
            "duration": 2.0,
            "smoothness": 1.0,
        }

        restore_slider_values(
            sidebar,
            {
                "body_mass": 82.0,
                "height": 1.82,
                "bar_mass": 110.0,
                "smoothness": 1.4,
            },
        )

        assert sidebar.mass_slider.value() == 82.0
        assert sidebar.height_slider.value() == 1.82
        assert sidebar.bar_slider.value() == 110.0
        assert sidebar.smooth_slider.value() == 1.4
        assert sidebar.ll_slider.value() == 1.0
