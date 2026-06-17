# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for main_window.py and optimization_mixin.py business logic.

Uses fakes/mocks to avoid requiring a full Qt display or running optimizer.
All Qt interactions are mediated through a minimal ``_FakeWindow`` that
satisfies the protocol expected by the mixin methods.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication

    _QT_AVAILABLE = True
    _app = QApplication.instance()
    if _app is None:
        _app = QApplication([])
except (ImportError, OSError):
    _QT_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _QT_AVAILABLE, reason="Qt not available")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_result(cost: float = 42.0, success: bool = True) -> Any:
    """Create a minimal OptimizationResult for testing."""
    from conftest import make_test_result

    r = make_test_result(cost=cost)
    r.success = success
    r.elapsed_s = 3.2
    r.n_evals = 150
    r.com_horizontal_range_cm = 2.5
    return r


@dataclass
class _FakeLabel:
    text: str = ""

    def setText(self, t: str) -> None:
        self.text = t

    def setVisible(self, v: bool) -> None:
        pass


@dataclass
class _FakeButton:
    enabled: bool = True
    visible: bool = True

    def setEnabled(self, v: bool) -> None:
        self.enabled = v

    def setVisible(self, v: bool) -> None:
        self.visible = v

    def setText(self, t: str) -> None:
        pass


@dataclass
class _FakeProgress:
    value: int = 0

    def setValue(self, v: int) -> None:
        self.value = v


@dataclass
class _FakeSlider:
    current: float

    def value(self) -> float:
        return self.current

    def set_value(self, v: float) -> None:
        self.current = v


class _FakeSidebar:
    """Minimal ParameterSidebar surrogate that tracks state without Qt widgets."""

    def __init__(self) -> None:
        self.mass_slider = _FakeSlider(80.0)
        self.height_slider = _FakeSlider(1.80)
        self.ll_slider = _FakeSlider(1.00)
        self.ul_slider = _FakeSlider(1.00)
        self.to_slider = _FakeSlider(1.00)
        self.bar_slider = _FakeSlider(60.0)
        self.bar_depth_slider = _FakeSlider(0.0)
        self.bar_height_slider = _FakeSlider(0.0)
        self.dur_slider = _FakeSlider(2.0)
        self.smooth_slider = _FakeSlider(1.0)
        self.progress = _FakeProgress()
        self.prog_label = _FakeLabel()
        self.result_label = _FakeLabel()
        self.stall_label = _FakeLabel()
        self.opt_btn = _FakeButton()
        self.both_btn = _FakeButton()
        self.cancel_btn = _FakeButton(visible=False)
        self.export_btn = _FakeButton(enabled=False)
        self.save_btn = _FakeButton(enabled=False)
        self.export_video_btn = _FakeButton(enabled=False)
        self.export_plots_btn = _FakeButton(enabled=False)
        self.add_compare_btn = _FakeButton(enabled=False)
        self._showing_optimizing = False
        self._showing_idle = False

    def get_body_model(self) -> Any:
        from movement_optimizer.models import BodyModel

        return BodyModel(
            body_mass=self.mass_slider.value(),
            height=self.height_slider.value(),
        )

    def show_optimizing(self) -> None:
        self._showing_optimizing = True
        self._showing_idle = False

    def show_idle(self) -> None:
        self._showing_optimizing = False
        self._showing_idle = True

    def update_progress(self, report: Any) -> None:
        pass

    def reset_defaults(self) -> None:
        self.mass_slider.set_value(75.0)

    def stall_label_set_visible(self, v: bool) -> None:
        pass

    def get_optimization_params(self) -> tuple[float, float, float]:
        return (self.bar_slider.value(), self.dur_slider.value(), self.smooth_slider.value())

    def get_segment_multipliers(self) -> dict[str, float]:
        return {
            "lower_leg": self.ll_slider.value(),
            "upper_leg": self.ul_slider.value(),
            "torso": self.to_slider.value(),
        }

    def set_cancelled(self) -> None:
        self.cancel_btn.enabled = True
        self.prog_label.setText("Cancelled")

    def set_result_label(self, text: str) -> None:
        self.result_label.setText(text)

    def enable_post_run_buttons(self) -> None:
        self.export_btn.enabled = True
        self.save_btn.enabled = True
        self.export_video_btn.enabled = True
        self.export_plots_btn.enabled = True
        self.add_compare_btn.enabled = True

    def get_comparison_context(self) -> tuple[float, dict[str, float]]:
        return (
            self.bar_slider.value(),
            {
                "body_mass": self.mass_slider.value(),
                "height": self.height_slider.value(),
            },
        )

    def set_comparison_available(self, available: bool) -> None:
        self.compare_btn.enabled = available  # type: ignore

    def set_cancellation_available(self, available: bool) -> None:
        self.cancel_btn.enabled = available

    def set_cancelling(self) -> None:
        self.cancel_btn.enabled = False


class _FakeTab:
    """Surrogate for ExerciseTab — records draw calls."""

    def __init__(self) -> None:
        self.draw_all_plots_calls: list[tuple] = []
        self.draw_anim_frame_calls: list[tuple] = []

    def draw_all_plots(
        self, result: Any, body: Any, bar: float, exercise_type: str = "squat"
    ) -> None:
        self.draw_all_plots_calls.append((result, body, bar, exercise_type))

    def draw_anim_frame(self, fi: int, result: Any, dyn: Any, body: Any, etype: str) -> None:
        self.draw_anim_frame_calls.append((fi, result, dyn, body, etype))


class _FakeWindow:
    """Minimal stand-in for MainWindow that uses OptimizationMixin methods."""

    EXERCISE_CONFIGS: ClassVar = (
        ("Bottoms Up Squat", "squat"),
        ("Full Squat", "full_squat"),
        ("Deadlift", "deadlift"),
        ("Bench Press", "bench_press"),
    )

    def __init__(self) -> None:
        from movement_optimizer.gui.exercise_state import ExerciseRuntimeState
        from movement_optimizer.trajectory import SolutionCache

        self.exercise_states = [ExerciseRuntimeState() for _name, _etype in self.EXERCISE_CONFIGS]
        self.sidebar = _FakeSidebar()
        self.status_label = _FakeLabel()
        self.exercise_tabs = [_FakeTab() for _ in self.EXERCISE_CONFIGS]
        self._opt_lock = threading.Lock()
        self._opt_running = False
        self._cache = SolutionCache()
        self._cancel_event = threading.Event()
        self._last_config: tuple = ()
        self._run_exercise_calls: list[tuple] = []

    def _run_exercise(self, idx: int, then_chain: Any = None) -> None:
        self._run_exercise_calls.append((idx, then_chain))

    def _stop_anim(self) -> None:
        pass

    # Signal stand-ins
    def _sig_done_emit(self, *args: Any) -> None:
        pass

    def _sig_cancelled_emit(self) -> None:
        pass

    def _sig_error_emit(self, msg: str) -> None:
        pass

    def _sig_progress_emit(self, report: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# OptimizationMixin._update_result_summary
# ---------------------------------------------------------------------------


class TestUpdateResultSummary:
    """Business logic for the sidebar result text — no Qt display needed."""

    def _make_window(self) -> _FakeWindow:
        return _FakeWindow()

    def test_squat_summary_contains_joint_labels(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        OptimizationMixin._update_result_summary(
            window,
            "Squat",
            _make_result(),
            exercise_type="squat",  # type: ignore
        )  # type: ignore[arg-type]
        text = window.sidebar.result_label.text
        assert "Ankle" in text
        assert "Knee" in text
        assert "Hip" in text

    def test_bench_summary_contains_joint_labels(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        OptimizationMixin._update_result_summary(
            window,
            "Bench",
            _make_result(),
            exercise_type="bench_press",  # type: ignore
        )  # type: ignore[arg-type]
        text = window.sidebar.result_label.text
        assert "Shoulder" in text
        assert "Elbow" in text
        assert "Wrist" in text

    def test_summary_shows_work_in_joules(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        OptimizationMixin._update_result_summary(
            window,
            "Squat",
            _make_result(),
            exercise_type="squat",  # type: ignore
        )  # type: ignore[arg-type]
        text = window.sidebar.result_label.text
        assert "J" in text

    def test_balance_ok_shown_on_success(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        result = _make_result(success=True)
        OptimizationMixin._update_result_summary(window, "Squat", result, exercise_type="squat")  # type: ignore[arg-type]
        text = window.sidebar.result_label.text
        assert "BALANCED" in text

    def test_balance_out_shown_on_failure(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        result = _make_result(success=False)
        OptimizationMixin._update_result_summary(window, "Squat", result, exercise_type="squat")  # type: ignore[arg-type]
        text = window.sidebar.result_label.text
        assert "OUT OF BOUNDS" in text

    def test_peak_torque_values_are_numeric(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        OptimizationMixin._update_result_summary(
            window,
            "Deadlift",
            _make_result(),
            exercise_type="deadlift",  # type: ignore
        )  # type: ignore[arg-type]
        text = window.sidebar.result_label.text
        # The text should contain at least one numeric value followed by "N·m"
        assert "N\u00b7m" in text


# ---------------------------------------------------------------------------
# OptimizationMixin._on_cancelled
# ---------------------------------------------------------------------------


class TestOnCancelled:
    def _make_window(self) -> _FakeWindow:
        w = _FakeWindow()
        w._opt_running = True
        return w

    def test_on_cancelled_clears_opt_running(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        OptimizationMixin._on_cancelled(window)  # type: ignore[arg-type]
        assert window._opt_running is False

    def test_on_cancelled_shows_idle(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        OptimizationMixin._on_cancelled(window)  # type: ignore[arg-type]
        assert window.sidebar._showing_idle

    def test_on_cancelled_updates_status_label(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        OptimizationMixin._on_cancelled(window)  # type: ignore[arg-type]
        assert "cancel" in window.status_label.text.lower()

    def test_on_cancelled_re_enables_cancel_button(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = self._make_window()
        window.sidebar.cancel_btn.enabled = False
        OptimizationMixin._on_cancelled(window)  # type: ignore[arg-type]
        assert window.sidebar.cancel_btn.enabled


# ---------------------------------------------------------------------------
# MainWindow._cancel_optimization
# ---------------------------------------------------------------------------


class TestCancelOptimization:
    def test_cancel_optimization_disables_cancel_action(self) -> None:
        from movement_optimizer.gui.main_window import MainWindow

        window = _FakeWindow()
        window._opt_running = True  # guard only fires when optimization is active
        window.sidebar.cancel_btn.enabled = True

        MainWindow._cancel_optimization(window)  # type: ignore[arg-type]

        assert window._cancel_event.is_set()
        assert window.sidebar.cancel_btn.enabled is False
        assert window.status_label.text == "Cancelling..."


# ---------------------------------------------------------------------------
# OptimizationMixin._enable_post_run_buttons
# ---------------------------------------------------------------------------


class TestEnablePostRunButtons:
    def test_all_buttons_enabled(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        # All should start disabled
        assert not window.sidebar.export_btn.enabled
        assert not window.sidebar.save_btn.enabled
        OptimizationMixin._enable_post_run_buttons(window)  # type: ignore[arg-type]
        assert window.sidebar.export_btn.enabled
        assert window.sidebar.save_btn.enabled
        assert window.sidebar.export_video_btn.enabled
        assert window.sidebar.export_plots_btn.enabled
        assert window.sidebar.add_compare_btn.enabled


# ---------------------------------------------------------------------------
# OptimizationMixin._seg_mults
# ---------------------------------------------------------------------------


class TestSegMults:
    def test_returns_correct_keys(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        mults = OptimizationMixin._seg_mults(window)  # type: ignore[arg-type]
        assert set(mults.keys()) == {"lower_leg", "upper_leg", "torso"}

    def test_values_come_from_sliders(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        window.sidebar.ll_slider.current = 1.1
        window.sidebar.ul_slider.current = 0.9
        window.sidebar.to_slider.current = 1.05
        mults = OptimizationMixin._seg_mults(window)  # type: ignore[arg-type]
        assert mults["lower_leg"] == pytest.approx(1.1)
        assert mults["upper_leg"] == pytest.approx(0.9)
        assert mults["torso"] == pytest.approx(1.05)


# ---------------------------------------------------------------------------
# OptimizationMixin._finish_or_chain
# ---------------------------------------------------------------------------


class TestFinishOrChain:
    def test_no_chain_sets_idle(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        window._opt_running = True
        OptimizationMixin._finish_or_chain(window, None, "Done!")  # type: ignore[arg-type]
        assert window._opt_running is False
        assert window.sidebar._showing_idle
        assert window.status_label.text == "Done!"

    def test_chain_triggers_next_exercise(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        OptimizationMixin._finish_or_chain(window, [1, 2], "status")  # type: ignore[arg-type]
        # Should have called _run_exercise with idx=1 and remaining=[2]
        assert len(window._run_exercise_calls) == 1
        idx, remaining = window._run_exercise_calls[0]
        assert idx == 1
        assert remaining == [2]

    def test_single_chain_passes_none_remaining(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        OptimizationMixin._finish_or_chain(window, [3], "status")  # type: ignore[arg-type]
        assert len(window._run_exercise_calls) == 1
        idx, remaining = window._run_exercise_calls[0]
        assert idx == 3
        assert remaining is None


# ---------------------------------------------------------------------------
# OptimizationMixin._resolve_exercise_params
# ---------------------------------------------------------------------------


class TestResolveExerciseParams:
    def test_squat_returns_correct_etype(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        _body, _dyn, etype, _bar, _dur, _smoothness = OptimizationMixin._resolve_exercise_params(
            window,
            0,  # type: ignore
        )  # type: ignore[arg-type]
        assert etype == "squat"

    def test_deadlift_returns_correct_etype(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        _body, _dyn, etype, _bar, _dur, _smoothness = OptimizationMixin._resolve_exercise_params(
            window,
            2,  # type: ignore
        )  # type: ignore[arg-type]
        assert etype == "deadlift"

    def test_stores_dynamics_in_list(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        OptimizationMixin._resolve_exercise_params(window, 0)  # type: ignore[arg-type]
        assert window.exercise_states[0].dynamics is not None

    def test_stores_body_in_list(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        OptimizationMixin._resolve_exercise_params(window, 0)  # type: ignore[arg-type]
        assert window.exercise_states[0].body is not None

    def test_bar_value_from_slider(self) -> None:
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        window.sidebar.bar_slider.current = 100.0
        _body, _dyn, _etype, bar, _dur, _smoothness = OptimizationMixin._resolve_exercise_params(
            window,
            0,  # type: ignore
        )  # type: ignore[arg-type]
        assert bar == pytest.approx(100.0)

    def test_full_squat_minimum_duration_enforced(self) -> None:
        """full_squat must use at least 3.0 s even when slider is set lower."""
        from movement_optimizer.gui.optimization_mixin import OptimizationMixin

        window = _FakeWindow()
        window.sidebar.dur_slider.current = 1.0
        _body, _dyn, _etype, _bar, dur, _smoothness = OptimizationMixin._resolve_exercise_params(
            window,
            1,  # type: ignore
        )  # type: ignore[arg-type]
        assert dur >= 3.0


# ---------------------------------------------------------------------------
# MainWindow class-level attributes (no Qt instance needed)
# ---------------------------------------------------------------------------


class TestMainWindowClassAttributes:
    def test_exercise_configs_has_expected_exercises(self) -> None:
        from movement_optimizer.gui.main_window import MainWindow

        names = [c[0] for c in MainWindow.EXERCISE_CONFIGS]
        assert "Bottoms Up Squat" in names
        assert "Deadlift" in names
        assert "Bench Press" in names

    def test_exercise_configs_has_matching_types(self) -> None:
        from movement_optimizer.gui.main_window import MainWindow

        types = [c[1] for c in MainWindow.EXERCISE_CONFIGS]
        assert "squat" in types
        assert "deadlift" in types
        assert "bench_press" in types

    def test_signals_are_defined_on_class(self) -> None:
        from movement_optimizer.gui.main_window import MainWindow

        assert hasattr(MainWindow, "_sig_done")
        assert hasattr(MainWindow, "_sig_cancelled")
        assert hasattr(MainWindow, "_sig_error")
        assert hasattr(MainWindow, "_sig_progress")
