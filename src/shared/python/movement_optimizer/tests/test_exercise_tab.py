# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for exercise_tab.py business logic.

Uses QT_QPA_PLATFORM=offscreen to avoid requiring a display.  All rendering
calls that touch matplotlib are patched so the tests remain fast and
deterministic.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

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
# Helpers
# ---------------------------------------------------------------------------


def _make_result(seed: int = 0):
    """Return a minimal OptimizationResult for testing."""
    from conftest import make_test_result

    return make_test_result(seed=seed, cost=99.0)


def _make_body():
    from movement_optimizer.models import BodyModel

    return BodyModel(75.0, 1.75)


# ---------------------------------------------------------------------------
# ExerciseTab construction
# ---------------------------------------------------------------------------


class TestExerciseTabConstruction:
    def test_tab_has_expected_axes_keys(self) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Test Exercise")
        expected_keys = {
            "anim",
            "com_path",
            "angles",
            "torques",
            "power",
            "com_time",
            "spine_comp",
            "spine_shear",
        }
        assert set(tab.axes.keys()) == expected_keys

    def test_tab_name_attribute_set(self) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Squat")
        assert tab.name == "Squat"

    def test_tab_has_figure(self) -> None:
        from matplotlib.figure import Figure

        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Squat")
        assert isinstance(tab.fig, Figure)

    def test_tab_has_canvas(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Deadlift")
        assert isinstance(tab.canvas, FigureCanvasQTAgg)

    def test_anim_axis_equal_aspect(self) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Squat")
        # The anim axis should have equal aspect set.  matplotlib may return
        # the string "equal" or the numeric ratio 1.0 depending on version.
        aspect = tab.axes["anim"].get_aspect()
        assert aspect == "equal" or aspect == pytest.approx(1.0)

    def test_all_axes_are_matplotlib_axes(self) -> None:
        from matplotlib.axes import Axes

        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Bench Press")
        for key, ax in tab.axes.items():
            assert isinstance(ax, Axes), f"axes['{key}'] is not an Axes instance"


# ---------------------------------------------------------------------------
# ExerciseTab.draw_all_plots delegates correctly
# ---------------------------------------------------------------------------


class TestExerciseTabDrawAllPlots:
    """Verify draw_all_plots dispatches to the correct plot_renderer helpers."""

    def _tab_with_patches(self) -> tuple:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Squat")
        return tab  # type: ignore

    @patch("movement_optimizer.gui.exercise_tab.plot_renderer")
    def test_draw_all_plots_calls_plot_angles(self, mock_renderer) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Squat")
        result = _make_result()
        body = _make_body()
        tab.draw_all_plots(result, body, 60.0, exercise_type="squat")
        mock_renderer.plot_angles.assert_called_once()

    @patch("movement_optimizer.gui.exercise_tab.plot_renderer")
    def test_draw_all_plots_calls_plot_torques(self, mock_renderer) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Squat")
        result = _make_result()
        body = _make_body()
        tab.draw_all_plots(result, body, 60.0, exercise_type="squat")
        mock_renderer.plot_torques.assert_called_once()

    @patch("movement_optimizer.gui.exercise_tab.plot_renderer")
    def test_draw_all_plots_calls_plot_power(self, mock_renderer) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Deadlift")
        result = _make_result()
        body = _make_body()
        tab.draw_all_plots(result, body, 60.0, exercise_type="deadlift")
        mock_renderer.plot_power.assert_called_once()

    @patch("movement_optimizer.gui.exercise_tab.plot_renderer")
    def test_draw_all_plots_calls_plot_spine_loads(self, mock_renderer) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Deadlift")
        result = _make_result()
        body = _make_body()
        tab.draw_all_plots(result, body, 80.0, exercise_type="deadlift")
        mock_renderer.plot_spine_loads.assert_called_once()

    @patch("movement_optimizer.gui.exercise_tab.plot_renderer")
    def test_draw_all_plots_uses_bench_labels_for_bench(self, mock_renderer) -> None:
        """Bench press should use BENCH_LABELS, not SEG_LABELS."""
        from movement_optimizer.gui.exercise_tab import ExerciseTab
        from movement_optimizer.rendering import Palette

        tab = ExerciseTab("Bench Press")
        result = _make_result()
        body = _make_body()
        tab.draw_all_plots(result, body, 60.0, exercise_type="bench_press")
        # The labels argument passed to plot_angles should be BENCH_LABELS
        call_args = mock_renderer.plot_angles.call_args
        labels_arg = call_args[0][2]  # positional: (ax, result, labels)
        assert labels_arg is Palette.BENCH_LABELS

    @patch("movement_optimizer.gui.exercise_tab.plot_renderer")
    def test_draw_all_plots_uses_seg_labels_for_squat(self, mock_renderer) -> None:
        """Squat should use SEG_LABELS, not BENCH_LABELS."""
        from movement_optimizer.gui.exercise_tab import ExerciseTab
        from movement_optimizer.rendering import Palette

        tab = ExerciseTab("Squat")
        result = _make_result()
        body = _make_body()
        tab.draw_all_plots(result, body, 60.0, exercise_type="squat")
        call_args = mock_renderer.plot_angles.call_args
        labels_arg = call_args[0][2]
        assert labels_arg is Palette.SEG_LABELS

    @patch("movement_optimizer.gui.exercise_tab.plot_renderer")
    def test_draw_all_plots_updates_fig_title(self, mock_renderer) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Squat")
        result = _make_result()
        body = _make_body()
        tab.draw_all_plots(result, body, 60.0, exercise_type="squat")
        title = tab.fig._suptitle  # type: ignore
        assert title is not None
        title_text = title.get_text()
        assert "75" in title_text  # body mass
        assert "60" in title_text  # bar mass


# ---------------------------------------------------------------------------
# ExerciseTab.draw_anim_frame delegates correctly
# ---------------------------------------------------------------------------


class TestExerciseTabDrawAnimFrame:
    @patch("movement_optimizer.gui.exercise_tab.anim_renderer")
    def test_draw_anim_frame_calls_anim_renderer(self, mock_anim_renderer) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Squat")
        result = _make_result()
        body = _make_body()
        dyn_mock = MagicMock()
        tab.draw_anim_frame(0, result, dyn_mock, body, "squat")
        mock_anim_renderer.draw_anim_frame.assert_called_once()

    @patch("movement_optimizer.gui.exercise_tab.anim_renderer")
    def test_draw_anim_frame_passes_tab_name(self, mock_anim_renderer) -> None:
        """The tab name should be forwarded to the renderer as the exercise display name."""
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Deadlift")
        result = _make_result()
        body = _make_body()
        dyn_mock = MagicMock()
        tab.draw_anim_frame(5, result, dyn_mock, body, "deadlift")
        call_args = mock_anim_renderer.draw_anim_frame.call_args[0]
        # Signature: draw_anim_frame(ax, fi, result, dynamics, body, name, exercise_type)
        assert "Deadlift" in call_args

    @patch("movement_optimizer.gui.exercise_tab.anim_renderer")
    def test_draw_anim_frame_passes_correct_frame_index(self, mock_anim_renderer) -> None:
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        tab = ExerciseTab("Squat")
        result = _make_result()
        body = _make_body()
        dyn_mock = MagicMock()
        tab.draw_anim_frame(7, result, dyn_mock, body, "squat")
        call_args = mock_anim_renderer.draw_anim_frame.call_args[0]
        # fi = 7 should appear in the positional args
        assert 7 in call_args
