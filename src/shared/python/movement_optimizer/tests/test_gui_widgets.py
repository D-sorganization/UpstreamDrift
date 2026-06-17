# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Basic smoke tests for GUI widget classes.

These tests verify that widget classes can be instantiated and have the
expected attributes/methods without requiring a running QApplication
(they test the class structure, not the rendering).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from movement_optimizer.gui import MainWindow  # noqa: F401

    _GUI_AVAILABLE = True
except (ImportError, OSError):
    _GUI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _GUI_AVAILABLE, reason="Qt bindings not available")


class TestWidgetImports:
    """Verify that all GUI widget classes are importable."""

    def test_import_main_window(self):
        from movement_optimizer.gui import MainWindow

        assert MainWindow is not None

    def test_import_exercise_tab(self):
        from movement_optimizer.gui import ExerciseTab

        assert ExerciseTab is not None

    def test_import_comparison_dialog(self):
        from movement_optimizer.gui import ComparisonDialog

        assert ComparisonDialog is not None

    def test_import_labelled_slider(self):
        from movement_optimizer.gui import LabelledSlider

        assert LabelledSlider is not None

    def test_import_parameter_sidebar(self):
        from movement_optimizer.gui import ParameterSidebar

        assert ParameterSidebar is not None

    def test_import_playback_controls(self):
        from movement_optimizer.gui import PlaybackControls

        assert PlaybackControls is not None

    def test_parameter_sidebar_lod_facades_defined(self):
        from movement_optimizer.gui import ParameterSidebar

        assert hasattr(ParameterSidebar, "connect_action_handlers")
        assert hasattr(ParameterSidebar, "get_comparison_context")
        assert hasattr(ParameterSidebar, "set_comparison_available")
        assert hasattr(ParameterSidebar, "set_cancellation_available")

    def test_playback_controls_lod_facades_defined(self):
        from movement_optimizer.gui import PlaybackControls

        assert hasattr(PlaybackControls, "connect_action_handlers")
        assert hasattr(PlaybackControls, "set_frame_position")
        assert hasattr(PlaybackControls, "speed_multiplier")
        assert hasattr(PlaybackControls, "set_speed_multiplier_text")


class TestGuiPackageStructure:
    """Verify the gui package sub-module structure."""

    def test_main_window_module(self):
        from movement_optimizer.gui import main_window

        assert hasattr(main_window, "MainWindow")

    def test_exercise_tab_module(self):
        from movement_optimizer.gui import exercise_tab

        assert hasattr(exercise_tab, "ExerciseTab")

    def test_comparison_dialog_module(self):
        from movement_optimizer.gui import comparison_dialog

        assert hasattr(comparison_dialog, "ComparisonDialog")

    def test_widgets_module(self):
        from movement_optimizer.gui import widgets

        assert hasattr(widgets, "LabelledSlider")
        assert hasattr(widgets, "ParameterSidebar")
        assert hasattr(widgets, "PlaybackControls")


class TestMainWindowAttributes:
    """Verify that MainWindow has expected class-level attributes."""

    def test_exercise_configs_defined(self):
        from movement_optimizer.gui import MainWindow

        assert hasattr(MainWindow, "EXERCISE_CONFIGS")
        configs = MainWindow.EXERCISE_CONFIGS
        assert len(configs) >= 4
        names = [c[0] for c in configs]
        assert "Bottoms Up Squat" in names
        assert "Deadlift" in names

    def test_signals_defined(self):
        from movement_optimizer.gui import MainWindow

        assert hasattr(MainWindow, "_sig_done")
        assert hasattr(MainWindow, "_sig_cancelled")
        assert hasattr(MainWindow, "_sig_error")
        assert hasattr(MainWindow, "_sig_progress")


class TestComparisonDialogAttributes:
    """Verify ComparisonDialog class-level attributes."""

    def test_trial_colors_defined(self):
        from movement_optimizer.gui import ComparisonDialog

        assert hasattr(ComparisonDialog, "TRIAL_COLORS")
        assert len(ComparisonDialog.TRIAL_COLORS) >= 5
