# Copyright (c) 2026 D-Sorganization. All rights reserved.
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QLabel, QPushButton, QTabWidget, QWidget

from movement_optimizer.gui.ui_builder import build_central_widget
from movement_optimizer.gui.widgets import ParameterSidebar, PlaybackControls


class TestUIBuilder:
    def test_build_central_widget(self, qapp):
        window = QWidget()
        exercise_configs = (
            ("Squat", "squat"),
            ("Deadlift", "deadlift"),
        )
        (
            central,
            sidebar,
            tabs,
            exercise_tabs,
            controls,
            status_label,
            _left_sidebar_toggle_btn,
            _right_sidebar_toggle_btn,
        ) = build_central_widget(window, exercise_configs)

        assert isinstance(central, QWidget)
        assert isinstance(sidebar, ParameterSidebar)
        assert isinstance(tabs, QTabWidget)
        assert len(exercise_tabs) == 2
        assert isinstance(controls, PlaybackControls)
        assert isinstance(status_label, QLabel)
        assert isinstance(_left_sidebar_toggle_btn, QPushButton)
        assert isinstance(_right_sidebar_toggle_btn, QPushButton)

        assert tabs.count() == 2
        assert tabs.tabText(0) == "  Squat  "
        assert tabs.tabText(1) == "  Deadlift  "

    def test_playback_controls_expose_frame_and_speed_helpers(self, qapp):
        window = QWidget()
        exercise_configs = (("Squat", "squat"),)
        (
            _central,
            _sidebar,
            _tabs,
            _exercise_tabs,
            controls,
            _status_label,
            _left_sidebar_toggle_btn,
            _right_sidebar_toggle_btn,
        ) = build_central_widget(window, exercise_configs)

        controls.speed_slider.setValue(15)
        controls.set_playback_status(4, 12, 1.5)

        assert controls.frame_label.text() == "Frame 4/12"
        assert controls.speed_label.text() == "1.5x"
        assert controls.speed_multiplier() == pytest.approx(1.5)

    def test_playback_buttons_have_visible_labels_and_accessible_metadata(self, qapp):
        controls = PlaybackControls()

        buttons = (
            controls.btn_rewind,
            controls.btn_back,
            controls.btn_play,
            controls.btn_fwd,
        )
        for button in buttons:
            assert _has_visible_word_label(button)
            assert button.accessibleName()
            assert button.accessibleDescription()

        controls.set_playing(True)

        assert controls.btn_play.text() == "Pause"
        assert controls.btn_play.accessibleName() == "Pause"
        assert controls.btn_play.accessibleDescription()

    def test_sidebar_action_buttons_have_accessible_metadata(self, qapp):
        window = QWidget()
        (
            _central,
            sidebar,
            _tabs,
            _exercise_tabs,
            _controls,
            _status_label,
            left_sidebar_toggle_btn,
            right_sidebar_toggle_btn,
        ) = build_central_widget(window, (("Squat", "squat"),))

        buttons = [
            left_sidebar_toggle_btn,
            right_sidebar_toggle_btn,
            sidebar.opt_btn,
            sidebar.both_btn,
            sidebar.cancel_btn,
            sidebar.export_btn,
            sidebar.reset_btn,
            sidebar.save_btn,
            sidebar.load_btn,
            sidebar.export_video_btn,
            sidebar.export_plots_btn,
            sidebar.export_excel_btn,
            sidebar.add_compare_btn,
            sidebar.compare_btn,
            sidebar.clear_compare_btn,
        ]
        for button in buttons:
            assert _has_visible_word_label(button)
            assert button.accessibleName()
            assert button.accessibleDescription()


def _has_visible_word_label(button: QPushButton) -> bool:
    return any(char.isalnum() for char in button.text())
