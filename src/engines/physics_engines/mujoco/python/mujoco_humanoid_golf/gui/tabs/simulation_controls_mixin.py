"""Mixin providing simulation playback/recording/screenshot handlers."""

from __future__ import annotations

import typing
from datetime import datetime
from pathlib import Path

from PyQt6 import QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger

if typing.TYPE_CHECKING:
    from ...sim_widget import MuJoCoSimWidget
    from ..advanced_gui import AdvancedGolfAnalysisWindow

logger = get_logger(__name__)


class _SimulationControlsMixin:
    """Mixin: play/pause, reset, record, screenshot, and export handlers."""

    # Attribute declarations for type checking (set by ControlsTab)
    sim_widget: MuJoCoSimWidget
    main_window: AdvancedGolfAnalysisWindow
    play_pause_btn: QtWidgets.QPushButton
    record_btn: QtWidgets.QPushButton
    recording_label: QtWidgets.QLabel

    def on_play_pause_toggled(self, checked: bool) -> None:
        """Toggle simulation between paused and running states."""
        if not (checked is not None):
            raise ValueError("checked must be provided")
        self.sim_widget.set_running(not checked)
        self.play_pause_btn.setText("Resume" if checked else "Pause")

        style = self.style()  # type: ignore[attr-defined]
        if style:
            icon = (
                QtWidgets.QStyle.StandardPixmap.SP_MediaPlay
                if checked
                else QtWidgets.QStyle.StandardPixmap.SP_MediaPause
            )
            self.play_pause_btn.setIcon(style.standardIcon(icon))

    def on_reset_clicked(self) -> None:
        """Reset the simulation to the initial state."""
        self.sim_widget.reset_state()
        self.play_pause_btn.setChecked(False)
        self.sim_widget.set_running(True)

    def on_record_toggled(self, checked: bool) -> None:
        """Start or stop recording simulation data."""
        if not (checked is not None):
            raise ValueError("checked must be provided")
        recorder = self.sim_widget.get_recorder()
        if checked:
            self.record_btn.setText("Stop Recording")
            if style := self.style():  # type: ignore[attr-defined]
                self.record_btn.setIcon(
                    style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop)
                )
            recorder.start_recording()
        else:
            self.record_btn.setText("Start Recording")
            if style := self.style():  # type: ignore[attr-defined]
                self.record_btn.setIcon(
                    style.standardIcon(
                        QtWidgets.QStyle.StandardPixmap.SP_DialogYesButton
                    )
                )
            recorder.stop_recording()

    def on_take_screenshot(self) -> None:
        """Save the current simulation view as a PNG screenshot."""
        pixmap = self.sim_widget.get_pixmap()
        if not pixmap or pixmap.isNull():
            return

        output_dir = Path("output/screenshots")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            output_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        pixmap.save(str(filename))
        logger.info("Screenshot saved: %s", filename)

        if self.main_window.statusBar():
            self.main_window.statusBar().showMessage(
                f"Screenshot saved: {filename}", 3000
            )

    def on_export_data(self) -> None:
        """Delegate data export to the main window handler."""
        if hasattr(self.main_window, "on_export_data"):
            self.main_window.on_export_data()
