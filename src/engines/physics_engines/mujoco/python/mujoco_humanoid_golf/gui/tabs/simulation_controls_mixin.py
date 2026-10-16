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
            self.recording_label.setText("Recording...")
        else:
            self.record_btn.setText("Start Recording")
            if style := self.style():  # type: ignore[attr-defined]
                self.record_btn.setIcon(
                    style.standardIcon(
                        QtWidgets.QStyle.StandardPixmap.SP_DialogYesButton
                    )
                )
            recorder.stop_recording()
            self.recording_label.setText(
                f"Stopped: {recorder.get_num_frames()} frames recorded"
            )

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

    def _dialog_parent(self) -> QtWidgets.QWidget | None:
        """Return this host as a dialog parent when it is a widget."""
        return self if isinstance(self, QtWidgets.QWidget) else None

    def on_export_video(self) -> None:
        """Render the recorded swing to a video file chosen by the user."""
        from ...video_export import export_recorded_swing_video

        parent = self._dialog_parent()
        recorder = self.sim_widget.get_recorder()
        if recorder.get_num_frames() == 0:
            QtWidgets.QMessageBox.warning(
                parent,
                "Export Video",
                "Nothing is recorded: press Start Recording, run the "
                "simulation, stop, then export.",
            )
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent, "Export Video", "swing.mp4", "MP4 Video (*.mp4)"
        )
        if not filename:
            return
        target = Path(filename)
        if not target.suffix:
            target = target.with_suffix(".mp4")
        try:
            out = export_recorded_swing_video(
                self.sim_widget.model, self.sim_widget.data, recorder, target
            )
        except (ValueError, RuntimeError, OSError) as exc:
            logger.exception("Video export failed: %s", exc)
            QtWidgets.QMessageBox.critical(
                parent, "Export Video", f"Could not export video: {exc}"
            )
            return
        QtWidgets.QMessageBox.information(
            parent, "Export Video", f"Exported video to: {out}"
        )

    def on_export_data(self) -> None:
        """Delegate data export to the main window handler."""
        if hasattr(self.main_window, "on_export_data"):
            self.main_window.on_export_data()
