"""Thin facade for the legacy motion-capture golf plotter."""

from __future__ import annotations

import sys

from motion_capture_plotter_data import MotionCapturePlotterDataMixin
from motion_capture_plotter_ui import MotionCapturePlotterUIMixin
from motion_capture_plotter_visualization import MotionCapturePlotterVisualizationMixin
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow


class MotionCapturePlotter(
    MotionCapturePlotterVisualizationMixin,
    MotionCapturePlotterDataMixin,
    MotionCapturePlotterUIMixin,
    QMainWindow,
):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Motion Capture Plotter - PyQt6")
        self.setGeometry(100, 100, 1400, 900)

        # Data storage - now supporting multiple data sources simultaneously
        self.swing_data: dict[str, object] = {}  # Motion capture data
        self.simscape_data: dict[str, object] = {}  # Simscape data
        self.current_swing: str | None = None
        self.current_frame = 0
        self.is_playing = False
        self.current_filter = "none"

        # Club parameters
        self.shaft_length = 0.9  # meters
        self.motion_scale = 1.0  # Use actual scale since we have real coordinates

        # Mouse interaction state
        self._last_pos: tuple[float, float] | None = None

        # Setup UI
        self.setup_ui()

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.next_frame)

        # Data source tracking
        self.current_data_source = "Motion Capture (Excel)"
        self.show_motion_capture = True
        self.show_simscape = False

        # Try to auto-load the Excel file if it exists
        self.auto_load_excel_file()


def main() -> None:
    """Launch the Motion Capture Plotter GUI application."""
    app = QApplication(sys.argv)
    window = MotionCapturePlotter()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
