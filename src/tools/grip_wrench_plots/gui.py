"""Grip wrench plot widget: hands' loading on the club over time (GCV-10, #11716).

Pure presentation over :mod:`biomechanics.grip_plot_model` and the shared
matplotlib renderer; the maths lives in the grip wrench core.  Data arrives as a
:class:`GripPlotSeries`, from a ``GET /api/analysis/grip-wrench`` payload saved
as JSON, or from the video export's ``*_grip_wrench.json``.  The split method is
always shown; unavailable runs say "unavailable" with the reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.biomechanics.grip_plot_model import (
    GripPlotSeries,
    series_from_payload,
)
from src.shared.python.plotting.renderers.grip_wrench import (
    COUPLE_FRAMES,
    plot_grip_wrench,
)

UNAVAILABLE_TEXT = "unavailable"


class GripWrenchPlotWidget(QWidget):
    """Four-panel plot with a couple-frame toggle and a JSON loader."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: GripPlotSeries | None = None
        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.frame_combo = QComboBox()
        self.frame_combo.addItems(list(COUPLE_FRAMES))
        self.frame_combo.setToolTip("Frame of the equivalent couple panel")
        self.open_button = QPushButton("Open Grip Wrench JSON...")
        controls.addWidget(QLabel("Couple frame"))
        controls.addWidget(self.frame_combo)
        controls.addStretch(1)
        controls.addWidget(self.open_button)
        root.addLayout(controls)
        self.status_label = QLabel(f"{UNAVAILABLE_TEXT}: no grip wrench loaded")
        root.addWidget(self.status_label)
        self.figure = Figure(figsize=(5.5, 7.0))
        self.canvas = FigureCanvasQTAgg(self.figure)
        root.addWidget(self.canvas, 1)
        self.frame_combo.currentTextChanged.connect(self._redraw)
        # noqa: gui-thread/ok - opens a modal file dialog; the JSON is small.
        self.open_button.clicked.connect(self._choose_file)

    def set_series(self, series: GripPlotSeries | None) -> None:
        """Show ``series`` (``None`` clears to unavailable)."""
        self._series = series
        self._redraw()

    def load_json(self, path: str | Path) -> None:
        """Load an API payload or export JSON file.

        Raises:
            ValueError: if the file is not a grip wrench payload.
        """
        data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("grip wrench JSON must be an object")
        self.set_series(series_from_payload(data))

    def _choose_file(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Open Grip Wrench JSON", "", "JSON (*.json)"
        )
        if name:
            try:
                self.load_json(name)
            except (ValueError, OSError) as exc:
                self.status_label.setText(f"{UNAVAILABLE_TEXT}: {exc}")

    def _redraw(self, *_: Any) -> None:
        series = self._series
        if series is None:
            self.figure.clear()
            self.status_label.setText(f"{UNAVAILABLE_TEXT}: no grip wrench loaded")
        else:
            plot_grip_wrench(
                self.figure, series, couple_frame=self.frame_combo.currentText()
            )
            if series.available:
                self.status_label.setText(
                    f"Split method: {series.split_method} | {len(series.time_s)} samples"
                )
            else:
                self.status_label.setText(f"{UNAVAILABLE_TEXT}: {series.reason}")
        self.canvas.draw_idle()

    def cleanup(self) -> None:
        """Release data references; safe to call repeatedly."""
        self._series = None
