"""Ground-reaction plot widget: forces under the feet over time (GCV-5, #11711).

Pure presentation over :mod:`biomechanics.ground_reaction_plot_model` and the
shared matplotlib renderer; the maths lives in the GCV-1 ground-reaction core.
Data arrives as a :class:`GroundReactionPlotSeries` or as a
``GET /api/analysis/ground-reaction`` payload saved as JSON.  Unavailable runs
say "unavailable" with the reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.biomechanics.ground_reaction_plot_model import (
    GroundReactionPlotSeries,
    series_from_payload,
)
from src.shared.python.plotting.renderers.ground_reaction import plot_ground_reaction

UNAVAILABLE_TEXT = "unavailable"
_NOTHING_LOADED = f"{UNAVAILABLE_TEXT}: no ground reaction loaded"


class GroundReactionPlotWidget(QWidget):
    """Six-panel ground-reaction sheet with a JSON loader."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: GroundReactionPlotSeries | None = None
        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.open_button = QPushButton("Open Ground Reaction JSON...")
        controls.addStretch(1)
        controls.addWidget(self.open_button)
        root.addLayout(controls)
        self.status_label = QLabel(_NOTHING_LOADED)
        root.addWidget(self.status_label)
        self.figure = Figure(figsize=(9.0, 7.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        root.addWidget(self.canvas, 1)
        # noqa: gui-thread/ok - opens a modal file dialog; the JSON is small.
        self.open_button.clicked.connect(self._choose_file)

    def set_series(self, series: GroundReactionPlotSeries | None) -> None:
        """Show ``series`` (``None`` clears to unavailable)."""
        self._series = series
        self._redraw()

    def load_json(self, path: str | Path) -> None:
        """Load an API payload JSON file.

        Raises:
            ValueError: if the file is not a ground-reaction payload.
        """
        data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("ground-reaction JSON must be an object")
        self.set_series(series_from_payload(data))

    def _choose_file(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Open Ground Reaction JSON", "", "JSON (*.json)"
        )
        if name:
            try:
                self.load_json(name)
            except (ValueError, OSError) as exc:
                self.status_label.setText(f"{UNAVAILABLE_TEXT}: {exc}")

    def _redraw(self) -> None:
        series = self._series
        if series is None:
            self.figure.clear()
            self.status_label.setText(_NOTHING_LOADED)
        else:
            plot_ground_reaction(self.figure, series)
            if series.available:
                feet = ", ".join(series.feet) or "none"
                self.status_label.setText(
                    f"Feet: {feet} | {len(series.time_s)} samples"
                )
            else:
                self.status_label.setText(f"{UNAVAILABLE_TEXT}: {series.reason}")
        self.canvas.draw_idle()

    def cleanup(self) -> None:
        """Release data references; safe to call repeatedly."""
        self._series = None
