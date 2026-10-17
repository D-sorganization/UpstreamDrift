"""Ground-reaction plot widget: forces under the feet over time (GCV-5, #11711).

Pure presentation over :mod:`biomechanics.ground_reaction_plot_model` and the
shared matplotlib renderer; the maths lives in the GCV-1 ground-reaction core.
Data arrives as a :class:`GroundReactionPlotSeries` or as a
``GET /api/analysis/ground-reaction`` payload saved as JSON.  Unavailable runs
say "unavailable" with the reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from src.shared.python.force_plates import (
    PlateTimeAlignment,
    ReadingOptions,
    TargetedPlacement,
)
from src.shared.python.force_plates.open_fp_format import PlateDeclarations
from src.shared.python.force_plates.open_fp_format.overlay_file import (
    overlay_swing_file,
)
from src.shared.python.plotting.renderers.ground_reaction import (
    UNQUALIFIED_BANNER,
    plot_ground_reaction,
)

UNAVAILABLE_TEXT = "unavailable"
_NOTHING_LOADED = f"{UNAVAILABLE_TEXT}: no ground reaction loaded"


@dataclass(frozen=True)
class PlateImportContext:
    """Facts a plate trial file cannot carry: set by the host before importing.

    ``declared`` is the plate uncertainty and geometry, ``placement`` the
    stance frame in the model's world, ``alignment`` the clock mapping.
    """

    declared: PlateDeclarations
    placement: TargetedPlacement
    alignment: PlateTimeAlignment
    options: ReadingOptions | None = None


class GroundReactionPlotWidget(QWidget):
    """Six-panel ground-reaction sheet with a JSON loader."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: GroundReactionPlotSeries | None = None
        self._model_series: GroundReactionPlotSeries | None = None
        self._plate_context: PlateImportContext | None = None
        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.open_button = QPushButton("Open Ground Reaction JSON...")
        self.plate_button = QPushButton("Import Plate Trial...")
        controls.addStretch(1)
        controls.addWidget(self.plate_button)
        controls.addWidget(self.open_button)
        root.addLayout(controls)
        self.banner_label = QLabel(UNQUALIFIED_BANNER)
        self.banner_label.setStyleSheet("color: #B45309; font-weight: bold;")
        self.banner_label.setVisible(False)
        root.addWidget(self.banner_label)
        self.status_label = QLabel(_NOTHING_LOADED)
        root.addWidget(self.status_label)
        self.figure = Figure(figsize=(9.0, 7.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        root.addWidget(self.canvas, 1)
        # noqa: gui-thread/ok - opens a modal file dialog; the JSON is small.
        self.open_button.clicked.connect(self._choose_file)
        # noqa: gui-thread/ok - modal file dialog; the swing pair is size-capped.
        self.plate_button.clicked.connect(self._choose_plate_trial)

    def set_series(self, series: GroundReactionPlotSeries | None) -> None:
        """Show ``series`` (``None`` clears to unavailable).

        The series is also the base that plate trials are overlaid on.
        """
        self._model_series = series
        self._series = series
        self._redraw()

    def set_plate_context(self, context: PlateImportContext | None) -> None:
        """Declare how plate trials are placed, aligned and qualified."""
        if context is not None and not isinstance(context, PlateImportContext):
            raise TypeError("context must be a PlateImportContext or None")
        self._plate_context = context

    def load_plate_trial(
        self,
        path: str | Path,
        declared: PlateDeclarations,
        placement: TargetedPlacement,
        alignment: PlateTimeAlignment,
    ) -> None:
        """Overlay the measured swing at ``path`` on the model ground reaction.

        Shows the "Unqualified measurement" banner when the trial carries no
        device qualification.  Replaces any earlier overlay.

        Raises:
            ValueError: if no model series is loaded or the trial is refused.
        """
        if self._model_series is None:
            raise ValueError("load a model ground reaction before importing a trial")
        self._series = overlay_swing_file(
            self._model_series,
            path,
            declared=declared,
            placement=placement,
            alignment=alignment,
        )
        self._redraw()

    def _choose_plate_trial(self) -> None:
        context = self._plate_context
        if context is None:
            self.status_label.setText(f"{UNAVAILABLE_TEXT}: no plate context declared")
            return
        name, _ = QFileDialog.getOpenFileName(
            self, "Import Plate Trial", "", "open-fp swing (*.json)"
        )
        if name:
            self._import_with_context(name, context)

    def _import_with_context(self, name: str, context: PlateImportContext) -> None:
        try:
            self.load_plate_trial(
                name, context.declared, context.placement, context.alignment
            )
        except (ValueError, TypeError, OSError) as exc:
            self.status_label.setText(f"{UNAVAILABLE_TEXT}: {exc}")

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
        self.banner_label.setVisible(
            series is not None and series.measured_qualified is False
        )
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
        self._model_series = None
        self._plate_context = None
