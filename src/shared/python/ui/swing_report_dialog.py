"""PyQt dialog for choosing and exporting swing report plots (#12329).

Consumes only the validated report through ``ReportViewModel``; it imports no
engine code. Withheld plots are shown disabled with their reason and a
"Not Validated" label; advertised plots show their uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable
import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.reporting.swing_report import SwingReport
from src.shared.python.reporting.swing_report.view_model import (
    MODE_CUSTOM,
    MODE_FULL,
    PlotRow,
    ReportViewModel,
)

logger = logging.getLogger(__name__)

DirectoryChooser = Callable[[QWidget], str]


def _choose_directory(parent: QWidget) -> str:
    return QFileDialog.getExistingDirectory(parent, "Export Swing Report")


class SwingReportDialog(QDialog):
    """Full/Custom plot selection over a validated ``SwingReport``."""

    exported = pyqtSignal(str)

    def __init__(
        self,
        report: SwingReport,
        parent: QWidget | None = None,
        choose_directory: DirectoryChooser = _choose_directory,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Swing Report")
        self._model = ReportViewModel(report)
        self._choose_directory = choose_directory
        self._boxes: dict[str, QCheckBox] = {}
        self._reason_labels: dict[str, QLabel] = {}
        self._build()
        self._sync()

    # -- construction ----------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        mode_row = QHBoxLayout()
        self.full_radio = QRadioButton("Full Report")
        self.custom_radio = QRadioButton("Custom Selection")
        self.full_radio.setChecked(True)
        for radio in (self.full_radio, self.custom_radio):
            mode_row.addWidget(radio)
            radio.toggled.connect(self._on_mode_changed)
        layout.addLayout(mode_row)
        for row in self._model.rows:
            layout.addLayout(self._build_row(row))
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_selected)
        layout.addWidget(self.export_button)

    def _build_row(self, row: PlotRow) -> QVBoxLayout:
        box = QCheckBox(f"{row.title} [{row.status_label}]")
        box.setEnabled(row.enabled)
        box.toggled.connect(self._on_box_toggled)
        self._boxes[row.plot_id] = box
        out = QVBoxLayout()
        out.addWidget(box)
        if row.detail:
            label = QLabel(row.detail)
            label.setWordWrap(True)
            self._reason_labels[row.plot_id] = label
            out.addWidget(label)
            box.setToolTip(row.detail)
        return out

    # -- state sync ------------------------------------------------------
    def _sync(self) -> None:
        """Push the model state into the widgets without feedback loops."""
        full = self._model.mode == MODE_FULL
        chosen = set(self._model.selected_ids)
        for row in self._model.rows:
            box = self._boxes[row.plot_id]
            box.blockSignals(True)
            box.setChecked(row.plot_id in chosen)
            box.setEnabled(row.enabled and not full)
            box.blockSignals(False)
        self.export_button.setEnabled(self._model.can_export())

    def _on_mode_changed(self, _checked: bool) -> None:
        mode = MODE_CUSTOM if self.custom_radio.isChecked() else MODE_FULL
        self._model.set_mode(mode)
        self._sync()

    def _on_box_toggled(self, _checked: bool) -> None:
        if self._model.mode != MODE_CUSTOM:
            return
        ids = [pid for pid, box in self._boxes.items() if box.isChecked()]
        self._model.set_custom_selection(ids)
        self.export_button.setEnabled(self._model.can_export())

    # -- public API ------------------------------------------------------
    @property
    def model(self) -> ReportViewModel:
        return self._model

    def selected_plot_ids(self) -> tuple[str, ...]:
        return self._model.selected_ids

    def set_custom_selection(self, plot_ids: list[str]) -> None:
        """Switch to Custom mode with ``plot_ids`` selected."""
        self._model.set_custom_selection(plot_ids)
        self.custom_radio.setChecked(True)
        self._model.set_mode(MODE_CUSTOM)
        self._sync()

    def plot_checkbox(self, plot_id: str) -> QCheckBox:
        return self._boxes[plot_id]

    def reason_label(self, plot_id: str) -> QLabel | None:
        return self._reason_labels.get(plot_id)

    def export_selected(self) -> bool:
        """Ask for a directory and export the selection; True on success."""
        directory = self._choose_directory(self)
        if not directory:
            return False
        try:
            self._model.export(directory)
        except (ValueError, OSError) as exc:
            logger.warning("Swing report export failed: %s", exc)
            QMessageBox.warning(self, "Export Failed", str(exc))
            return False
        self.exported.emit(directory)
        return True
