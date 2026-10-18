"""PyQt6 Variability & Consistency tab (NSX-100, #12729).

A thin shell over :class:`~.variability_view.VariabilityView`: the request
controls on top, then the header, one figure and, per panel, an evidence badge,
a status and a table.  Nothing is computed or gated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.api.services import loop_nullspace_service as service
from src.shared.python.loop_nullspace.plot_model import NSXPlotModel
from src.tools.grip_wrench_plots.variability_controls import VariabilityControls
from src.tools.grip_wrench_plots.variability_draw import draw_sections
from src.tools.grip_wrench_plots.variability_view import (
    NOT_REQUESTED,
    SectionView,
    VariabilityView,
    variability_view,
)

__all__ = ["VariabilityTab"]

TAB_TITLE = "Variability & Consistency"


def _label(name: str) -> QLabel:
    label = QLabel("")
    label.setObjectName(name)
    label.setWordWrap(True)
    return label


class _SectionBox(QWidget):
    """Title, badge, status and table of one panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.title_label = _label("var_title")
        self.badge_label = _label("var_badge")
        self.status_label = _label("var_status")
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.notes_label = _label("var_notes")
        for widget in (self.title_label, self.badge_label, self.status_label):
            layout.addWidget(widget)
        layout.addWidget(self.table)
        layout.addWidget(self.notes_label)

    def show_section(self, sec: SectionView) -> None:
        self.title_label.setText(sec.title)
        self.badge_label.setText(sec.badge)
        self.status_label.setText(sec.status)
        self.notes_label.setText("\n".join(sec.notes))
        self.table.setColumnCount(len(sec.columns))
        self.table.setHorizontalHeaderLabels(list(sec.columns))
        self.table.setRowCount(len(sec.rows))
        for r, row in enumerate(sec.rows):
            for c, text in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(text))

    def text(self) -> str:
        cells = [
            self.table.item(r, c).text()
            for r in range(self.table.rowCount())
            for c in range(self.table.columnCount())
            if self.table.item(r, c) is not None
        ]
        heads = [
            self.table.horizontalHeaderItem(c).text()
            for c in range(self.table.columnCount())
            if self.table.horizontalHeaderItem(c) is not None
        ]
        labels = [
            self.title_label.text(),
            self.badge_label.text(),
            self.status_label.text(),
            self.notes_label.text(),
        ]
        return "\n".join([*labels, *heads, *cells])


class VariabilityTab(QWidget):
    """Controls plus the four sensitivity panels of one plot model."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        catalogue: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        cat = service.sensitivity_catalogue() if catalogue is None else catalogue
        outer = QVBoxLayout(self)
        self.controls = VariabilityControls(cat)
        outer.addWidget(self.controls)
        self.status_label = _label("var_run_status")
        self.header_label = _label("var_header")
        self.figure = Figure(figsize=(5.5, 6.0))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.boxes: dict[str, _SectionBox] = {}
        body = QWidget()
        inner = QVBoxLayout(body)
        for widget in (self.status_label, self.header_label, self.canvas):
            inner.addWidget(widget)
        self._sections_layout = inner
        self.non_claims_label = _label("var_non_claims")
        inner.addWidget(self.non_claims_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self._view: VariabilityView | None = None
        self.set_plot_model(None)

    def view(self) -> VariabilityView | None:
        return self._view

    def set_plot_model(self, model: NSXPlotModel | None) -> None:
        """Show ``model``'s sensitivity panels (``None`` clears to the empty state)."""
        if model is None:
            self._show(VariabilityView("no analysis run yet", (), (), ()))
        else:
            self._show(variability_view(model))

    def _box(self, name: str) -> _SectionBox:
        if name not in self.boxes:
            box = _SectionBox()
            self.boxes[name] = box
            self._sections_layout.insertWidget(self._sections_layout.count() - 1, box)
        return self.boxes[name]

    def _show(self, view: VariabilityView) -> None:
        self._view = view
        self.status_label.setText(view.status)
        self.header_label.setText("\n".join(view.header))
        self.non_claims_label.setText("\n".join(view.non_claims))
        draw_sections(self.figure, view.sections)
        self.canvas.draw_idle()
        for name, box in self.boxes.items():
            box.setVisible(name in {s.name for s in view.sections})
        for sec in view.sections:
            box = self._box(sec.name)
            box.setVisible(True)
            box.show_section(sec)

    def all_text(self) -> str:
        """Every text the tab shows (labels, tables, notes), for tests and copy."""
        parts = [
            self.status_label.text(),
            self.header_label.text(),
            self.non_claims_label.text(),
        ]
        parts += [b.text() for b in self.boxes.values() if not b.isHidden()]
        if self._view is not None and not self._view.sections:
            parts.append(self._view.status or NOT_REQUESTED)
        return "\n".join(parts)
