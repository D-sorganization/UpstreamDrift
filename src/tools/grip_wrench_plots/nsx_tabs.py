"""PyQt6 tabs that render an NSX plot model (NSX-43, #12575).

Every tab is a thin shell over a :class:`~.nsx_view.PanelView`: an evidence
badge (the ``display_decision`` outcome), the status, a figure drawn by
:func:`~.nsx_draw.draw_panel` and a table of the quantities with their state,
evidence and reason.  Nothing is computed or gated here.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.loop_nullspace.plot_model import NSXPlotModel
from src.tools.grip_wrench_plots.nsx_draw import draw_panel
from src.tools.grip_wrench_plots.nsx_view import (
    PANEL_TABS,
    PanelView,
    panel_views,
)

__all__ = ["NSXPanelTab", "NSXTabs"]

COLUMNS = ("Quantity", "Units", "State", "Evidence", "Reason", "Detail")
EMPTY_TEXT = "no analysis run yet"


class NSXPanelTab(QWidget):
    """One panel: badge, status, figure, quantity table, notes."""

    def __init__(self, panel: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if panel not in PANEL_TABS:
            raise ValueError(f"unknown NSX tab {panel!r}")
        self.panel = panel
        root = QVBoxLayout(self)
        self.badge_label = QLabel(EMPTY_TEXT)
        self.badge_label.setObjectName(f"nsx_badge_{panel}")
        self.badge_label.setWordWrap(True)
        self.status_label = QLabel("")
        self.status_label.setObjectName(f"nsx_status_{panel}")
        self.figure = Figure(figsize=(5.5, 5.0))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.notes_label = QLabel("")
        self.notes_label.setWordWrap(True)
        for widget in (self.badge_label, self.status_label):
            root.addWidget(widget)
        root.addWidget(self.canvas, 3)
        root.addWidget(self.table, 2)
        root.addWidget(self.notes_label)
        self._view: PanelView | None = None

    def set_view(self, view: PanelView | None) -> None:
        """Show ``view`` (``None`` clears to the empty state)."""
        self._view = view
        if view is None:
            self.badge_label.setText(EMPTY_TEXT)
            self.status_label.setText("")
            self.figure.clear()
            self.table.setRowCount(0)
            self.notes_label.setText("")
        else:
            self._show(view)
        self.canvas.draw_idle()

    def _show(self, view: PanelView) -> None:
        self.badge_label.setText(view.badge)
        self.status_label.setText(view.status)
        draw_panel(self.figure, view)
        self.table.setRowCount(len(view.rows))
        for r, row in enumerate(view.rows):
            cells = (
                row.name,
                row.units,
                row.state,
                row.evidence,
                row.reason,
                row.detail,
            )
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.notes_label.setText("\n".join(view.notes))

    def row_texts(self) -> list[tuple[str, ...]]:
        """The table as text, one tuple per row."""
        return [
            tuple(self._cell(r, c) for c in range(len(COLUMNS)))
            for r in range(self.table.rowCount())
        ]

    def _cell(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return "" if item is None else item.text()

    def view(self) -> PanelView | None:
        return self._view


class NSXTabs(QTabWidget):
    """Invariants, Envelopes, Hand Split and Diagnostics tabs of one model."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.panel_tabs: dict[str, NSXPanelTab] = {}
        for panel in PANEL_TABS:
            tab = NSXPanelTab(panel)
            self.panel_tabs[panel] = tab
            self.addTab(tab, panel.replace("_", " ").title())

    def set_plot_model(self, model: NSXPlotModel | None) -> None:
        """Render ``model`` in every tab (``None`` clears them).

        Raises:
            TypeError: ``model`` is neither ``None`` nor an ``NSXPlotModel``.
        """
        if model is not None and not isinstance(model, NSXPlotModel):
            raise TypeError("model must be an NSXPlotModel or None")
        views = None if model is None else panel_views(model)
        for panel, tab in self.panel_tabs.items():
            tab.set_view(None if views is None else views[panel])
