"""Workspace table widget for the Sidekick tools sidebar."""

from __future__ import annotations

import contextlib
from typing import Any

from . import design_tokens as theme
from .appearance import (
    DEFAULT_LIGHT_PANEL_APPEARANCE,
    PanelAppearance,
    is_panel_appearance,
    panel_qss,
)
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .qt_compat import QtCore, QtGui, QtWidgets, Signal
from .registry import WorkspaceRegistry, is_workspace_registry

QtGui_QStandardItemModel = QtGui.QStandardItemModel
QtGui_QStandardItem = QtGui.QStandardItem

WORKSPACE_TABLE_COLUMNS: tuple[str, ...] = ("Name", "Type", "Size", "Preview")


class WorkspaceTableWidget(QtWidgets.QWidget):
    """MATLAB-style workspace table bound to a :class:`WorkspaceRegistry`.

    Columns: Name / Type / Size / Preview. The widget subscribes to the
    registry so changes are reflected without manual ``refresh()`` calls
    (LOD: receives only the registry reference; no reach into a
    sidebar/theme tree).

    Args:
        registry: Workspace registry providing variables. Required.
        parent: Optional Qt parent.

    Raises:
        TypeError: If ``registry`` is missing or not a :class:`WorkspaceRegistry`.
    """

    inspect_requested = Signal(str)
    _COLUMNS = WORKSPACE_TABLE_COLUMNS

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if registry is None:
            raise TypeError("registry must be provided")
        if not is_workspace_registry(registry):
            raise TypeError("registry must be a WorkspaceRegistry")
        super().__init__(parent)
        self.setObjectName(theme.SIDEKICK_WORKSPACE_TAB_OBJECT_NAME)
        self._registry = registry
        self._appearance: PanelAppearance = DEFAULT_LIGHT_PANEL_APPEARANCE
        self._build_ui()
        self.apply_appearance(self._appearance)
        self._subscription = registry.subscribe(self._on_registry_event)
        self.refresh()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._heading = QtWidgets.QLabel("Workspace variables", self)
        self._heading.setObjectName("SidekickWorkspaceHeading")
        self._heading.setToolTip(
            "Variables shared across the Python REPL, Calculator, and chat."
        )
        layout.addWidget(self._heading)

        self._table = QtWidgets.QTableView(self)
        self._table.setObjectName("SidekickWorkspaceTable")
        self._table.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["workspace"]["summary"])
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSortingEnabled(True)
        self._table.doubleClicked.connect(self._on_double_clicked)

        self._model = QtGui_QStandardItemModel(0, len(self._COLUMNS), self)
        self._model.setHorizontalHeaderLabels(list(self._COLUMNS))
        self._table.setModel(self._model)
        layout.addWidget(self._table, stretch=3)

        # Empty-state guidance shown over the (otherwise blank) table so the
        # tab does not read as undifferentiated white space.
        self._empty_label = QtWidgets.QLabel(
            "No workspace variables yet.\n"
            "Run code in the Python REPL or Calculator — assigned variables "
            "appear here automatically.",
            self,
        )
        self._empty_label.setObjectName("SidekickWorkspaceEmptyState")
        self._empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        layout.addWidget(self._empty_label, stretch=1)

        self._history_label = QtWidgets.QLabel("Command History", self)
        self._history_label.setObjectName("SidekickWorkspaceHistoryLabel")
        layout.addWidget(self._history_label)
        self._history = QtWidgets.QListWidget(self)
        self._history.setObjectName("SidekickWorkspaceHistory")
        self._history.setToolTip("Recent commands run from this sidebar's REPL.")
        layout.addWidget(self._history, stretch=1)

    def apply_appearance(self, appearance: PanelAppearance) -> None:
        """Apply user-adjustable colours/border to the workspace surfaces."""
        if not is_panel_appearance(appearance):
            raise TypeError("appearance must be a PanelAppearance")
        self._appearance = appearance
        self.setStyleSheet(panel_qss(self.objectName(), appearance))

    def appearance(self) -> PanelAppearance:
        """Return the currently applied appearance."""
        return self._appearance

    def column_headers(self) -> tuple[str, ...]:
        """Return column header labels."""
        return tuple(self._COLUMNS)

    def row_data(self) -> list[tuple[str, str, str, str]]:
        """Return current rows as ``(name, type, size, preview)`` tuples."""
        rows: list[tuple[str, str, str, str]] = []
        for row in range(self._model.rowCount()):
            cells: list[str] = []
            for col in range(len(self._COLUMNS)):
                item = self._model.item(row, col)
                cells.append(item.text() if item else "")
            rows.append(tuple(cells))  # type: ignore[arg-type]
        return rows

    def sort_by_column(self, column: int, *, ascending: bool = True) -> None:
        """Sort the table by ``column`` (0=Name)."""
        if not 0 <= column < len(self._COLUMNS):
            raise ValueError("column out of range")
        order = (
            QtCore.Qt.SortOrder.AscendingOrder
            if ascending
            else QtCore.Qt.SortOrder.DescendingOrder
        )
        self._table.sortByColumn(column, order)

    def trigger_inspect(self, name: str) -> None:
        """Emit ``inspect_requested`` for ``name`` (test/programmatic hook)."""
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty str")
        self.inspect_requested.emit(name)

    def append_history(self, command: str) -> None:
        """Record a command line in the history pane."""
        if not isinstance(command, str) or not command.strip():
            return
        self._history.addItem(command.strip())

    def refresh(self) -> None:
        """Rebuild the table from the current registry contents."""
        self._model.removeRows(0, self._model.rowCount())
        for variable in self._registry.variables():
            self._append_row(variable)
        self._update_empty_state()

    def _update_empty_state(self) -> None:
        """Show the empty-state hint and hide the table when there are no rows."""
        is_empty = self._model.rowCount() == 0
        self._empty_label.setVisible(is_empty)
        self._table.setVisible(not is_empty)

    def _append_row(self, variable: Any) -> None:
        size_text = "" if variable.size is None else str(variable.size)
        items = [
            QtGui_QStandardItem(variable.name),
            QtGui_QStandardItem(variable.type_name),
            QtGui_QStandardItem(size_text),
            QtGui_QStandardItem(variable.preview or ""),
        ]
        for item in items:
            item.setEditable(False)
        self._model.appendRow(items)

    def _on_registry_event(self, event: str, name: str) -> None:
        # Re-render whole table: small N, simple, avoids row-index drift.
        del event, name
        self.refresh()

    def _on_double_clicked(self, index: Any) -> None:
        name_item = self._model.item(index.row(), 0)
        if name_item is not None:
            self.inspect_requested.emit(name_item.text())

    def deleteLater(self) -> None:
        with contextlib.suppress(Exception):
            self._subscription.unsubscribe()
        super().deleteLater()


def build_workspace_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the workspace variable inspector tab (MATLAB-style table)."""
    widget = WorkspaceTableWidget(registry=sidebar.registry, parent=sidebar)
    sidebar.register_workspace_table_widget(widget)
    return widget


def refresh_workspace_list(sidebar: Any) -> None:
    """Refresh the workspace table widget from the sidebar registry."""
    table_widget = getattr(sidebar, "workspace_table_widget", None)
    table = table_widget() if callable(table_widget) else None
    if table is not None:
        table.refresh()
        return
    workspace_list = sidebar.workspace_list_widget()
    if workspace_list is None:
        return
    workspace_list.clear()
    for variable in sidebar.registry.variables():
        details = [variable.summary]
        if variable.dtype:
            details.append(variable.dtype)
        if variable.size is not None:
            details.append(f"size={variable.size}")
        label = (
            f"{variable.name}: {variable.type_name} ({', '.join(details)}) "
            f"{variable.preview}"
        )
        workspace_list.addItem(label)
