#!/usr/bin/env python3
"""Data Processor Widget - PyQt6 Interface for Data Processing.

A comprehensive data processing tool with:
- File loading (CSV, Excel, JSON, Parquet)
- Interactive data table with editing
- Column operations (add, remove, rename, transform)
- Filtering and querying
- Statistical analysis
- Curve fitting and visualization
- Data export

Can be used as a standalone widget, tab, or popup dialog.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ...data_processing.core import (
    AggregationType,
    DataProcessorEngine,
    FitType,
)
from .base_calculator_widget import BaseCalculatorWidget
from .mixins.data_processor_ops import DataProcessorOpsMixin

logger = logging.getLogger(__name__)


class DataProcessorWidget(DataProcessorOpsMixin, BaseCalculatorWidget):
    """PyQt6 widget for data processing and analysis.

    Provides a comprehensive interface for loading, manipulating,
    analyzing, and exporting data.
    """

    # Signals
    data_loaded = pyqtSignal(str)  # file_path
    data_modified = pyqtSignal()
    data_exported = pyqtSignal(str)  # file_path

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the data processor widget."""
        super().__init__(calculator_name="DataProcessor", parent=parent)

        self.engine = DataProcessorEngine()
        self.current_file: str | None = None
        self.current_page = 0
        self.total_pages = 1

        self.init_ui()
        self.setup_connections()
        self.setup_shortcuts()

        # State management setup
        QTimer.singleShot(0, self.setup_state_management)

    def setup_state_management(self) -> None:
        """Setup state management for splitters and copyable widgets."""
        for splitter in self.findChildren(QSplitter):
            self.register_splitter(splitter, "data_processor_splitter")

        for table in self.findChildren(QTableWidget):
            self.register_copyable_widget(table, "table")

    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Title
        title = QLabel("Data Processor")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Data table
        left_panel = self._create_data_panel()
        main_splitter.addWidget(left_panel)

        # Right panel - Tools and statistics
        right_panel = self._create_tools_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions (3:1)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        layout.addWidget(main_splitter)

        # Status bar
        self.status_label = QLabel("Ready - Load a file to begin")
        self.status_label.setStyleSheet("color: #666; font-size: 10px; padding: 4px;")
        layout.addWidget(self.status_label)

    def _create_toolbar(self) -> QToolBar:
        """Create the main toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)

        # File operations
        open_action = QAction("Open", self)
        open_action.setToolTip("Open a data file (Ctrl+O)")
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setToolTip("Save data (Ctrl+S)")
        save_action.triggered.connect(self.save_file)
        toolbar.addAction(save_action)

        export_action = QAction("Export", self)
        export_action.setToolTip("Export to different format")
        export_action.triggered.connect(self.export_file)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        # Edit operations
        undo_action = QAction("Undo", self)
        undo_action.setToolTip("Undo last operation (Ctrl+Z)")
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setToolTip("Redo operation (Ctrl+Y)")
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)

        reset_action = QAction("Reset", self)
        reset_action.setToolTip("Reset to original data")
        reset_action.triggered.connect(self.reset_data)
        toolbar.addAction(reset_action)

        toolbar.addSeparator()

        # View operations
        refresh_action = QAction("Refresh", self)
        refresh_action.setToolTip("Refresh statistics")
        refresh_action.triggered.connect(self.refresh_statistics)
        toolbar.addAction(refresh_action)

        return toolbar

    def _create_data_panel(self) -> QWidget:
        """Create the main data table panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File info
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(self.file_label)

        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        if header := self.data_table.horizontalHeader():
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(True)
        self.data_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.data_table.customContextMenuRequested.connect(
            self._show_table_context_menu
        )
        layout.addWidget(self.data_table)

        # Pagination
        pagination_layout = QHBoxLayout()
        self.page_label = QLabel("Page 1 of 1")
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addStretch()
        self.rows_per_page = QSpinBox()
        self.rows_per_page.setRange(10, 1000)
        self.rows_per_page.setValue(100)
        self.rows_per_page.setPrefix("Rows: ")
        self.rows_per_page.valueChanged.connect(self._update_table)
        pagination_layout.addWidget(self.rows_per_page)
        prev_btn = QPushButton("<")
        prev_btn.setFixedWidth(40)
        prev_btn.clicked.connect(self._prev_page)
        pagination_layout.addWidget(prev_btn)
        next_btn = QPushButton(">")
        next_btn.setFixedWidth(40)
        next_btn.clicked.connect(self._next_page)
        pagination_layout.addWidget(next_btn)
        layout.addLayout(pagination_layout)

        return panel

    def _create_tools_panel(self) -> QWidget:
        """Create the right panel with tools and statistics."""
        tabs = QTabWidget()
        tabs.addTab(self._create_statistics_tab(), "Stats")
        tabs.addTab(self._create_filter_tab(), "Filter")
        tabs.addTab(self._create_column_tab(), "Columns")
        tabs.addTab(self._create_fit_tab(), "Curve Fit")
        return tabs

    def _create_statistics_tab(self) -> QWidget:
        """Build the statistics tab with data summary and per-column stats."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Data Summary")
        form = QFormLayout()
        self.row_count_label = QLabel("0")
        form.addRow("Rows:", self.row_count_label)
        self.col_count_label = QLabel("0")
        form.addRow("Cols:", self.col_count_label)
        self.memory_label = QLabel("0 KB")
        form.addRow("Memory:", self.memory_label)
        self.null_count_label = QLabel("0")
        form.addRow("Nulls:", self.null_count_label)
        group.setLayout(form)
        layout.addWidget(group)
        col_group = QGroupBox("Column Stats")
        col_layout = QVBoxLayout()
        self.column_selector = QComboBox()
        self.column_selector.currentTextChanged.connect(self._update_column_stats)
        col_layout.addWidget(self.column_selector)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        col_layout.addWidget(self.stats_text)
        col_group.setLayout(col_layout)
        layout.addWidget(col_group)
        layout.addStretch()
        return tab

    def _create_filter_tab(self) -> QWidget:
        """Build the filter tab with quick-filter, query, and aggregation tools."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Quick Filter")
        form = QFormLayout()
        self.filter_column = QComboBox()
        form.addRow("Col:", self.filter_column)
        self.filter_operator = QComboBox()
        self.filter_operator.addItems(
            ["==", "!=", ">", "<", ">=", "<=", "contains", "in"]
        )
        form.addRow("Op:", self.filter_operator)
        self.filter_value = QLineEdit()
        form.addRow("Val:", self.filter_value)
        btn = QPushButton("Filter")
        btn.clicked.connect(self._apply_filter)
        form.addRow("", btn)
        group.setLayout(form)
        layout.addWidget(group)
        query_group = QGroupBox("Query")
        ql = QVBoxLayout()
        self.query_input = QLineEdit()
        ql.addWidget(self.query_input)
        qbtn = QPushButton("Run Query")
        qbtn.clicked.connect(self._execute_query)
        ql.addWidget(qbtn)
        query_group.setLayout(ql)
        layout.addWidget(query_group)
        agg_group = QGroupBox("Agg")
        al = QFormLayout()
        self.agg_group_by = QComboBox()
        self.agg_group_by.addItem("(None)")
        al.addRow("By:", self.agg_group_by)
        self.agg_column = QComboBox()
        al.addRow("Col:", self.agg_column)
        self.agg_type = QComboBox()
        self.agg_type.addItems([t.value for t in AggregationType])
        al.addRow("Type:", self.agg_type)
        abtn = QPushButton("Agg")
        abtn.clicked.connect(self._aggregate_data)
        al.addRow("", abtn)
        agg_group.setLayout(al)
        layout.addWidget(agg_group)
        layout.addStretch()
        return tab

    def _create_column_tab(self) -> QWidget:
        """Build the column management tab with add, transform, and rename tools."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        add = QGroupBox("Add")
        af = QFormLayout()
        self.new_col_name = QLineEdit()
        af.addRow("Name:", self.new_col_name)
        self.new_col_expr = QLineEdit()
        af.addRow("Expr:", self.new_col_expr)
        ab = QPushButton("Add")
        ab.clicked.connect(self._add_column)
        af.addRow("", ab)
        add.setLayout(af)
        layout.addWidget(add)
        trans = QGroupBox("Trans")
        tf = QFormLayout()
        self.transform_column = QComboBox()
        tf.addRow("Col:", self.transform_column)
        self.transform_type = QComboBox()
        self.transform_type.addItems(
            [
                "log",
                "exp",
                "sqrt",
                "abs",
                "normalize",
                "standardize",
                "round",
                "fillna",
                "dropna",
            ]
        )
        tf.addRow("T:", self.transform_type)
        self.transform_param = QDoubleSpinBox()
        tf.addRow("Val:", self.transform_param)
        tb = QPushButton("Trans")
        tb.clicked.connect(self._transform_column)
        tf.addRow("", tb)
        trans.setLayout(tf)
        layout.addWidget(trans)
        manage = QGroupBox("Manage")
        mf = QFormLayout()
        self.rename_column = QComboBox()
        mf.addRow("Col:", self.rename_column)
        self.rename_to = QLineEdit()
        mf.addRow("New:", self.rename_to)
        rb = QPushButton("Rename")
        rb.clicked.connect(self._rename_column)
        db = QPushButton("Drop")
        db.clicked.connect(self._drop_column)
        btn_row_layout = QHBoxLayout()
        btn_row_layout.addWidget(rb)
        btn_row_layout.addWidget(db)
        mf.addRow("", btn_row_layout)
        manage.setLayout(mf)
        layout.addWidget(manage)
        layout.addStretch()
        return tab

    def _create_fit_tab(self) -> QWidget:
        """Build the curve fitting tab with fit type selector and results display."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        fit = QGroupBox("Fit")
        ff = QFormLayout()
        self.fit_x_column = QComboBox()
        ff.addRow("X:", self.fit_x_column)
        self.fit_y_column = QComboBox()
        ff.addRow("Y:", self.fit_y_column)
        self.fit_type = QComboBox()
        self.fit_type.addItems([t.value for t in FitType])
        ff.addRow("Type:", self.fit_type)
        self.fit_degree = QSpinBox()
        ff.addRow("Deg:", self.fit_degree)
        fb = QPushButton("Fit")
        fb.clicked.connect(self._fit_curve)
        ff.addRow("", fb)
        fit.setLayout(ff)
        layout.addWidget(fit)
        self.fit_results_text = QTextEdit()
        self.fit_results_text.setReadOnly(True)
        layout.addWidget(self.fit_results_text)
        layout.addStretch()
        return tab

    def setup_connections(self) -> None:
        """Wire internal signals to their handler slots."""
        self.data_loaded.connect(self._on_data_loaded)
        self.data_modified.connect(self._on_data_modified)

    def setup_shortcuts(self) -> None:
        """Register keyboard shortcuts for file, undo, and redo operations."""
        from PyQt6.QtGui import QShortcut

        QShortcut(QKeySequence.StandardKey.Open, self, self.open_file)
        QShortcut(QKeySequence.StandardKey.Save, self, self.save_file)
        QShortcut(QKeySequence.StandardKey.Undo, self, self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, self.redo)

    def open_file(self) -> None:
        """Prompt the user to select a data file and load it into the engine."""
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "All (*.*)")
        if path:
            res = self.engine.load_file(path)
            if res.success:
                self.current_file = path
                self.data_loaded.emit(path)
                self._update_table()
                self._update_column_selectors()
                self.refresh_statistics()
                self._set_status(f"Loaded: {Path(path).name}")
            else:
                QMessageBox.warning(self, "Error", res.message)

    def save_file(self) -> None:
        """Save data to the current file, or prompt for export if no file loaded."""
        if not self.engine.has_data():
            return
        if self.current_file:
            res = self.engine.export_data(self.current_file)
            if res.success:
                self._set_status("Saved")
            else:
                QMessageBox.warning(self, "Error", res.message)
        else:
            self.export_file()

    def export_file(self) -> None:
        """Prompt for a destination path and export the current data as CSV."""
        path, _ = QFileDialog.getSaveFileName(self, "Export", "", "CSV (*.csv)")
        if path:
            res = self.engine.export_data(path)
            if res.success:
                self.data_exported.emit(path)
                self._set_status("Exported")

    def undo(self) -> None:
        """Undo the last data operation and refresh the view."""
        res = self.engine.undo()
        if res.success:
            self._update_table()
            self.refresh_statistics()
            self._set_status("Undo")

    def redo(self) -> None:
        """Re-apply the previously undone data operation and refresh the view."""
        res = self.engine.redo()
        if res.success:
            self._update_table()
            self.refresh_statistics()
            self._set_status("Redo")

    def reset_data(self) -> None:
        """Reset all data transformations back to the originally loaded state."""
        if self.engine.reset().success:
            self._update_table()
            self.refresh_statistics()
            self._set_status("Reset")

    def _update_table(self) -> None:
        if self.engine.data is None:
            self.data_table.clear()
            return
        df = self.engine.data
        rows = self.rows_per_page.value()
        self.total_pages = max(1, (len(df) + rows - 1) // rows)
        self.current_page = min(self.current_page, self.total_pages - 1)
        start = self.current_page * rows
        page = df.iloc[start : start + rows]
        self.data_table.setRowCount(len(page))
        self.data_table.setColumnCount(len(page.columns))
        self.data_table.setHorizontalHeaderLabels(list(page.columns))
        for r, (_, row) in enumerate(page.iterrows()):
            for c, v in enumerate(row):
                item = QTableWidgetItem(str(v) if v is not None and v == v else "")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.data_table.setItem(r, c, item)
        self.page_label.setText(
            f"Page {self.current_page + 1}/{self.total_pages} ({len(df)} rows)"
        )

    def _prev_page(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            self._update_table()

    def _next_page(self) -> None:
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._update_table()

    def _update_column_selectors(self) -> None:
        cols = self.engine.get_column_names()
        num_cols = self.engine.get_numeric_columns()
        for s in [
            self.column_selector,
            self.filter_column,
            self.transform_column,
            self.rename_column,
            self.agg_column,
        ]:
            curr = s.currentText()
            s.clear()
            s.addItems(cols)
            if curr in cols:
                s.setCurrentText(curr)
        self.agg_group_by.clear()
        self.agg_group_by.addItem("(None)")
        self.agg_group_by.addItems(cols)
        for s in [self.fit_x_column, self.fit_y_column]:
            curr = s.currentText()
            s.clear()
            s.addItems(num_cols)
            if curr in num_cols:
                s.setCurrentText(curr)

    def refresh_statistics(self) -> None:
        """Recalculate and display summary statistics for the loaded data."""
        if not self.engine.has_data():
            return
        stats = self.engine.get_statistics()
        self.row_count_label.setText(str(len(self.engine.data)))
        self.col_count_label.setText(str(len(self.engine.data.columns)))
        mem = self.engine.data.memory_usage(deep=True).sum()
        self.memory_label.setText(
            f"{mem / 1024:.1f} KB"
            if mem < 1024 * 1024
            else f"{mem / (1024 * 1024):.1f} MB"
        )
        self.null_count_label.setText(str(sum(s.null_count for s in stats.values())))
        self._update_column_stats()

    def _update_column_stats(self) -> None:
        col = self.column_selector.currentText()
        if not col:
            return
        cs = self.engine.get_statistics().get(col)
        if not cs:
            return
        h = f"<b>{cs.name}</b> ({cs.dtype})<br>Count: {cs.count}<br>Nulls: {cs.null_count}<br>Unique: {cs.unique_count}"
        if cs.mean is not None:
            h += f"<br><br>Mean: {cs.mean:.4f}<br>Std: {cs.std:.4f}<br>Min: {cs.min_val}<br>Max: {cs.max_val}"
        self.stats_text.setHtml(h)

    def _show_table_context_menu(self, pos: QPoint) -> None:
        menu = QMenu()
        action = menu.addAction("Copy Selected")
        if action is not None:
            action.triggered.connect(self._copy_selected)
        menu.exec(self.data_table.mapToGlobal(pos))

    def _copy_selected(self) -> None:
        items = self.data_table.selectedItems()
        if not items:
            return
        rs = sorted({i.row() for i in items})
        cs = sorted({i.column() for i in items})
        txt = "\n".join(
            [
                "\t".join(
                    [
                        (
                            item.text()
                            if (item := self.data_table.item(r, c)) is not None
                            else ""
                        )
                        for c in cs
                    ]
                )
                for r in rs
            ]
        )
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(txt)

    def _on_data_loaded(self, path: str) -> None:
        self.current_page = 0
        self._update_table()

    def _on_data_modified(self) -> None:
        pass

    def _set_status(self, message: str, success: bool = False) -> None:
        self.status_label.setText(message)
        QTimer.singleShot(5000, lambda: self.status_label.setText("Ready"))


__all__ = ["DataProcessorWidget"]
