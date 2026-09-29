# ruff: noqa: E501
# ruff: noqa
# mypy: ignore-errors
# TRACKED_TASK: see #2310 — architecture debt extraction schedule
# UPDATE: Decomposed into ui/ package.

"""
PyQt6 GUI for Two-Stage PSA System Analysis.

This GUI provides interactive visualization and analysis of PSA system
performance, including sensitivity analysis and O2 safety calculations.
"""

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QDoubleValidator, QFont, QPixmap
from src.shared.python.theme.integration import ThemedWindowMixin
import logging
import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .psa_model import (
    DEFAULT_COMPONENTS,
    ComponentData,
    PSAModel,
    PSAResults,
    calculate_o2_safety_analysis,
    calculate_sensitivity,
    get_flammability_status,
)

# Expose components for backward compatibility
from .ui import (
    InputPanel,
    MplCanvas,
    PFDWidget,
    PSAMainWindow,
    ResultsPanel,
    SensitivityPlotWidget,
    create_slider,
)

__all__ = [
    "InputPanel",
    "MplCanvas",
    "PFDWidget",
    "PSAMainWindow",
    "ResultsPanel",
    "SensitivityPlotWidget",
    "create_slider",
    "main",
]

_logger = logging.getLogger(__name__)


def create_slider(  # noqa: F811
    min_value: int,
    max_value: int,
    default_value: int,
    orientation: Qt.Orientation,  # noqa: F821
    value_changed_callback: Callable[[int], None] | None = None,
) -> QSlider:  # noqa: F821
    slider = QSlider(orientation)  # noqa: F821
    slider.setRange(min_value, max_value)
    slider.setValue(default_value)
    if value_changed_callback:
        slider.valueChanged.connect(value_changed_callback)
    return slider


matplotlib.use("QtAgg")  # noqa: F821


class MplCanvas(FigureCanvas):  # noqa: F811, F821
    """Matplotlib canvas widget for embedding in PyQt6."""

    def __init__(
        self,
        parent: QWidget | None = None,
        width: float = 8,
        height: float = 6,  # noqa: F821
    ) -> None:
        if width is None:
            raise ValueError("width must be provided")
        self.fig = Figure(figsize=(width, height), dpi=100)  # noqa: F821
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # noqa: F821


class InputPanel(QWidget):  # noqa: F811, F821
    """Panel for PSA model input parameters."""

    input_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: F821
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)  # noqa: F821

        # Operating Parameters Group
        op_group = QGroupBox("Operating Parameters")  # noqa: F821
        op_layout = QGridLayout()  # noqa: F821

        # Total Feed
        op_layout.addWidget(QLabel("Total Feed (SCFM):"), 0, 0)  # noqa: F821
        self.feed_input = QLineEdit("1100")  # noqa: F821
        self.feed_input.setValidator(QDoubleValidator(0, 100000, 2))  # noqa: F821
        op_layout.addWidget(self.feed_input, 0, 1)

        # S2 Tail Recycle
        op_layout.addWidget(QLabel("S2 Tail Recycle (%):"), 1, 0)  # noqa: F821
        self.s2_recycle_slider = create_slider(
            min_value=0,
            max_value=100,
            default_value=100,
            orientation=Qt.Orientation.Horizontal,  # noqa: F821
            value_changed_callback=lambda v: self.s2_recycle_label.setText(f"{v}%"),
        )
        self.s2_recycle_label = QLabel("100%")  # noqa: F821
        op_layout.addWidget(self.s2_recycle_slider, 1, 1)
        op_layout.addWidget(self.s2_recycle_label, 1, 2)

        # Product Recycle
        op_layout.addWidget(QLabel("Product Recycle (%):"), 2, 0)  # noqa: F821
        self.prod_recycle_slider = create_slider(
            min_value=0,
            max_value=100,
            default_value=0,
            orientation=Qt.Orientation.Horizontal,  # noqa: F821
            value_changed_callback=lambda v: self.prod_recycle_label.setText(f"{v}%"),
        )
        self.prod_recycle_label = QLabel("0%")  # noqa: F821
        op_layout.addWidget(self.prod_recycle_slider, 2, 1)
        op_layout.addWidget(self.prod_recycle_label, 2, 2)

        op_group.setLayout(op_layout)
        layout.addWidget(op_group)

        # Component Data Group
        comp_group = QGroupBox("Component Data (Feed % | S1 Removal % | S2 Removal %)")  # noqa: F821
        comp_layout = QVBoxLayout()  # noqa: F821

        self.component_table = QTableWidget(7, 4)  # noqa: F821
        self.component_table.setHorizontalHeaderLabels(
            ["Component", "Feed %", "S1 Removal %", "S2 Removal %"]
        )
        header = self.component_table.verticalHeader()
        if header is not None:
            header.setVisible(False)

        for i, comp in enumerate(DEFAULT_COMPONENTS):  # noqa: F821
            self.component_table.setItem(i, 0, QTableWidgetItem(comp["name"]))  # noqa: F821
            self.component_table.setItem(i, 1, QTableWidgetItem(str(comp["feed_pct"])))  # noqa: F821
            self.component_table.setItem(
                i,
                2,
                QTableWidgetItem(str(comp["stage1_removal_pct"])),  # noqa: F821
            )
            self.component_table.setItem(
                i,
                3,
                QTableWidgetItem(str(comp["stage2_removal_pct"])),  # noqa: F821
            )

        self.component_table.resizeColumnsToContents()
        comp_layout.addWidget(self.component_table)

        comp_group.setLayout(comp_layout)
        layout.addWidget(comp_group)

        # Reset Button
        self.reset_button = QPushButton("Reset to Defaults")  # noqa: F821
        layout.addWidget(self.reset_button)
        self.reset_button.clicked.connect(self._reset_defaults)

        layout.addStretch()

        # Connect input changes to the panel-level change contract.
        self.feed_input.textChanged.connect(self._on_input_change)
        self.s2_recycle_slider.valueChanged.connect(self._on_input_change)
        self.prod_recycle_slider.valueChanged.connect(self._on_input_change)
        self.component_table.cellChanged.connect(self._on_input_change)

    def _on_input_change(self) -> None:
        """Signal that inputs have changed - emitted for auto-calculate."""
        self.input_changed.emit()

    def _reset_defaults(self) -> None:
        """Reset all inputs to default values."""
        self.feed_input.setText("1100")
        self.s2_recycle_slider.setValue(100)
        self.prod_recycle_slider.setValue(0)

        for i, comp in enumerate(DEFAULT_COMPONENTS):  # noqa: F821
            self.component_table.setItem(i, 1, QTableWidgetItem(str(comp["feed_pct"])))  # noqa: F821
            self.component_table.setItem(
                i,
                2,
                QTableWidgetItem(str(comp["stage1_removal_pct"])),  # noqa: F821
            )
            self.component_table.setItem(
                i,
                3,
                QTableWidgetItem(str(comp["stage2_removal_pct"])),  # noqa: F821
            )

    def get_parameters(
        self,
    ) -> tuple[float, float, float, list[ComponentData]]:  # noqa: F821
        """Get current input parameters."""
        total_feed = float(self.feed_input.text())
        s2_recycle = self.s2_recycle_slider.value() / 100.0
        prod_recycle = self.prod_recycle_slider.value() / 100.0

        components: list[ComponentData] = []  # noqa: F821
        for i in range(7):
            name_item = self.component_table.item(i, 0)
            feed_item = self.component_table.item(i, 1)
            s1_item = self.component_table.item(i, 2)
            s2_item = self.component_table.item(i, 3)

            if name_item and feed_item and s1_item and s2_item:
                components.append(
                    {
                        "name": name_item.text(),
                        "feed_pct": float(feed_item.text()),
                        "stage1_removal_pct": float(s1_item.text()),
                        "stage2_removal_pct": float(s2_item.text()),
                    }
                )

        return total_feed, s2_recycle, prod_recycle, components


class ResultsPanel(QWidget):  # noqa: F811, F821
    """Panel for displaying calculation results."""

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: F821
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)  # noqa: F821

        # Key Metrics Group
        metrics_group = QGroupBox("Key Performance Metrics")  # noqa: F821
        metrics_layout = QGridLayout()  # noqa: F821

        self.h2_recovery_label = QLabel("--")  # noqa: F821
        self.h2_purity_label = QLabel("--")  # noqa: F821
        self.net_product_label = QLabel("--")  # noqa: F821
        self.exhaust_label = QLabel("--")  # noqa: F821
        self.mass_balance_label = QLabel("--")  # noqa: F821

        font = QFont()  # noqa: F821
        font.setPointSize(12)
        font.setBold(True)

        for label in [
            self.h2_recovery_label,
            self.h2_purity_label,
            self.net_product_label,
        ]:
            label.setFont(font)

        metrics_layout.addWidget(QLabel("H2 Recovery:"), 0, 0)  # noqa: F821
        metrics_layout.addWidget(self.h2_recovery_label, 0, 1)
        metrics_layout.addWidget(QLabel("H2 Purity:"), 1, 0)  # noqa: F821
        metrics_layout.addWidget(self.h2_purity_label, 1, 1)
        metrics_layout.addWidget(QLabel("Net Product:"), 2, 0)  # noqa: F821
        metrics_layout.addWidget(self.net_product_label, 2, 1)
        metrics_layout.addWidget(QLabel("Exhaust:"), 3, 0)  # noqa: F821
        metrics_layout.addWidget(self.exhaust_label, 3, 1)
        metrics_layout.addWidget(QLabel("Mass Balance:"), 4, 0)  # noqa: F821
        metrics_layout.addWidget(self.mass_balance_label, 4, 1)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # Safety Metrics Group
        safety_group = QGroupBox("Safety Metrics")  # noqa: F821
        safety_layout = QGridLayout()  # noqa: F821

        self.s2_tail_h2_label = QLabel("--")  # noqa: F821
        self.s2_tail_o2_label = QLabel("--")  # noqa: F821
        self.flammability_label = QLabel("--")  # noqa: F821
        self.flammability_label.setStyleSheet("font-weight: bold;")

        safety_layout.addWidget(QLabel("S2 Tail H2:"), 0, 0)  # noqa: F821
        safety_layout.addWidget(self.s2_tail_h2_label, 0, 1)
        safety_layout.addWidget(QLabel("S2 Tail O2:"), 1, 0)  # noqa: F821
        safety_layout.addWidget(self.s2_tail_o2_label, 1, 1)
        safety_layout.addWidget(QLabel("Status:"), 2, 0)  # noqa: F821
        safety_layout.addWidget(self.flammability_label, 2, 1)

        safety_group.setLayout(safety_layout)
        layout.addWidget(safety_group)

        # Stream Flows Table
        flows_group = QGroupBox("Stream Flows (SCFM)")  # noqa: F821
        flows_layout = QVBoxLayout()  # noqa: F821

        self.flows_table = QTableWidget()  # noqa: F821
        self.flows_table.setColumnCount(9)
        self.flows_table.setHorizontalHeaderLabels(
            [
                "Component",
                "Fresh Feed",
                "Mixed Feed",
                "Exhaust",
                "Interstage",
                "S2 Tail",
                "S2 Tail Recy",
                "Gross Prod",
                "Net Prod",
            ]
        )
        flows_layout.addWidget(self.flows_table)

        flows_group.setLayout(flows_layout)
        layout.addWidget(flows_group)

        # Compositions Table
        comp_group = QGroupBox("Stream Compositions (%)")  # noqa: F821
        comp_layout = QVBoxLayout()  # noqa: F821

        self.comp_table = QTableWidget()  # noqa: F821
        self.comp_table.setColumnCount(7)
        self.comp_table.setHorizontalHeaderLabels(
            [
                "Component",
                "Fresh Feed",
                "Mixed Feed",
                "Exhaust",
                "Interstage",
                "S2 Tail",
                "Net Prod",
            ]
        )
        comp_layout.addWidget(self.comp_table)

        comp_group.setLayout(comp_layout)
        layout.addWidget(comp_group)

    def update_results(self, results: PSAResults) -> None:  # noqa: F821
        """Update display with calculation results."""
        if results is None:
            raise ValueError("results must be provided")
        self._update_key_metrics(results)
        self._update_safety_metrics(results)
        self._update_flows_table(results)
        self._update_compositions_table(results)

    def _update_key_metrics(self, results: PSAResults) -> None:  # noqa: F821
        """Update key performance metric labels.

        ``results`` is validated once by :meth:`update_results`; this private
        helper trusts that boundary check.
        """
        self.h2_recovery_label.setText(f"{results.h2_recovery_pct:.2f}%")
        self.h2_purity_label.setText(f"{results.h2_purity_pct:.5f}%")
        self.net_product_label.setText(f"{results.total_net_product_scfm:.2f} SCFM")
        self.exhaust_label.setText(f"{results.total_exhaust_scfm:.2f} SCFM")
        self.mass_balance_label.setText(f"{results.mass_balance_error:.2e}")

    def _update_safety_metrics(self, results: PSAResults) -> None:  # noqa: F821
        """Update safety/flammability metric labels and styling.

        ``results`` is validated once by :meth:`update_results`.
        """
        self.s2_tail_h2_label.setText(f"{results.s2_tail_h2_pct:.2f}%")
        self.s2_tail_o2_label.setText(f"{results.s2_tail_o2_pct:.2f}%")

        status = get_flammability_status(results.s2_tail_h2_pct, results.s2_tail_o2_pct)  # noqa: F821
        self.flammability_label.setText(status)

        if "CRITICAL" in status or "FLAMMABLE" in status or "DANGEROUS" in status:
            self.flammability_label.setStyleSheet(
                "font-weight: bold; color: red; background-color: #ffcccc;"
            )
        elif "Caution" in status:
            self.flammability_label.setStyleSheet(
                "font-weight: bold; color: orange; background-color: #ffffcc;"
            )
        else:
            self.flammability_label.setStyleSheet(
                "font-weight: bold; color: green; background-color: #ccffcc;"
            )

    def _update_flows_table(self, results: PSAResults) -> None:  # noqa: F821
        """Populate the flows table with component flow data and totals.

        ``results`` is validated once by :meth:`update_results`.
        """
        n_comp = len(results.component_names)
        self.flows_table.setRowCount(n_comp + 1)

        flow_columns = [
            results.flows.fresh_feed,
            results.flows.mixed_feed,
            results.flows.exhaust,
            results.flows.interstage,
            results.flows.s2_tail,
            results.flows.s2_tail_recycle,
            results.flows.gross_product,
            results.flows.net_product,
        ]

        for i, name in enumerate(results.component_names):
            self.flows_table.setItem(i, 0, QTableWidgetItem(name))  # noqa: F821
            for col_idx, col_data in enumerate(flow_columns):
                self.flows_table.setItem(
                    i,
                    col_idx + 1,
                    QTableWidgetItem(f"{col_data[i]:.4f}"),  # noqa: F821
                )

        # Totals row
        self.flows_table.setItem(n_comp, 0, QTableWidgetItem("TOTAL"))  # noqa: F821
        totals = [
            results.total_feed_scfm,
            np.sum(results.flows.mixed_feed),  # noqa: F821
            results.total_exhaust_scfm,
            np.sum(results.flows.interstage),  # noqa: F821
            np.sum(results.flows.s2_tail),  # noqa: F821
            np.sum(results.flows.s2_tail_recycle),  # noqa: F821
            np.sum(results.flows.gross_product),  # noqa: F821
            results.total_net_product_scfm,
        ]
        for col_idx, total in enumerate(totals):
            self.flows_table.setItem(
                n_comp,
                col_idx + 1,
                QTableWidgetItem(f"{total:.2f}"),  # noqa: F821
            )

        self.flows_table.resizeColumnsToContents()

    def _update_compositions_table(self, results: PSAResults) -> None:  # noqa: F821
        """Populate the compositions table with component percentage data.

        ``results`` is validated once by :meth:`update_results`.
        """
        n_comp = len(results.component_names)
        self.comp_table.setRowCount(n_comp + 1)

        comp_columns = [
            results.compositions.fresh_feed,
            results.compositions.mixed_feed,
            results.compositions.exhaust,
            results.compositions.interstage,
            results.compositions.s2_tail,
            results.compositions.net_product,
        ]

        for i, name in enumerate(results.component_names):
            self.comp_table.setItem(i, 0, QTableWidgetItem(name))  # noqa: F821
            for col_idx, col_data in enumerate(comp_columns):
                self.comp_table.setItem(
                    i,
                    col_idx + 1,
                    QTableWidgetItem(f"{col_data[i]:.4f}"),  # noqa: F821
                )

        # Totals row
        self.comp_table.setItem(n_comp, 0, QTableWidgetItem("TOTAL"))  # noqa: F821
        for j in range(1, 7):
            self.comp_table.setItem(n_comp, j, QTableWidgetItem("100.00"))  # noqa: F821

        self.comp_table.resizeColumnsToContents()


class SensitivityPlotWidget(QWidget):  # noqa: F811, F821
    """Widget for sensitivity analysis plots."""

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: F821
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)  # noqa: F821

        # Controls
        controls_layout = QHBoxLayout()  # noqa: F821

        controls_layout.addWidget(QLabel("Plot Type:"))  # noqa: F821
        self.plot_type_combo = QComboBox()  # noqa: F821
        self.plot_type_combo.addItems(
            [
                "H2 Recovery vs Recycle",
                "Net Product vs Recycle",
                "O2 Safety Analysis",
                "3D Recovery Surface",
                "Contour Map",
            ]
        )
        controls_layout.addWidget(self.plot_type_combo)

        # Line/Marker options
        controls_layout.addWidget(QLabel("  "))  # noqa: F821
        self.show_lines_check = QCheckBox("Lines")  # noqa: F821
        self.show_lines_check.setChecked(True)
        controls_layout.addWidget(self.show_lines_check)

        self.show_markers_check = QCheckBox("Markers")  # noqa: F821
        self.show_markers_check.setChecked(False)
        controls_layout.addWidget(self.show_markers_check)

        # Number of points
        controls_layout.addWidget(QLabel("Points:"))  # noqa: F821
        self.num_points_spin = QSpinBox()  # noqa: F821
        self.num_points_spin.setRange(11, 101)
        self.num_points_spin.setValue(51)
        self.num_points_spin.setSingleStep(10)
        controls_layout.addWidget(self.num_points_spin)

        self.update_button = QPushButton("Update Plot")  # noqa: F821
        controls_layout.addWidget(self.update_button)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Canvas
        self.canvas = MplCanvas(self, width=10, height=7)
        self.toolbar = NavigationToolbar(self.canvas, self)  # noqa: F821

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Connect signals
        self.update_button.clicked.connect(self._update_plot)
        self.plot_type_combo.currentIndexChanged.connect(self._update_plot)
        self.show_lines_check.stateChanged.connect(self._update_plot)
        self.show_markers_check.stateChanged.connect(self._update_plot)
        self.num_points_spin.valueChanged.connect(self._update_plot)

        # Store components for later use
        self._components: list[ComponentData] = list(DEFAULT_COMPONENTS)  # noqa: F821

    def set_components(self, components: list[ComponentData]) -> None:  # noqa: F821
        """Set component data for sensitivity calculations."""
        self._components = components

    def _update_plot(self) -> None:
        """Update the sensitivity plot based on selected type."""
        plot_type = self.plot_type_combo.currentText()
        self.canvas.fig.clear()

        if plot_type == "H2 Recovery vs Recycle":
            self._plot_recovery_vs_recycle()
        elif plot_type == "Net Product vs Recycle":
            self._plot_product_vs_recycle()
        elif plot_type == "O2 Safety Analysis":
            self._plot_o2_safety()
        elif plot_type == "3D Recovery Surface":
            self._plot_3d_surface()
        elif plot_type == "Contour Map":
            self._plot_contour()

        self.canvas.draw()

    def _get_plot_style(self) -> tuple[str, str]:
        """Get the line and marker style based on checkbox states."""
        show_lines = self.show_lines_check.isChecked()
        show_markers = self.show_markers_check.isChecked()

        if show_lines and show_markers:
            return "-", "o"
        elif show_lines:
            return "-", ""
        elif show_markers:
            return "", "o"
        else:
            return "-", ""  # Default to lines

    def _plot_recovery_vs_recycle(self) -> None:
        """Plot H2 recovery vs recycle fractions."""
        num_points = self.num_points_spin.value()
        s2_range = np.linspace(0, 1, num_points)  # noqa: F821
        prod_range = np.array([0.0, 0.1, 0.2])  # noqa: F821

        sensitivity = calculate_sensitivity(  # noqa: F821
            s2_tail_recycle_range=s2_range,
            product_recycle_range=prod_range,
            components=self._components,
        )

        ax = self.canvas.fig.add_subplot(111)
        linestyle, marker = self._get_plot_style()
        markers = ["o", "s", "^"]

        for j, r_prod in enumerate(prod_range):
            ax.plot(
                s2_range * 100,
                sensitivity["h2_recovery"][:, j],
                linestyle=linestyle,
                marker=markers[j] if marker else "",
                markersize=5,
                linewidth=2,
                label=f"Product Recycle = {r_prod * 100:.0f}%",
            )

        ax.set_xlabel("Stage 2 Tail Recycle (%)")
        ax.set_ylabel("H2 Recovery (%)")
        ax.set_title("H2 Recovery vs Recycle Fractions")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_product_vs_recycle(self) -> None:
        """Plot net product vs recycle fractions."""
        num_points = self.num_points_spin.value()
        s2_range = np.linspace(0, 1, num_points)  # noqa: F821
        prod_range = np.array([0.0, 0.1, 0.2])  # noqa: F821

        sensitivity = calculate_sensitivity(  # noqa: F821
            s2_tail_recycle_range=s2_range,
            product_recycle_range=prod_range,
            components=self._components,
        )

        ax = self.canvas.fig.add_subplot(111)
        linestyle, marker = self._get_plot_style()
        markers = ["s", "^", "D"]

        for j, r_prod in enumerate(prod_range):
            ax.plot(
                s2_range * 100,
                sensitivity["net_product"][:, j],
                linestyle=linestyle,
                marker=markers[j] if marker else "",
                markersize=5,
                linewidth=2,
                label=f"Product Recycle = {r_prod * 100:.0f}%",
            )

        ax.set_xlabel("Stage 2 Tail Recycle (%)")
        ax.set_ylabel("Net Product Flow (SCFM)")
        ax.set_title("Net Product Flow vs Recycle Fractions")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_o2_safety(self) -> None:
        """Plot O2 safety analysis."""
        num_points = min(self.num_points_spin.value(), 51)  # Cap at 51 for O2 analysis
        inlet_o2_values = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float64)  # noqa: F821
        s1_removal_range = np.linspace(50.0, 95.0, num_points, dtype=np.float64)  # noqa: F821

        o2_analysis = calculate_o2_safety_analysis(  # noqa: F821
            inlet_o2_pcts=inlet_o2_values,
            stage1_o2_removal_range=s1_removal_range,
            components=self._components,
        )

        ax = self.canvas.fig.add_subplot(111)
        linestyle, marker = self._get_plot_style()
        markers_list = ["o", "s", "^", "D"]

        for j, inlet_o2 in enumerate(inlet_o2_values):
            ax.plot(
                s1_removal_range,
                o2_analysis["s2_tail_o2"][:, j],
                linestyle=linestyle,
                marker=markers_list[j] if marker else "",
                linewidth=2,
                markersize=5,
                label=f"Inlet O2 = {inlet_o2}%",
            )

        ax.axhline(y=2.0, color="red", linestyle="--", linewidth=2, label="DANGER (2%)")
        ax.fill_between(
            s1_removal_range,
            2.0,
            ax.get_ylim()[1] if ax.get_ylim()[1] > 2 else 50,
            alpha=0.2,
            color="red",
        )
        ax.axvline(x=95, color="green", linestyle=":", alpha=0.7, label="Current (95%)")
        ax.axvline(
            x=80, color="orange", linestyle=":", alpha=0.7, label="Concern (80%)"
        )

        ax.set_xlabel("Stage 1 O2 Removal (%)")
        ax.set_ylabel("Stage 2 Tail O2 (%)")
        ax.set_title("O2 Safety Analysis: S2 Tail O2 vs S1 Removal")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim((50.0, 95.0))

    def _plot_3d_surface(self) -> None:
        """Plot 3D surface of H2 recovery."""
        num_points = self.num_points_spin.value()
        s2_range = np.linspace(0, 1, num_points)  # noqa: F821
        prod_range = np.linspace(0, 0.5, max(11, num_points // 2))  # noqa: F821

        sensitivity = calculate_sensitivity(  # noqa: F821
            s2_tail_recycle_range=s2_range,
            product_recycle_range=prod_range,
            components=self._components,
        )

        S2, PROD = np.meshgrid(s2_range, prod_range, indexing="ij")  # noqa: F821

        ax: Any = self.canvas.fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            S2 * 100, PROD * 100, sensitivity["h2_recovery"], cmap="viridis", alpha=0.8
        )
        ax.set_xlabel("S2 Tail Recycle (%)")
        ax.set_ylabel("Product Recycle (%)")
        ax.set_zlabel("H2 Recovery (%)")
        ax.set_title("H2 Recovery Surface")
        self.canvas.fig.colorbar(surf, ax=ax, shrink=0.5, label="H2 Recovery (%)")

    def _plot_contour(self) -> None:
        """Plot contour map of H2 recovery."""
        num_points = self.num_points_spin.value()
        s2_range = np.linspace(0, 1, num_points)  # noqa: F821
        prod_range = np.linspace(0, 0.5, max(11, num_points // 2))  # noqa: F821

        sensitivity = calculate_sensitivity(  # noqa: F821
            s2_tail_recycle_range=s2_range,
            product_recycle_range=prod_range,
            components=self._components,
        )

        S2, PROD = np.meshgrid(s2_range, prod_range, indexing="ij")  # noqa: F821

        ax = self.canvas.fig.add_subplot(111)
        cs = ax.contourf(
            S2 * 100, PROD * 100, sensitivity["h2_recovery"], levels=20, cmap="viridis"
        )
        ax.contour(
            S2 * 100,
            PROD * 100,
            sensitivity["h2_recovery"],
            levels=[75, 77, 79, 80],
            colors="white",
            linewidths=1,
        )
        self.canvas.fig.colorbar(cs, ax=ax, label="H2 Recovery (%)")
        ax.set_xlabel("S2 Tail Recycle (%)")
        ax.set_ylabel("Product Recycle (%)")
        ax.set_title("H2 Recovery Contour Map")
        ax.plot([100], [0], "r*", markersize=15, label="Current Operation")
        ax.legend()


class PFDWidget(QWidget):  # noqa: F811, F821
    """Widget for displaying the Process Flow Diagram."""

    def __init__(self, parent: QWidget | None = None) -> None:  # noqa: F821
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)  # noqa: F821

        # Title
        title = QLabel("Process Flow Diagram")  # noqa: F821
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))  # noqa: F821
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)  # noqa: F821
        layout.addWidget(title)

        # Image label
        self.image_label = QLabel()  # noqa: F821
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # noqa: F821

        # Try to load the PFD image using a robust path resolution
        try:
            from pathlib import Path

            # Use pathlib for cleaner path handling
            script_path = Path(__file__).resolve()
            image_path = script_path.parent / "PSA System PFD.jpg"

            if image_path.exists():
                pixmap = QPixmap(str(image_path))  # noqa: F821
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        800,
                        600,
                        Qt.AspectRatioMode.KeepAspectRatio,  # noqa: F821
                        Qt.TransformationMode.SmoothTransformation,  # noqa: F821
                    )
                    self.image_label.setPixmap(scaled)
                else:
                    self.image_label.setText("PFD image could not be loaded")
            else:
                self.image_label.setText(f"PFD image not found at: {image_path}")
        except (PermissionError, OSError) as e:
            self.image_label.setText(f"Error loading PFD: {e}")

        layout.addWidget(self.image_label)

        # Stream legend
        legend_group = QGroupBox("Stream Legend")  # noqa: F821
        legend_layout = QGridLayout()  # noqa: F821

        streams = [
            ("1", "Fresh Feed (from gasifier)"),
            ("2", "Exhaust (PSA 1 tail)"),
            ("3G", "Gross Product (PSA 2 output)"),
            ("3N", "Net Product (final product)"),
            ("3R", "Product Recycle"),
            ("4", "Stage 2 Tail Recycle"),
            ("5A/5B", "Mixed Feed"),
            ("6", "Interstage (PSA 1 to PSA 2)"),
        ]

        for i, (num, desc) in enumerate(streams):
            row = i // 2
            col = (i % 2) * 2
            legend_layout.addWidget(QLabel(f"<b>{num}:</b>"), row, col)  # noqa: F821
            legend_layout.addWidget(QLabel(desc), row, col + 1)  # noqa: F821

        legend_group.setLayout(legend_layout)
        layout.addWidget(legend_group)


class PSAMainWindow(ThemedWindowMixin, QMainWindow):  # noqa: F811, F821
    """Main window for PSA analysis application."""

    def __init__(self) -> None:
        super().__init__()
        self.setup_theme_support()
        self.setWindowTitle("Two-Stage PSA System Analysis")
        self.setMinimumSize(1400, 900)
        self._setup_menu()
        self._setup_ui()
        self._connect_signals()

    def _setup_menu(self) -> None:
        """Setup the menu bar with launch options."""
        menubar = self.menuBar()
        if menubar is None:
            return

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        if tools_menu is None:
            return

        # Launch Jupyter Notebook
        notebook_action = QAction("Open Jupyter Notebook", self)  # noqa: F821
        notebook_action.setShortcut("Ctrl+J")
        notebook_action.triggered.connect(self._launch_jupyter)
        tools_menu.addAction(notebook_action)

        # Launch Colab
        colab_action = QAction("Open in Google Colab", self)  # noqa: F821
        colab_action.setShortcut("Ctrl+G")
        colab_action.triggered.connect(self._launch_colab)
        tools_menu.addAction(colab_action)

        tools_menu.addSeparator()

        # Launch Web App
        webapp_action = QAction("Launch Web App (Streamlit)", self)  # noqa: F821
        webapp_action.setShortcut("Ctrl+W")
        webapp_action.triggered.connect(self._launch_webapp)
        tools_menu.addAction(webapp_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        if help_menu is None:
            return

        about_action = QAction("About", self)  # noqa: F821
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _launch_jupyter(self) -> None:
        """Launch the Jupyter notebook."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        notebook_path = os.path.join(script_dir, "psa_analysis.ipynb")

        if os.path.exists(notebook_path):
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["jupyter", "notebook", notebook_path],
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                    )
                else:
                    subprocess.Popen(["jupyter", "notebook", notebook_path])
                QMessageBox.information(  # noqa: F821
                    self, "Jupyter Notebook", "Launching Jupyter Notebook..."
                )
            except FileNotFoundError:
                QMessageBox.warning(  # noqa: F821
                    self,
                    "Jupyter Not Found",
                    "Jupyter is not installed. Install with: pip install jupyter",
                )
        else:
            QMessageBox.warning(  # noqa: F821
                self, "File Not Found", f"Notebook not found: {notebook_path}"
            )

    def _launch_colab(self) -> None:
        """Open the Colab-compatible notebook in Google Colab."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_notebook = os.path.join(script_dir, "psa_analysis_colab.ipynb")

        msg = QMessageBox(self)  # noqa: F821
        msg.setWindowTitle("Open in Google Colab")
        msg.setText("To use Google Colab, you need to upload the notebook to GitHub.")
        msg.setInformativeText(
            f"Local notebook location:\n{local_notebook}\n\n"
            "Options:\n"
            "1. Upload to GitHub and update the Colab URL\n"
            "2. Upload directly to Google Drive and open in Colab\n"
            "3. Copy the notebook content manually"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel  # noqa: F821
        )
        msg.setDefaultButton(QMessageBox.StandardButton.Open)  # noqa: F821

        if msg.exec() == QMessageBox.StandardButton.Open:  # noqa: F821
            webbrowser.open("https://colab.research.google.com/")

    def _launch_webapp(self) -> None:
        """Launch the Streamlit web app."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        webapp_path = os.path.join(script_dir, "psa_webapp.py")

        if os.path.exists(webapp_path):
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["streamlit", "run", webapp_path],
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                    )
                else:
                    subprocess.Popen(["streamlit", "run", webapp_path])
                QMessageBox.information(  # noqa: F821
                    self,
                    "Web App",
                    "Launching Streamlit web app...\n\n"
                    "The app will open in your default browser.",
                )
            except FileNotFoundError:
                QMessageBox.warning(  # noqa: F821
                    self,
                    "Streamlit Not Found",
                    "Streamlit is not installed. Install with: pip install streamlit",
                )
        else:
            QMessageBox.warning(  # noqa: F821
                self, "File Not Found", f"Web app not found: {webapp_path}"
            )

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(  # noqa: F821
            self,
            "About PSA System Analysis",
            "<h3>Two-Stage PSA System Analysis</h3>"
            "<p>Version 1.0</p>"
            "<p>A comprehensive tool for analyzing pressure swing adsorption systems.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Mass balance calculations</li>"
            "<li>Sensitivity analysis</li>"
            "<li>O2 safety analysis</li>"
            "<li>Interactive plots</li>"
            "</ul>"
            "<p>All calculations validated against Excel reference model.</p>",
        )

    def _setup_ui(self) -> None:
        central_widget = QWidget()  # noqa: F821
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)  # noqa: F821

        # Left panel - Inputs
        self.input_panel = InputPanel()
        self.input_panel.setMaximumWidth(400)
        main_layout.addWidget(self.input_panel)

        # Right panel - Tabs for different views
        self.tab_widget = QTabWidget()  # noqa: F821

        # Results tab
        results_scroll = QScrollArea()  # noqa: F821
        results_scroll.setWidgetResizable(True)
        self.results_panel = ResultsPanel()
        results_scroll.setWidget(self.results_panel)
        self.tab_widget.addTab(results_scroll, "Results")

        # Sensitivity Analysis tab
        self.sensitivity_widget = SensitivityPlotWidget()
        self.tab_widget.addTab(self.sensitivity_widget, "Sensitivity Analysis")

        # PFD tab
        pfd_scroll = QScrollArea()  # noqa: F821
        pfd_scroll.setWidgetResizable(True)
        self.pfd_widget = PFDWidget()
        pfd_scroll.setWidget(self.pfd_widget)
        self.tab_widget.addTab(pfd_scroll, "Process Flow Diagram")

        main_layout.addWidget(self.tab_widget, stretch=1)

        # Run initial calculation
        self._calculate()

    def _connect_signals(self) -> None:
        """Connect UI signals to slots."""
        self.input_panel.input_changed.connect(self._on_input_change)

        # Tab change triggers plot pre-calculation
        self.tab_widget.currentChanged.connect(self._on_tab_change)

    def _on_input_change(self) -> None:
        """Handle any input value changes - auto-calculate."""
        self._calculate()

    def _on_tab_change(self, index: int) -> None:
        """Handle tab changes - pre-calculate plots when switching to sensitivity tab."""
        if index == 1:  # Sensitivity Analysis tab
            self.sensitivity_widget._update_plot()

    def _calculate(self) -> None:
        """Run PSA calculation with current inputs."""
        try:
            total_feed, s2_recycle, prod_recycle, components = (
                self.input_panel.get_parameters()
            )

            model = PSAModel(  # noqa: F821
                total_feed_scfm=total_feed,
                s2_tail_recycle_frac=s2_recycle,
                product_recycle_frac=prod_recycle,
                components=components,
            )

            results = model.calculate()
            self.results_panel.update_results(results)

            # Update sensitivity widget with current components
            self.sensitivity_widget.set_components(components)

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")  # noqa: F821
        except (RuntimeError, AttributeError) as e:
            QMessageBox.critical(self, "Calculation Error", f"Error: {e}")  # noqa: F821


def main() -> None:
    """Main entry point for the GUI application."""
    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = PSAMainWindow()
    setup_themed_app(app, window, settings_app="PSAPackage")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
