"""Syngas Compression Calculator — PyQt6 GUI layer.

The computation engine has been extracted to
:mod:`._syngas_compression_engine` (Issue #2892) to separate domain logic
from the presentation layer.

Public API (unchanged for backwards compatibility)
--------------------------------------------------
- :class:`CompressionStage`
- :class:`SyngasCompressionEngine`
- :class:`SyngasCompressionCalculatorWidget`
- :func:`create_syngas_compression_calculator`
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
from matplotlib.figure import Figure

# Try PyQt6 imports - these are optional for core calculations
try:
    from PyQt6.QtCore import QThread, QTimer, pyqtSignal, pyqtSlot
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QLabel,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSplitter,
        QTableWidget,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QWidget = object  # type: ignore[assignment,misc]
    QThread = object  # type: ignore[assignment,misc]

# Logging
try:
    from integrated_process_simulator.utilities.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# Matplotlib backend selection
if os.environ.get("HEADLESS", "false").lower() == "true":
    try:
        mpl.use("Agg")
    except (ImportError, RuntimeError) as e:
        logging.getLogger(__name__).debug("Failed to set Agg backend: %s", e)
else:
    try:
        mpl.use("QtAgg")
    except (RuntimeError, AttributeError):
        mpl.use("Agg")

if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
else:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ---------------------------------------------------------------------------
# Engine imports (pure computation — no GUI dependency)
# ---------------------------------------------------------------------------
from ._syngas_compression_engine import (  # noqa: E402
    CompressionStage,
    SyngasCompressionEngine,
)
from .constants import (  # noqa: E402
    ATOL_ZERO,
    CELSIUS_TO_KELVIN_OFFSET,
    INTERCOOLER_OUTLET_TEMP_K,
)

# Import BaseCalculatorWidget for state management
try:
    from ..ui.widgets.base_calculator_widget import BaseCalculatorWidget

    BASE_CALCULATOR_AVAILABLE = True
except ImportError:
    BASE_CALCULATOR_AVAILABLE = False

    class BaseCalculatorWidget(QWidget):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            QWidget.__init__(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Qt worker thread
# ---------------------------------------------------------------------------


class CompressionCalculationWorker(QThread):
    """Worker thread for compression calculations."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        engine: Any,
        stages: Any,
        flow_rate: float,
        composition: Any,
        intercooling: bool,
    ) -> None:
        """Initialize the worker."""
        if flow_rate is None:
            raise ValueError("flow_rate must be provided")
        super().__init__()
        self.engine = engine
        self.stages = stages
        self.flow_rate = flow_rate
        self.composition = composition
        self.intercooling = intercooling

    def run(self) -> None:
        """Run the compression calculation."""
        try:
            result = self.engine.calculate_multistage_compression(
                self.stages,
                self.flow_rate,
                self.composition,
                self.intercooling,
            )
            analysis = self.engine.analyze_process_conditions(result)
            self.finished.emit({"result": result, "analysis": analysis})
        except (ValueError, TypeError, ArithmeticError) as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# GUI widget
# ---------------------------------------------------------------------------

if HAS_PYQT:
    BaseClass = BaseCalculatorWidget if BASE_CALCULATOR_AVAILABLE else QWidget

    class SyngasCompressionCalculatorWidget(BaseClass):  # type: ignore[valid-type, misc]
        """Main syngas compression calculator widget."""

        calculation_finished = pyqtSignal(dict)

        def __init__(self, parent: Any = None) -> None:
            """Initialize the widget."""
            if BASE_CALCULATOR_AVAILABLE:
                super().__init__(calculator_name="SyngasCompression", parent=parent)
            else:
                super().__init__(parent)
            self.engine = SyngasCompressionEngine()
            self.init_ui()
            QTimer.singleShot(100, self.set_default_values)
            QTimer.singleShot(200, self.setup_state_management)

        def setup_state_management(self) -> None:
            """Register UI components for state persistence and copy functionality."""
            for splitter in self.findChildren(QSplitter):
                self.register_splitter(splitter, "main_splitter")
            for table in self.findChildren(QTableWidget):
                self.register_copyable_widget(table, "table")
            for text_edit in self.findChildren(QTextEdit):
                self.register_copyable_widget(text_edit, "text")
            for label in self.findChildren(QLabel):
                if (
                    "result" in label.objectName().lower()
                    or "value" in label.objectName().lower()
                ):
                    self.register_copyable_widget(label, "label")

        def closeEvent(self, event: Any) -> None:
            """Save state before closing."""
            self.save_state()
            super().closeEvent(event)

        def showEvent(self, event: Any) -> None:
            """Refresh layout when widget becomes visible."""
            super().showEvent(event)
            QTimer.singleShot(50, self._refresh_layout)

        def _refresh_layout(self) -> None:
            """Fix visibility issues when dynamically added to tabs."""
            try:
                if hasattr(self, "tab_widget"):
                    self.tab_widget.show()
                    current_idx = self.tab_widget.currentIndex()
                    if current_idx >= 0:
                        current_widget = self.tab_widget.widget(current_idx)
                        if current_widget:
                            current_widget.show()
                            current_widget.updateGeometry()
                layout = self.layout()
                if layout is not None:
                    layout.activate()
                self.updateGeometry()
                self.update()
            except RuntimeError:
                pass

        def init_ui(self) -> None:
            """Initialize the user interface."""
            layout = QVBoxLayout()
            self.tab_widget = QTabWidget()
            self.create_input_tab()
            self.create_results_tab()
            self.create_analysis_tab()
            self.create_plots_tab()
            layout.addWidget(self.tab_widget)
            self.setLayout(layout)

        def create_input_tab(self) -> None:
            """Create the input parameters tab."""
            input_widget = QWidget()

            scroll = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout()

            scroll_layout.addWidget(self._create_composition_group())
            scroll_layout.addWidget(self._create_process_conditions_group())
            scroll_layout.addWidget(self._create_stages_group())
            scroll_layout.addWidget(self._create_config_group())

            self.calculate_button = QPushButton("Calculate Compression")
            self.calculate_button.clicked.connect(self.calculate_compression)
            scroll_layout.addWidget(self.calculate_button)

            scroll_widget.setLayout(scroll_layout)
            scroll.setWidget(scroll_widget)
            scroll.setWidgetResizable(True)

            input_widget.setLayout(QVBoxLayout())
            layout = input_widget.layout()
            if layout:
                layout.addWidget(scroll)

            self.tab_widget.addTab(input_widget, "Input Parameters")

        def _create_composition_group(self) -> QGroupBox:
            """Create the gas composition input group."""
            comp_group = QGroupBox("Syngas Composition (mol%)")
            comp_layout = QGridLayout()

            self.composition_inputs = {}
            components = ["H2", "CO", "CO2", "CH4", "N2", "H2O", "Ar"]
            for i, comp in enumerate(components):
                row = i // 3
                col = i % 3
                comp_layout.addWidget(QLabel(f"{comp}:"), row, col * 2)
                spinbox = QDoubleSpinBox()
                spinbox.setRange(0, 100)
                spinbox.setDecimals(2)
                spinbox.setSuffix(" %")
                self.composition_inputs[comp] = spinbox
                comp_layout.addWidget(spinbox, row, col * 2 + 1)

            comp_group.setLayout(comp_layout)
            return comp_group

        def _create_process_conditions_group(self) -> QGroupBox:
            """Create the process conditions input group."""
            process_group = QGroupBox("Process Conditions")
            process_layout = QFormLayout()

            self.flow_rate_input = QDoubleSpinBox()
            self.flow_rate_input.setRange(0, 10000)
            self.flow_rate_input.setDecimals(1)
            self.flow_rate_input.setSuffix(" kmol/h")

            self.inlet_temp_input = QDoubleSpinBox()
            self.inlet_temp_input.setRange(-50, 500)
            self.inlet_temp_input.setDecimals(1)
            self.inlet_temp_input.setSuffix(" °C")

            self.inlet_pressure_input = QDoubleSpinBox()
            self.inlet_pressure_input.setRange(0.1, 1000)
            self.inlet_pressure_input.setDecimals(2)
            self.inlet_pressure_input.setSuffix(" bar")

            process_layout.addRow("Flow Rate:", self.flow_rate_input)
            process_layout.addRow("Inlet Temperature:", self.inlet_temp_input)
            process_layout.addRow("Inlet Pressure:", self.inlet_pressure_input)

            process_group.setLayout(process_layout)
            return process_group

        def _create_stages_group(self) -> QGroupBox:
            """Create the compression stages input group."""
            stages_group = QGroupBox("Compression Stages")
            stages_layout = QVBoxLayout()

            self.stage_table = QTableWidget()
            self.stage_table.setColumnCount(4)
            self.stage_table.setRowCount(4)
            self.stage_table.setHorizontalHeaderLabels(
                ["Inlet P (bar)", "Outlet P (bar)", "Efficiency (%)", "Active"],
            )

            header = self.stage_table.horizontalHeader()
            if header is not None:
                header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            self.stage_inputs = []
            for row in range(4):
                row_inputs: list[QWidget] = []
                for col in range(3):
                    if col == 2:
                        spinbox = QDoubleSpinBox()
                        spinbox.setRange(50, 100)
                        spinbox.setDecimals(1)
                        spinbox.setSuffix(" %")
                    else:
                        spinbox = QDoubleSpinBox()
                        spinbox.setRange(0.1, 1000)
                        spinbox.setDecimals(2)
                        spinbox.setSuffix(" bar")

                    self.stage_table.setCellWidget(row, col, spinbox)
                    row_inputs.append(spinbox)

                checkbox = QCheckBox()
                checkbox.setChecked(True)
                self.stage_table.setCellWidget(row, 3, checkbox)
                row_inputs.append(checkbox)

                self.stage_inputs.append(row_inputs)

            stages_layout.addWidget(self.stage_table)
            stages_group.setLayout(stages_layout)
            return stages_group

        def _create_config_group(self) -> QGroupBox:
            """Create the compression configuration input group."""
            config_group = QGroupBox("Compression Configuration")
            config_layout = QFormLayout()

            self.compression_type_combo = QComboBox()
            self.compression_type_combo.addItems(
                ["Isentropic", "Polytropic", "Isothermal"],
            )

            self.intercooling_checkbox = QCheckBox("Enable intercooling between stages")
            self.intercooling_checkbox.setChecked(True)

            config_layout.addRow("Compression Type:", self.compression_type_combo)
            config_layout.addRow("", self.intercooling_checkbox)

            config_group.setLayout(config_layout)
            return config_group

        def create_results_tab(self) -> None:
            """Create the results display tab."""
            results_widget = QWidget()
            layout = QVBoxLayout()

            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            layout.addWidget(self.results_text)

            results_widget.setLayout(layout)
            self.tab_widget.addTab(results_widget, "Results")

        def create_analysis_tab(self) -> None:
            """Create the analysis and concerns tab."""
            analysis_widget = QWidget()
            layout = QVBoxLayout()

            self.analysis_text = QTextEdit()
            self.analysis_text.setReadOnly(True)
            layout.addWidget(self.analysis_text)

            analysis_widget.setLayout(layout)
            self.tab_widget.addTab(analysis_widget, "Analysis & Concerns")

        def create_plots_tab(self) -> None:
            """Create the plots tab."""
            plots_widget = QWidget()
            layout = QVBoxLayout()

            self.figure = Figure(figsize=(10, 8))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)

            plots_widget.setLayout(layout)
            self.tab_widget.addTab(plots_widget, "Plots")

        def set_default_values(self) -> None:
            """Set default input values."""
            try:
                default_composition = {
                    "H2": 20.0,
                    "CO": 25.0,
                    "CO2": 15.0,
                    "CH4": 5.0,
                    "N2": 30.0,
                    "H2O": 5.0,
                    "Ar": 0.0,
                }

                for comp, value in default_composition.items():
                    if comp in self.composition_inputs:
                        self.composition_inputs[comp].setValue(value)

                self.flow_rate_input.setValue(100.0)
                self.inlet_temp_input.setValue(40.0)
                self.inlet_pressure_input.setValue(1.0)

                default_stages = [
                    [1.0, 3.0, 85.0],
                    [3.0, 9.0, 85.0],
                    [9.0, 27.0, 85.0],
                    [27.0, 81.0, 85.0],
                ]

                for i, stage_data in enumerate(default_stages):
                    for j, value in enumerate(stage_data):
                        cast(QDoubleSpinBox, self.stage_inputs[i][j]).setValue(value)
            except RuntimeError:
                pass
            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.warning("Failed to set default values: %s", e)

        def calculate_compression(self) -> None:
            """Perform compression calculations."""
            try:
                composition = {
                    comp: self.composition_inputs[comp].value()
                    for comp in self.composition_inputs
                }

                flow_rate = self.flow_rate_input.value()
                inlet_temp = self.inlet_temp_input.value() + CELSIUS_TO_KELVIN_OFFSET
                self.inlet_pressure_input.value()
                compression_type = self.compression_type_combo.currentText().lower()
                intercooling = self.intercooling_checkbox.isChecked()

                stages = []
                for i, stage_inputs in enumerate(self.stage_inputs):
                    if cast(QCheckBox, stage_inputs[3]).isChecked():
                        stage = CompressionStage(
                            inlet_pressure=cast(
                                QDoubleSpinBox, stage_inputs[0]
                            ).value(),
                            outlet_pressure=cast(
                                QDoubleSpinBox, stage_inputs[1]
                            ).value(),
                            inlet_temperature=(
                                inlet_temp if i == 0 else INTERCOOLER_OUTLET_TEMP_K
                            ),
                            efficiency=cast(QDoubleSpinBox, stage_inputs[2]).value()
                            / 100.0,
                            compression_type=compression_type,
                        )
                        stages.append(stage)

                if not stages:
                    QMessageBox.warning(
                        self,
                        "Error",
                        "No valid compression stages defined",
                    )
                    return

                self.worker = CompressionCalculationWorker(
                    self.engine,
                    stages,
                    flow_rate,
                    composition,
                    intercooling,
                )
                self.worker.finished.connect(self.on_calculation_finished)
                self.worker.error.connect(self.on_calculation_error)
                self.worker.start()

            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                QMessageBox.critical(
                    self,
                    "Calculation Error",
                    f"An error occurred: {e!s}",
                )

        @pyqtSlot(dict)
        def on_calculation_finished(self, data: dict[str, Any]) -> None:
            """Handle calculation completion."""
            if data is None:
                raise ValueError("data must be provided")
            result = data["result"]
            analysis = data["analysis"]
            self.display_results(result, analysis)
            self.display_analysis(analysis)
            self.create_plots(result)
            self.calculation_finished.emit(data)

        @pyqtSlot(str)
        def on_calculation_error(self, error_message: str) -> None:
            """Handle calculation error."""
            QMessageBox.critical(
                self,
                "Calculation Error",
                f"An error occurred: {error_message}",
            )

        def display_results(
            self, result: dict[str, Any], analysis: dict[str, Any]
        ) -> None:
            """Display calculation results."""
            if result is None:
                raise ValueError("result must be provided")
            output_parts = [
                "SYNGAS COMPRESSION CALCULATION RESULTS\n",
                "=" * 50 + "\n\n",
            ]

            mix_props = result["mixture_properties"]
            output_parts.extend(
                [
                    "Mixture Properties:\n",
                    f"  Molecular Weight: {mix_props['molecular_weight']:.2f} g/mol\n",
                    f"  Critical Temperature: {mix_props['critical_temperature']:.1f} K\n",
                    f"  Critical Pressure: {mix_props['critical_pressure']:.1f} bar\n",
                    f"  Heat Capacity Ratio (γ): {mix_props['heat_capacity_ratio']:.3f}\n\n",
                    "Compression Stages:\n",
                    "-" * 30 + "\n",
                ]
            )

            for stage_result in result["stages"]:
                stage_num = stage_result["stage_number"]
                output_parts.extend(
                    [
                        f"\nStage {stage_num}:\n",
                        f"  Inlet Temperature: {stage_result['inlet_temp']:.1f} K "
                        f"({stage_result['inlet_temp'] - CELSIUS_TO_KELVIN_OFFSET:.1f} deg C)\n",
                        f"  Outlet Temperature: {stage_result['outlet_temp']:.1f} K "
                        f"({stage_result['outlet_temp'] - CELSIUS_TO_KELVIN_OFFSET:.1f} deg C)\n",
                        f"  Heat Rise: {stage_result['heat_rise']:.1f} K\n",
                        f"  Pressure Ratio: {stage_result['pressure_ratio']:.2f}\n",
                        f"  Power Required: {stage_result['power_hp']:.1f} HP\n",
                    ]
                )

                water_info = stage_result["water_dropout"]
                if water_info["water_dropout"] > ATOL_ZERO:
                    output_parts.extend(
                        [
                            f"  Water Dropout: {water_info['water_dropout']:.3f} mol%\n",
                            f"  Condensation Rate: {water_info['condensation_rate']:.1f}%\n",
                        ]
                    )

            output_parts.extend(
                [
                    "\nSUMMARY:\n",
                    "-" * 20 + "\n",
                    f"Total Power Required: {result['total_power_hp']:.1f} HP\n",
                    f"Final Temperature: {result['final_temperature']:.1f} K "
                    f"({result['final_temperature'] - CELSIUS_TO_KELVIN_OFFSET:.1f} deg C)\n",
                    f"Final Pressure: {result['final_pressure']:.1f} bar\n",
                    f"Total Water Dropout: {analysis['total_water_dropout']:.3f} mol%\n",
                ]
            )

            if analysis["average_efficiency"]:
                output_parts.append(
                    f"Average Efficiency: {analysis['average_efficiency'] * 100:.1f}%\n"
                )

            self.results_text.setText("".join(output_parts))

        def display_analysis(self, analysis: dict[str, Any]) -> None:
            """Display analysis and concerns."""
            if analysis is None:
                raise ValueError("analysis must be provided")
            output_parts = [
                "PROCESS ANALYSIS & CONCERNS\n",
                "=" * 40 + "\n\n",
            ]

            if analysis["warnings"]:
                output_parts.extend(["⚠️  CRITICAL WARNINGS:\n", "-" * 25 + "\n"])
                for warning in analysis["warnings"]:
                    output_parts.append(f"• {warning}\n")
                output_parts.append("\n")

            if analysis["concerns"]:
                output_parts.extend(["⚠️  CONCERNS:\n", "-" * 15 + "\n"])
                for concern in analysis["concerns"]:
                    output_parts.append(f"• {concern}\n")
                output_parts.append("\n")

            if analysis["recommendations"]:
                output_parts.extend(["💡 RECOMMENDATIONS:\n", "-" * 20 + "\n"])
                for rec in analysis["recommendations"]:
                    output_parts.append(f"• {rec}\n")
                output_parts.append("\n")

            if not analysis["warnings"] and not analysis["concerns"]:
                output_parts.extend(
                    [
                        "✅ No significant concerns detected.\n",
                        "Process conditions appear to be within acceptable limits.\n",
                    ]
                )

            self.analysis_text.setText("".join(output_parts))

        def create_plots(self, result: dict[str, Any]) -> None:
            """Create visualization plots."""
            if result is None:
                raise ValueError("result must be provided")
            self.figure.clear()

            stages = result["stages"]
            stage_nums = [s["stage_number"] for s in stages]
            temperatures = [s["outlet_temp"] - CELSIUS_TO_KELVIN_OFFSET for s in stages]
            pressures = [s["pressure_ratio"] for s in stages]
            powers = [s["power_hp"] for s in stages]
            water_dropouts = [s["water_dropout"]["water_dropout"] for s in stages]

            ax1 = self.figure.add_subplot(2, 2, 1)
            ax2 = self.figure.add_subplot(2, 2, 2)
            ax3 = self.figure.add_subplot(2, 2, 3)
            ax4 = self.figure.add_subplot(2, 2, 4)

            ax1.plot(stage_nums, temperatures, "bo-", linewidth=2, markersize=8)
            ax1.set_xlabel("Compression Stage")
            ax1.set_ylabel("Temperature (°C)")
            ax1.set_title("Temperature Profile")
            ax1.grid(True, alpha=0.3)

            ax2.bar(stage_nums, pressures, alpha=0.7, color="green")
            ax2.set_xlabel("Compression Stage")
            ax2.set_ylabel("Pressure Ratio")
            ax2.set_title("Pressure Ratio per Stage")
            ax2.grid(True, alpha=0.3)

            ax3.bar(stage_nums, powers, alpha=0.7, color="orange")
            ax3.set_xlabel("Compression Stage")
            ax3.set_ylabel("Power (HP)")
            ax3.set_title("Power Requirement per Stage")
            ax3.grid(True, alpha=0.3)

            ax4.bar(stage_nums, water_dropouts, alpha=0.7, color="blue")
            ax4.set_xlabel("Compression Stage")
            ax4.set_ylabel("Water Dropout (mol%)")
            ax4.set_title("Water Dropout per Stage")
            ax4.grid(True, alpha=0.3)

            self.figure.tight_layout()
            self.canvas.draw()


def create_syngas_compression_calculator(parent: Any = None) -> QWidget:
    """Factory function to create syngas compression calculator widget."""
    return SyngasCompressionCalculatorWidget(parent=parent)


__all__ = [
    "CompressionStage",
    "SyngasCompressionEngine",
    "CompressionCalculationWorker",
    "SyngasCompressionCalculatorWidget",
    "create_syngas_compression_calculator",
]
