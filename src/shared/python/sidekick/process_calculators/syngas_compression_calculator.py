# ruff: noqa: E501
#!/usr/bin/env python3
"""Advanced Syngas Compression Calculator — GUI widget and worker thread.

Separation of concerns (closes #1943)
--------------------------------------
The original 1,218-line file previously mixed three concerns in one module.
They have been fully separated:

* **Calculation engine** → ``syngas_compression_engine.py``
  ``SyngasCompressionEngine`` and ``CompressionStage`` live there.  No Qt
  dependency.  Safe to import headlessly (tests, API handlers, batch jobs).

* **Worker thread** → ``CompressionCalculationWorker`` (this file)
  Thin ``QThread`` subclass that delegates to the engine and emits signals.

* **GUI widget** → ``SyngasCompressionCalculatorWidget`` (this file)
  PyQt6 widget.  Imports from the engine module and uses the worker.

``CompressionStage`` and ``SyngasCompressionEngine`` are re-exported here so
existing ``from ...syngas_compression_calculator import X`` imports continue
to work without changes.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, cast

from shared.python.theme.integration import get_theme_manager
from shared.python.theme.matplotlib_style import apply_plot_theme

from .constants import (
    ATOL_ZERO,
    CELSIUS_TO_KELVIN_OFFSET,
    INTERCOOLER_OUTLET_TEMP_K,
)

# Re-export engine classes so existing ``from ...syngas_compression_calculator import X``  # noqa: E501
# imports continue to work.
from .syngas_compression_engine import (  # noqa: F401
    CompressionStage,
    SyngasCompressionEngine,
)

# matplotlib is imported lazily inside methods to prevent Windows hang

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

    def pyqtSignal(*args, **kwargs):
        return None

    def pyqtSlot(*args, **kwargs):
        return lambda f: f


try:
    from integrated_process_simulator.utilities.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


def _setup_matplotlib_backend() -> None:
    """Configure the matplotlib backend lazily (called at GUI init time)."""
    import matplotlib as mpl  # lazy import

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


def _get_figure_canvas_class() -> type:
    """Lazily load FigureCanvas to prevent matplotlib backend hang at import."""
    try:
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas,
        )

        return FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_agg import (  # type: ignore[assignment]
            FigureCanvasAgg as FigureCanvas,
        )

        return FigureCanvas


if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Import existing syngas water content utility
# ruff: noqa: E402


# Import BaseCalculatorWidget for state management
try:
    from ..ui.widgets.base_calculator_widget import BaseCalculatorWidget

    BASE_CALCULATOR_AVAILABLE = True
except ImportError:
    BASE_CALCULATOR_AVAILABLE = False

    # Fallback to QWidget if BaseCalculatorWidget is not available
    class BaseCalculatorWidget(QWidget):  # type: ignore[attr-defined]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            QWidget.__init__(self, *args, **kwargs)


# Import species database with fallback
try:
    from integrated_process_simulator.calculators.thermodynamic_properties.species_database import (  # noqa: E501
        get_species_database,
    )
except ImportError:
    # Minimal fallback for standalone use
    from dataclasses import dataclass as _dc

    @_dc
    class _SpeciesData:
        molecular_weight: float  # kg/mol
        critical_temperature: float  # K
        critical_pressure: float  # Pa

    # Source: NIST / Perry's Chemical Engineers' Handbook (approximate values)
    _SPECIES_TABLE: dict[str, "_SpeciesData"] = {
        "CO": _SpeciesData(0.02801, 132.9, 3.499e6),
        "CO2": _SpeciesData(0.04401, 304.2, 7.376e6),
        "H2": _SpeciesData(0.00202, 33.2, 1.297e6),
        "H2O": _SpeciesData(0.01802, 647.1, 22.064e6),
        "CH4": _SpeciesData(0.01604, 190.6, 4.604e6),
        "N2": _SpeciesData(0.02801, 126.2, 3.390e6),
        "O2": _SpeciesData(0.03200, 154.6, 5.046e6),
        "H2S": _SpeciesData(0.03408, 373.5, 8.963e6),
        "Ar": _SpeciesData(0.03995, 150.8, 4.874e6),
    }

    class _MinimalSpeciesDB:
        def get_molecular_weight(self, species: str) -> float | None:
            s = _SPECIES_TABLE.get(species)
            return s.molecular_weight if s else None

        def get_species(self, species: str) -> "_SpeciesData | None":
            return _SPECIES_TABLE.get(species)

    def get_species_database() -> Any:
        return _MinimalSpeciesDB()


class CompressionCalculationWorker(QThread):
    """Worker thread for compression calculations"""

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
        """Initialize the class."""
        if flow_rate is None:
            raise ValueError("flow_rate must be provided")
        super().__init__()
        self.engine = engine
        self.stages = stages
        self.flow_rate = flow_rate
        self.composition = composition
        self.intercooling = intercooling

    def run(self) -> None:
        """Run the compression calculation"""
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


if HAS_PYQT:
    # Handle dynamic base class based on availability
    BaseClass = BaseCalculatorWidget if BASE_CALCULATOR_AVAILABLE else QWidget

    class SyngasCompressionCalculatorWidget(BaseClass):  # type: ignore[valid-type, misc]
        """Main syngas compression calculator widget"""

        calculation_finished = pyqtSignal(dict)

        def __init__(self, parent: Any = None) -> None:
            """Initialize the class."""
            if BASE_CALCULATOR_AVAILABLE:
                super().__init__(calculator_name="SyngasCompression", parent=parent)
            else:
                super().__init__(parent)
            self.engine = SyngasCompressionEngine()
            self.init_ui()
            # init_ui() completes synchronously; call directly rather than
            # relying on timed delays (which masked init-order bugs and raced
            # with parent show). See #2098.
            self.set_default_values()
            self.setup_state_management()

        def setup_state_management(self) -> None:
            """Set up state management for UI components.

            Registers splitters, tables, text editors, and result labels
            for state persistence and copy functionality.
            """
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
            """Handle widget close event.

            Saves the current state before closing the widget.

            Args:
                event: The close event to handle
            """
            self.save_state()
            super().closeEvent(event)

        def showEvent(self, event: Any) -> None:
            """Handle widget show event.

            Ensures proper layout and visibility when the widget becomes visible,
            particularly when added dynamically to a tab widget.

            Args:
                event: The show event to handle
            """
            super().showEvent(event)
            # Defer layout refresh to ensure all child widgets are properly initialized
            QTimer.singleShot(50, self._refresh_layout)

        def _refresh_layout(self) -> None:
            """Refresh the widget layout to ensure proper display.

            Fixes visibility issues when the widget is dynamically added to tabs.
            """
            try:
                # Ensure the tab widget and its contents are visible
                if hasattr(self, "tab_widget"):
                    self.tab_widget.show()
                    # Refresh the current tab
                    current_idx = self.tab_widget.currentIndex()
                    if current_idx >= 0:
                        current_widget = self.tab_widget.widget(current_idx)
                        if current_widget:
                            current_widget.show()
                            current_widget.updateGeometry()

                # Force layout update
                layout = self.layout()
                if layout is not None:
                    layout.activate()
                self.updateGeometry()
                self.update()
            except RuntimeError:
                # Widget was deleted before timer fired
                pass

        def init_ui(self) -> None:
            """Initialize the user interface"""
            layout = QVBoxLayout()

            # Create tab widget
            self.tab_widget = QTabWidget()

            # Create tabs
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

            # Calculate button
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
                    if col == 2:  # Efficiency column
                        spinbox = QDoubleSpinBox()
                        spinbox.setRange(50, 100)
                        spinbox.setDecimals(1)
                        spinbox.setSuffix(" %")
                    else:  # Pressure columns
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
            """Create the results display tab"""
            results_widget = QWidget()
            layout = QVBoxLayout()

            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            layout.addWidget(self.results_text)

            results_widget.setLayout(layout)
            self.tab_widget.addTab(results_widget, "Results")

        def create_analysis_tab(self) -> None:
            """Create the analysis and concerns tab"""
            analysis_widget = QWidget()
            layout = QVBoxLayout()

            self.analysis_text = QTextEdit()
            self.analysis_text.setReadOnly(True)
            layout.addWidget(self.analysis_text)

            analysis_widget.setLayout(layout)
            self.tab_widget.addTab(analysis_widget, "Analysis & Concerns")

        def create_plots_tab(self) -> None:
            """Create the plots tab"""
            plots_widget = QWidget()
            layout = QVBoxLayout()

            # Create matplotlib figure
            from matplotlib.figure import Figure  # lazy import

            self.figure = Figure(figsize=(10, 8))
            _tm = get_theme_manager()
            apply_plot_theme(self.figure, _tm.get_current_colors())
            _tm.themeChanged.connect(
                lambda name: apply_plot_theme(
                    self.figure, _tm.get_theme_colors(name) or _tm.get_current_colors()
                )
            )
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)

            plots_widget.setLayout(layout)
            self.tab_widget.addTab(plots_widget, "Plots")

        def set_default_values(self) -> None:
            """Set default values for the calculator"""
            try:
                # Default syngas composition (typical biomass gasification)
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

                # Default process conditions
                self.flow_rate_input.setValue(100.0)
                self.inlet_temp_input.setValue(40.0)
                self.inlet_pressure_input.setValue(1.0)

                # Default compression stages
                default_stages = [
                    [1.0, 3.0, 85.0],  # Stage 1: 1 to 3 bar, 85% efficiency
                    [3.0, 9.0, 85.0],  # Stage 2: 3 to 9 bar, 85% efficiency
                    [9.0, 27.0, 85.0],  # Stage 3: 9 to 27 bar, 85% efficiency
                    [27.0, 81.0, 85.0],  # Stage 4: 27 to 81 bar, 85% efficiency
                ]

                for i, stage_data in enumerate(default_stages):
                    for j, value in enumerate(stage_data):
                        cast(QDoubleSpinBox, self.stage_inputs[i][j]).setValue(value)
            except RuntimeError:
                # Widget might be deleted (e.g. tab closed immediately), ignore
                pass
            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.warning(f"Failed to set default values: {e}")

        def calculate_compression(self) -> None:
            """Perform compression calculations"""
            try:
                # Get input values
                composition = {
                    comp: self.composition_inputs[comp].value()
                    for comp in self.composition_inputs
                }

                flow_rate = self.flow_rate_input.value()
                inlet_temp = (
                    self.inlet_temp_input.value() + CELSIUS_TO_KELVIN_OFFSET
                )  # Convert to K
                self.inlet_pressure_input.value()
                compression_type = self.compression_type_combo.currentText().lower()
                intercooling = self.intercooling_checkbox.isChecked()

                # Create compression stages
                stages = []
                for i, stage_inputs in enumerate(self.stage_inputs):
                    if cast(QCheckBox, stage_inputs[3]).isChecked():  # Active stage
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

                # Start calculation in background thread
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
            """Handle calculation completion.

            Args:
                data: Dictionary containing calculation results and analysis.
            """
            if data is None:
                raise ValueError("data must be provided")
            result = data["result"]
            analysis = data["analysis"]

            # Display results
            self.display_results(result, analysis)
            self.display_analysis(analysis)
            self.create_plots(result)

            # Emit signal for parent
            self.calculation_finished.emit(data)

        @pyqtSlot(str)
        def on_calculation_error(self, error_message: str) -> None:
            """Handle calculation error.

            Args:
                error_message: Error message to display.
            """
            QMessageBox.critical(
                self,
                "Calculation Error",
                f"An error occurred: {error_message}",
            )

        def display_results(
            self, result: dict[str, Any], analysis: dict[str, Any]
        ) -> None:
            """Display calculation results.

            Args:
                result: Dictionary containing calculation results.
                analysis: Dictionary containing analysis data.
            """
            # Use list join for O(n) instead of O(n²) string concatenation
            if result is None:
                raise ValueError("result must be provided")
            output_parts = [
                "SYNGAS COMPRESSION CALCULATION RESULTS\n",
                "=" * 50 + "\n\n",
            ]

            # Mixture properties
            mix_props = result["mixture_properties"]
            output_parts.extend(
                [
                    "Mixture Properties:\n",
                    f"  Molecular Weight: {mix_props['molecular_weight']:.2f} g/mol\n",
                    f"  Critical Temperature: {mix_props['critical_temperature']:.1f} K\n",  # noqa: E501
                    f"  Critical Pressure: {mix_props['critical_pressure']:.1f} bar\n",
                    f"  Heat Capacity Ratio (γ): {mix_props['heat_capacity_ratio']:.3f}\n\n",  # noqa: E501
                    "Compression Stages:\n",
                    "-" * 30 + "\n",
                ]
            )

            # Stage-by-stage results
            for stage_result in result["stages"]:
                stage_num = stage_result["stage_number"]
                output_parts.extend(
                    [
                        f"\nStage {stage_num}:\n",
                        f"  Inlet Temperature: {stage_result['inlet_temp']:.1f} K "
                        f"({stage_result['inlet_temp'] - CELSIUS_TO_KELVIN_OFFSET:.1f} deg C)\n",  # noqa: E501
                        f"  Outlet Temperature: {stage_result['outlet_temp']:.1f} K "
                        f"({stage_result['outlet_temp'] - CELSIUS_TO_KELVIN_OFFSET:.1f} deg C)\n",  # noqa: E501
                        f"  Heat Rise: {stage_result['heat_rise']:.1f} K\n",
                        f"  Pressure Ratio: {stage_result['pressure_ratio']:.2f}\n",
                        f"  Power Required: {stage_result['power_hp']:.1f} HP\n",
                    ]
                )

                # Water dropout
                water_info = stage_result["water_dropout"]
                if water_info["water_dropout"] > ATOL_ZERO:
                    output_parts.extend(
                        [
                            f"  Water Dropout: {water_info['water_dropout']:.3f} mol%\n",  # noqa: E501
                            f"  Condensation Rate: {water_info['condensation_rate']:.1f}%\n",  # noqa: E501
                        ]
                    )

            # Summary
            output_parts.extend(
                [
                    "\nSUMMARY:\n",
                    "-" * 20 + "\n",
                    f"Total Power Required: {result['total_power_hp']:.1f} HP\n",
                    f"Final Temperature: {result['final_temperature']:.1f} K "
                    f"({result['final_temperature'] - CELSIUS_TO_KELVIN_OFFSET:.1f} deg C)\n",  # noqa: E501
                    f"Final Pressure: {result['final_pressure']:.1f} bar\n",
                    f"Total Water Dropout: {analysis['total_water_dropout']:.3f} mol%\n",  # noqa: E501
                ]
            )

            if analysis["average_efficiency"]:
                output_parts.append(
                    f"Average Efficiency: {analysis['average_efficiency'] * 100:.1f}%\n"
                )

            output = "".join(output_parts)
            self.results_text.setText(output)

        def display_analysis(self, analysis: dict[str, Any]) -> None:
            """Display analysis and concerns.

            Args:
                analysis: Dictionary containing analysis data and warnings.
            """
            # Use list join for O(n) instead of O(n²) string concatenation
            if analysis is None:
                raise ValueError("analysis must be provided")
            output_parts = [
                "PROCESS ANALYSIS & CONCERNS\n",
                "=" * 40 + "\n\n",
            ]

            if analysis["warnings"]:
                output_parts.extend(
                    [
                        "⚠️  CRITICAL WARNINGS:\n",
                        "-" * 25 + "\n",
                    ]
                )
                output_parts.extend(
                    [f"• {warning}\n" for warning in analysis["warnings"]]
                )
                output_parts.append("\n")

            if analysis["concerns"]:
                output_parts.extend(
                    [
                        "⚠️  CONCERNS:\n",
                        "-" * 15 + "\n",
                    ]
                )
                output_parts.extend(
                    [f"• {concern}\n" for concern in analysis["concerns"]]
                )
                output_parts.append("\n")

            if analysis["recommendations"]:
                output_parts.extend(
                    [
                        "💡 RECOMMENDATIONS:\n",
                        "-" * 20 + "\n",
                    ]
                )
                output_parts.extend(
                    [f"• {rec}\n" for rec in analysis["recommendations"]]
                )
                output_parts.append("\n")

            if not analysis["warnings"] and not analysis["concerns"]:
                output_parts.extend(
                    [
                        "✅ No significant concerns detected.\n",
                        "Process conditions appear to be within acceptable limits.\n",
                    ]
                )

            output = "".join(output_parts)
            self.analysis_text.setText(output)

        def create_plots(self, result: dict[str, Any]) -> None:
            """Create visualization plots"""
            # Clear previous plots
            if result is None:
                raise ValueError("result must be provided")
            self.figure.clear()

            stages = result["stages"]
            stage_nums = [s["stage_number"] for s in stages]
            temperatures = [
                s["outlet_temp"] - CELSIUS_TO_KELVIN_OFFSET for s in stages
            ]  # Convert to deg C
            pressures = [s["pressure_ratio"] for s in stages]
            powers = [s["power_hp"] for s in stages]
            water_dropouts = [s["water_dropout"]["water_dropout"] for s in stages]

            # Create subplots
            ax1 = self.figure.add_subplot(2, 2, 1)
            ax2 = self.figure.add_subplot(2, 2, 2)
            ax3 = self.figure.add_subplot(2, 2, 3)
            ax4 = self.figure.add_subplot(2, 2, 4)

            # Temperature profile
            ax1.plot(stage_nums, temperatures, "bo-", linewidth=2, markersize=8)
            ax1.set_xlabel("Compression Stage")
            ax1.set_ylabel("Temperature (°C)")
            ax1.set_title("Temperature Profile")
            ax1.grid(True, alpha=0.3)

            # Pressure ratio
            ax2.bar(stage_nums, pressures, alpha=0.7, color="green")
            ax2.set_xlabel("Compression Stage")
            ax2.set_ylabel("Pressure Ratio")
            ax2.set_title("Pressure Ratio per Stage")
            ax2.grid(True, alpha=0.3)

            # Power requirement
            ax3.bar(stage_nums, powers, alpha=0.7, color="orange")
            ax3.set_xlabel("Compression Stage")
            ax3.set_ylabel("Power (HP)")
            ax3.set_title("Power Requirement per Stage")
            ax3.grid(True, alpha=0.3)

            # Water dropout
            ax4.bar(stage_nums, water_dropouts, alpha=0.7, color="blue")
            ax4.set_xlabel("Compression Stage")
            ax4.set_ylabel("Water Dropout (mol%)")
            ax4.set_title("Water Dropout per Stage")
            ax4.grid(True, alpha=0.3)

            # Adjust layout
            self.figure.tight_layout()
            self.canvas.draw()


def create_syngas_compression_calculator(parent: Any = None) -> QWidget:
    """Factory function to create syngas compression calculator widget"""
    return SyngasCompressionCalculatorWidget(parent=parent)
