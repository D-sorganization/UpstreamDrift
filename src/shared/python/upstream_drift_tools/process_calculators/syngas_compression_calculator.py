"""Advanced Syngas Compression Calculator widget.

Core orchestration widget; computation is delegated to
:mod:`syngas_compression_engine`, tab layout to
:mod:`syngas_compression_tabs_mixin`, and result formatting to
:mod:`syngas_compression_display`.

Integrated with the existing PyQt6-based calculator system.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

import matplotlib as mpl

try:
    from PyQt6.QtCore import QTimer, pyqtSignal, pyqtSlot
    from PyQt6.QtWidgets import (
        QCheckBox,
        QDoubleSpinBox,
        QLabel,
        QMessageBox,
        QSplitter,
        QTableWidget,
        QTabWidget,
        QTextEdit,
        QWidget,
    )

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QWidget = object  # type: ignore[assignment,misc]

try:
    from integrated_process_simulator.utilities.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

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

try:
    from ..ui.widgets.base_calculator_widget import BaseCalculatorWidget

    BASE_CALCULATOR_AVAILABLE = True
except ImportError:
    BASE_CALCULATOR_AVAILABLE = False

    class BaseCalculatorWidget(QWidget):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            QWidget.__init__(self, *args, **kwargs)


from .constants import CELSIUS_TO_KELVIN_OFFSET, INTERCOOLER_OUTLET_TEMP_K
from .syngas_compression_display import (
    format_analysis_text,
    format_results_text,
    render_compression_plots,
)
from .syngas_compression_engine import CompressionStage, SyngasCompressionEngine

if HAS_PYQT:
    from .syngas_compression_tabs_mixin import _SyngasTabsMixin
    from .syngas_compression_worker import CompressionCalculationWorker

    BaseClass = BaseCalculatorWidget if BASE_CALCULATOR_AVAILABLE else QWidget

    class SyngasCompressionCalculatorWidget(  # type: ignore[valid-type, misc]
        _SyngasTabsMixin,
        BaseClass,  # type: ignore[misc,valid-type]
    ):
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
            """Register widgets for state persistence."""
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
            """Save state on close."""
            self.save_state()
            super().closeEvent(event)

        def showEvent(self, event: Any) -> None:
            """Refresh layout when shown."""
            super().showEvent(event)
            QTimer.singleShot(50, self._refresh_layout)

        def _refresh_layout(self) -> None:
            """Fix visibility issues when widget is dynamically added to tabs."""
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
            from PyQt6.QtWidgets import QVBoxLayout

            layout = QVBoxLayout()
            self.tab_widget = QTabWidget()
            self.create_input_tab()
            self.create_results_tab()
            self.create_analysis_tab()
            self.create_plots_tab()
            layout.addWidget(self.tab_widget)
            self.setLayout(layout)

        def set_default_values(self) -> None:
            """Set default values for the calculator."""
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
            """Perform compression calculations in a background thread."""
            try:
                composition = {
                    comp: self.composition_inputs[comp].value()
                    for comp in self.composition_inputs
                }
                flow_rate = self.flow_rate_input.value()
                inlet_temp = self.inlet_temp_input.value() + CELSIUS_TO_KELVIN_OFFSET
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
                        self, "Error", "No valid compression stages defined"
                    )
                    return

                self.worker = CompressionCalculationWorker(
                    self.engine, stages, flow_rate, composition, intercooling
                )
                self.worker.finished.connect(self.on_calculation_finished)
                self.worker.error.connect(self.on_calculation_error)
                self.worker.start()

            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                QMessageBox.critical(
                    self, "Calculation Error", f"An error occurred: {e!s}"
                )

        @pyqtSlot(dict)
        def on_calculation_finished(self, data: dict[str, Any]) -> None:
            """Handle calculation completion."""
            if not (data is not None):
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
                self, "Calculation Error", f"An error occurred: {error_message}"
            )

        def display_results(
            self, result: dict[str, Any], analysis: dict[str, Any]
        ) -> None:
            """Display calculation results."""
            self.results_text.setText(format_results_text(result, analysis))

        def display_analysis(self, analysis: dict[str, Any]) -> None:
            """Display analysis and concerns."""
            self.analysis_text.setText(format_analysis_text(analysis))

        def create_plots(self, result: dict[str, Any]) -> None:
            """Create visualization plots."""
            render_compression_plots(self.figure, self.canvas, result)


def create_syngas_compression_calculator(parent: Any = None) -> Any:
    """Factory: return a SyngasCompressionCalculatorWidget or None if no Qt."""
    if not HAS_PYQT:
        return None
    return SyngasCompressionCalculatorWidget(parent)
