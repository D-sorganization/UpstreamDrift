from __future__ import annotations

from typing import Any

try:
    from PyQt6.QtCore import QTimer, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QLabel,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

try:
    from ..ui.widgets.base_calculator_widget import BaseCalculatorWidget

    BASE_CALCULATOR_AVAILABLE = True
except ImportError:
    BASE_CALCULATOR_AVAILABLE = False

    if GUI_AVAILABLE:

        class BaseCalculatorWidget(QWidget):  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                QWidget.__init__(self, *args, **kwargs)


from ._acid_gas_calculator import AcidGasDewpointCalculator
from ._acid_gas_models import AcidGasComposition, DewpointResult

if GUI_AVAILABLE:
    BaseClass = BaseCalculatorWidget if BASE_CALCULATOR_AVAILABLE else QWidget

    class AcidGasDewpointCalculatorWidget(BaseClass):  # type: ignore[valid-type, misc]
        """Acid gas dewpoint calculator widget"""

        calculation_completed = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            """Initialize the class."""
            if BASE_CALCULATOR_AVAILABLE:
                super().__init__(calculator_name="AcidGasDewpoint", parent=parent)
            else:
                super().__init__(parent)

            self.calculator = AcidGasDewpointCalculator()
            self.current_result = None
            self.setup_ui()
            self.setup_connections()
            self.set_default_values()

            if BASE_CALCULATOR_AVAILABLE:
                QTimer.singleShot(0, self.setup_state_management)

        def setup_connections(self) -> None:
            """Setup signal connections"""

        def set_default_values(self) -> None:
            """Set default values for input widgets"""

        def setup_state_management(self) -> None:
            """Setup state management for the calculator"""
            if not BASE_CALCULATOR_AVAILABLE:
                return

            # Find and register splitters
            for child_splitter in self.findChildren(QSplitter):
                self.register_splitter(child_splitter, "main_splitter")

            # Register result widgets for copy/paste
            for child_label in self.findChildren((QLabel, QTextEdit, QTableWidget)):
                if hasattr(child_label, "text") and child_label.text().strip():
                    self.register_copyable_widget(child_label, "label")
            for child_table in self.findChildren(QTableWidget):
                self.register_copyable_widget(child_table, "table")
            for child_text in self.findChildren(QTextEdit):
                self.register_copyable_widget(child_text, "text")

        def closeEvent(self, event: Any) -> None:
            """Save state when tab is closed"""
            if BASE_CALCULATOR_AVAILABLE:
                self.save_state()
            super().closeEvent(event)

        def setup_ui(self) -> None:
            """Setup the user interface"""
            layout = QVBoxLayout(self)
            title = QLabel("Acid Gas Dewpoint Calculator")
            title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            layout.addWidget(title)

            # Input fields
            input_group = QGroupBox("Input Conditions")
            input_layout = QGridLayout(input_group)
            self.temp_input = QDoubleSpinBox()
            self.temp_input.setRange(-100, 400)
            self.temp_input.setValue(150)
            self.temp_input.setSuffix(" \u00b0C")
            input_layout.addWidget(QLabel("Temperature:"), 0, 0)
            input_layout.addWidget(self.temp_input, 0, 1)

            self.pressure_input = QDoubleSpinBox()
            self.pressure_input.setRange(0.1, 300)
            self.pressure_input.setValue(30)
            self.pressure_input.setSuffix(" bar")
            input_layout.addWidget(QLabel("Pressure:"), 1, 0)
            input_layout.addWidget(self.pressure_input, 1, 1)

            # Acid gas composition
            self.h2o_input = QDoubleSpinBox()
            self.h2o_input.setRange(0, 1)
            self.h2o_input.setValue(0.15)
            self.hf_input = QDoubleSpinBox()
            self.hf_input.setRange(0, 1)
            self.hf_input.setValue(0.001)
            self.hcl_input = QDoubleSpinBox()
            self.hcl_input.setRange(0, 1)
            self.hcl_input.setValue(0.002)
            self.h2s_input = QDoubleSpinBox()
            self.h2s_input.setRange(0, 1)
            self.h2s_input.setValue(0.005)

            input_layout.addWidget(QLabel("H2O mole fraction:"), 2, 0)
            input_layout.addWidget(self.h2o_input, 2, 1)
            input_layout.addWidget(QLabel("HF mole fraction:"), 3, 0)
            input_layout.addWidget(self.hf_input, 3, 1)
            input_layout.addWidget(QLabel("HCl mole fraction:"), 4, 0)
            input_layout.addWidget(self.hcl_input, 4, 1)
            input_layout.addWidget(QLabel("H2S mole fraction:"), 5, 0)
            input_layout.addWidget(self.h2s_input, 5, 1)

            layout.addWidget(input_group)

            # Calculate button
            self.calc_btn = QPushButton("Calculate Dewpoint")
            self.calc_btn.clicked.connect(self.calculate)
            layout.addWidget(self.calc_btn)

            # Output area
            self.result_area = QTextEdit()
            self.result_area.setReadOnly(True)
            layout.addWidget(self.result_area)

        def calculate(self) -> None:
            """Collect inputs and run calculation."""
            temp = self.temp_input.value()
            pressure = self.pressure_input.value()
            comp = AcidGasComposition(
                h2o=self.h2o_input.value(),
                hf=self.hf_input.value(),
                hcl=self.hcl_input.value(),
                h2s=self.h2s_input.value(),
            )
            result = self.calculator.calculate_dewpoint_mixture(temp, pressure, comp)
            self.display_result(result)

        def display_result(self, result: DewpointResult) -> None:
            """Format and display results in the UI."""
            assert result is not None, "result must be provided"
            assert result is not None, "result must be provided"
            text = (
                f"<b>Input:</b> T = {result.temperature_c:.2f} \u00b0C, "
                f"P = {result.pressure_bar:.2f} bar<br>"
            )
            text += (
                f"<b>Composition:</b> H2O={result.composition.h2o:.4f}, "
                f"HF={result.composition.hf:.4f}, "
                f"HCl={result.composition.hcl:.4f}, "
                f"H2S={result.composition.h2s:.4f}<br>"
            )
            text += (
                f"<b>Dewpoints (\u00b0C):</b> H2O={result.h2o_dewpoint_c:.2f}, "
                f"HF={result.hf_dewpoint_c:.2f}, HCl={result.hcl_dewpoint_c:.2f}, "
                f"H2S={result.h2s_dewpoint_c:.2f}<br>"
            )
            text += (
                f"<b>Overall Dewpoint:</b> {result.overall_dewpoint_c:.2f} \u00b0C "
                f"({result.limiting_component})<br>"
            )
            text += (
                f"<b>Dewpoint Margin:</b> {result.dewpoint_margin_c:.2f} \u00b0C<br>"
            )
            text += f"<b>Condensation Risk:</b> {result.condensation_risk}<br>"
            if result.warnings:
                text += f"<b>Warnings:</b> {'; '.join(result.warnings)}<br>"
            self.result_area.setHtml(text)
