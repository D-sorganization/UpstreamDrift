"""PyQt6 GUI Widget for Signal Processing Toolkit.

This module provides a comprehensive visual interface for signal generation,
fitting, filtering, and analysis. It can be used standalone or integrated
into other applications.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PyQt6 import QtWidgets
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

from src.shared.python.signal_toolkit.calculus import (
    Differentiator,
    Integrator,
    compute_tangent_line,
)
from src.shared.python.signal_toolkit.core import Signal, SignalGenerator
from src.shared.python.signal_toolkit.filters import (
    FilterDesigner,
    FilterType,
    apply_filter,
    apply_moving_average,
    apply_savgol,
)
from src.shared.python.signal_toolkit.fitting import FunctionFitter
from src.shared.python.signal_toolkit.io import SignalExporter, SignalImporter
from src.shared.python.signal_toolkit.limits import SaturationMode, apply_saturation
from src.shared.python.signal_toolkit.noise import NoiseType, add_noise_to_signal

# Dark theme stylesheet
DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
QGroupBox {
    border: 1px solid #444;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
    color: #e0e0e0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 12px;
    color: #fff;
}
QPushButton:hover {
    background-color: #4d4d4d;
    border: 1px solid #666;
}
QPushButton:pressed {
    background-color: #2b2b2b;
}
QPushButton:disabled {
    background-color: #333;
    color: #666;
}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background-color: #1e1e1e;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 4px;
    color: #fff;
}
QComboBox::drop-down {
    border: none;
}
QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #555;
    border-radius: 4px;
    color: #fff;
}
QTabWidget::pane {
    border: 1px solid #444;
    background: #2b2b2b;
}
QTabBar::tab {
    background: #333;
    color: #ccc;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #0078d4;
    color: white;
}
QTabBar::tab:hover:!selected {
    background: #444;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #444;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -5px 0;
    background: #0078d4;
    border-radius: 8px;
}
QScrollArea {
    border: none;
}
QLabel {
    color: #cccccc;
}
"""


if HAS_MATPLOTLIB and HAS_PYQT:

    class MplCanvas(FigureCanvasQTAgg):
        """Matplotlib canvas for PyQt6."""

        def __init__(
            self,
            parent: QWidget | None = None,
            width: float = 5,
            height: float = 4,
            dpi: int = 100,
        ) -> None:
            """Initialize the canvas."""
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
            super().__init__(self.fig)
            self.setParent(parent)
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            self.updateGeometry()

        def setup_dark_theme(self) -> None:
            """Apply dark theme to the plot."""
            self.fig.patch.set_facecolor("#2b2b2b")
            self.axes.set_facecolor("#1e1e1e")
            self.axes.tick_params(colors="#aaaaaa", which="both")
            self.axes.xaxis.label.set_color("#aaaaaa")
            self.axes.yaxis.label.set_color("#aaaaaa")
            self.axes.title.set_color("#ffffff")
            for spine in self.axes.spines.values():
                spine.set_edgecolor("#555555")
            self.axes.grid(True, color="#444444", linestyle="--", linewidth=0.5)

    class SignalToolkitWidget(QWidget):
        """Comprehensive signal processing toolkit widget."""

        # Signals
        signal_generated = pyqtSignal(str, list)  # joint_name, coefficients
        signal_updated = pyqtSignal(object)  # Signal object

        def __init__(self, parent: QWidget | None = None) -> None:
            """Initialize the widget."""
            super().__init__(parent)

            self.setWindowTitle("Signal Processing Toolkit")
            self.resize(1200, 800)
            self.setStyleSheet(DARK_STYLESHEET)

            # State
            self.current_signal: Signal | None = None
            self.original_signal: Signal | None = None
            self.derivative_signal: Signal | None = None
            self.integral_signal: Signal | None = None
            self.joint_names: list[str] = []

            # Default time array
            self.t_default = np.linspace(0, 10, 1000)

            # Setup UI
            self._setup_ui()
            self._setup_connections()

            # Initialize with a default signal
            self._generate_default_signal()

        def _setup_ui(self) -> None:
            """Setup the user interface."""
            main_layout = QHBoxLayout(self)
            main_layout.setContentsMargins(10, 10, 10, 10)

            # Splitter for resizable panels
            splitter = QSplitter(Qt.Orientation.Horizontal)

            # Left Panel: Controls
            left_panel = self._create_left_panel()
            splitter.addWidget(left_panel)

            # Right Panel: Plots
            right_panel = self._create_right_panel()
            splitter.addWidget(right_panel)

            splitter.setSizes([400, 800])
            main_layout.addWidget(splitter)

        def _create_left_panel(self) -> QWidget:
            """Create the left control panel."""
            panel = QWidget()
            panel.setMinimumWidth(350)
            panel.setMaximumWidth(500)

            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)

            # Scroll area
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

            content = QWidget()
            content_layout = QVBoxLayout(content)
            content_layout.setSpacing(10)

            # Tab widget for organized controls
            tabs = QTabWidget()

            # Generation tab
            tabs.addTab(self._create_generation_tab(), "Generate")

            # Fitting tab
            tabs.addTab(self._create_fitting_tab(), "Fit")

            # Limits tab
            tabs.addTab(self._create_limits_tab(), "Limits")

            # Calculus tab
            tabs.addTab(self._create_calculus_tab(), "Calculus")

            # Filters tab
            tabs.addTab(self._create_filters_tab(), "Filters")

            # Noise tab
            tabs.addTab(self._create_noise_tab(), "Noise")

            # Import tab
            tabs.addTab(self._create_import_tab(), "Import")

            content_layout.addWidget(tabs)

            # Joint selector and apply button
            joint_group = QGroupBox("Output")
            joint_layout = QVBoxLayout(joint_group)

            joint_row = QHBoxLayout()
            joint_row.addWidget(QLabel("Joint:"))
            self.joint_combo = QComboBox()
            self.joint_combo.addItems(["Joint 1", "Joint 2", "Joint 3"])
            joint_row.addWidget(self.joint_combo, 1)
            joint_layout.addLayout(joint_row)

            self.apply_btn = QPushButton("Apply to Joint")
            self.apply_btn.setStyleSheet(
                "background-color: #107c10; font-weight: bold;"
            )
            joint_layout.addWidget(self.apply_btn)

            self.export_btn = QPushButton("Export Signal")
            joint_layout.addWidget(self.export_btn)

            content_layout.addWidget(joint_group)

            # Results display
            result_group = QGroupBox("Info")
            result_layout = QVBoxLayout(result_group)
            self.result_text = QTextEdit()
            self.result_text.setReadOnly(True)
            self.result_text.setMaximumHeight(100)
            result_layout.addWidget(self.result_text)
            content_layout.addWidget(result_group)

            scroll.setWidget(content)
            layout.addWidget(scroll)

            return panel

        def _create_generation_tab(self) -> QWidget:
            """Create signal generation controls."""
            tab = QWidget()
            layout = QVBoxLayout(tab)

            # Signal type selector
            type_group = QGroupBox("Signal Type")
            type_layout = QVBoxLayout(type_group)

            self.signal_type_combo = QComboBox()
            self.signal_type_combo.addItems(
                [
                    "Sinusoid",
                    "Cosine",
                    "Polynomial",
                    "Exponential",
                    "Linear",
                    "Step",
                    "Chirp",
                    "Square",
                    "Triangle",
                    "Custom",
                ]
            )
            type_layout.addWidget(self.signal_type_combo)

            layout.addWidget(type_group)

            # Parameters stack
            self.param_stack = QStackedWidget()

            # Sinusoid parameters
            sin_widget = self._create_sinusoid_params()
            self.param_stack.addWidget(sin_widget)

            # Cosine (same as sinusoid)
            cos_widget = self._create_sinusoid_params()
            self.param_stack.addWidget(cos_widget)

            # Polynomial parameters
            poly_widget = self._create_polynomial_params()
            self.param_stack.addWidget(poly_widget)

            # Exponential parameters
            exp_widget = self._create_exponential_params()
            self.param_stack.addWidget(exp_widget)

            # Linear parameters
            linear_widget = self._create_linear_params()
            self.param_stack.addWidget(linear_widget)

            # Step parameters
            step_widget = self._create_step_params()
            self.param_stack.addWidget(step_widget)

            # Chirp parameters
            chirp_widget = self._create_chirp_params()
            self.param_stack.addWidget(chirp_widget)

            # Square parameters
            square_widget = self._create_square_params()
            self.param_stack.addWidget(square_widget)

            # Triangle parameters
            triangle_widget = self._create_triangle_params()
            self.param_stack.addWidget(triangle_widget)

            # Custom expression
            custom_widget = self._create_custom_params()
            self.param_stack.addWidget(custom_widget)

            layout.addWidget(self.param_stack)

            # Time range
            time_group = QGroupBox("Time Range")
            time_layout = QGridLayout(time_group)

            time_layout.addWidget(QLabel("Start:"), 0, 0)
            self.t_start_spin = QDoubleSpinBox()
            self.t_start_spin.setRange(-1000, 1000)
            self.t_start_spin.setValue(0)
            time_layout.addWidget(self.t_start_spin, 0, 1)

            time_layout.addWidget(QLabel("End:"), 0, 2)
            self.t_end_spin = QDoubleSpinBox()
            self.t_end_spin.setRange(-1000, 1000)
            self.t_end_spin.setValue(10)
            time_layout.addWidget(self.t_end_spin, 0, 3)

            time_layout.addWidget(QLabel("Points:"), 1, 0)
            self.n_points_spin = QSpinBox()
            self.n_points_spin.setRange(10, 10000)
            self.n_points_spin.setValue(1000)
            time_layout.addWidget(self.n_points_spin, 1, 1)

            layout.addWidget(time_group)

            # Generate button
            self.generate_btn = QPushButton("Generate Signal")
            self.generate_btn.setStyleSheet(
                "background-color: #0078d4; font-weight: bold;"
            )
            layout.addWidget(self.generate_btn)

            layout.addStretch()
            return tab

        def _create_sinusoid_params(self) -> QWidget:
            """Create sinusoid parameter controls."""
            widget = QWidget()
            layout = QGridLayout(widget)

            layout.addWidget(QLabel("Amplitude:"), 0, 0)
            self.sin_amplitude = QDoubleSpinBox()
            self.sin_amplitude.setRange(-1000, 1000)
            self.sin_amplitude.setValue(1.0)
            self.sin_amplitude.setDecimals(3)
            layout.addWidget(self.sin_amplitude, 0, 1)

            layout.addWidget(QLabel("Frequency (Hz):"), 1, 0)
            self.sin_frequency = QDoubleSpinBox()
            self.sin_frequency.setRange(0.001, 1000)
            self.sin_frequency.setValue(1.0)
            self.sin_frequency.setDecimals(3)
            layout.addWidget(self.sin_frequency, 1, 1)

            layout.addWidget(QLabel("Phase (rad):"), 2, 0)
            self.sin_phase = QDoubleSpinBox()
            self.sin_phase.setRange(-6.28, 6.28)
            self.sin_phase.setValue(0.0)
            self.sin_phase.setDecimals(3)
            layout.addWidget(self.sin_phase, 2, 1)

            layout.addWidget(QLabel("Offset:"), 3, 0)
            self.sin_offset = QDoubleSpinBox()
            self.sin_offset.setRange(-1000, 1000)
            self.sin_offset.setValue(0.0)
            self.sin_offset.setDecimals(3)
            layout.addWidget(self.sin_offset, 3, 1)

            return widget

        def _create_polynomial_params(self) -> QWidget:
            """Create polynomial parameter controls."""
            widget = QWidget()
            layout = QVBoxLayout(widget)

            layout.addWidget(QLabel("Coefficients (c0, c1, c2, ...):"))
            self.poly_coeffs_input = QLineEdit()
            self.poly_coeffs_input.setPlaceholderText("0, 0, 1")  # x^2
            self.poly_coeffs_input.setText("0, 1, 0.5")
            layout.addWidget(self.poly_coeffs_input)

            layout.addWidget(QLabel("Order:"))
            self.poly_order_spin = QSpinBox()
            self.poly_order_spin.setRange(1, 10)
            self.poly_order_spin.setValue(6)
            layout.addWidget(self.poly_order_spin)

            return widget

        def _create_exponential_params(self) -> QWidget:
            """Create exponential parameter controls."""
            widget = QWidget()
            layout = QGridLayout(widget)

            layout.addWidget(QLabel("Amplitude:"), 0, 0)
            self.exp_amplitude = QDoubleSpinBox()
            self.exp_amplitude.setRange(-1000, 1000)
            self.exp_amplitude.setValue(1.0)
            layout.addWidget(self.exp_amplitude, 0, 1)

            layout.addWidget(QLabel("Decay Rate:"), 1, 0)
            self.exp_decay = QDoubleSpinBox()
            self.exp_decay.setRange(-100, 100)
            self.exp_decay.setValue(0.5)
            self.exp_decay.setDecimals(3)
            layout.addWidget(self.exp_decay, 1, 1)

            layout.addWidget(QLabel("Offset:"), 2, 0)
            self.exp_offset = QDoubleSpinBox()
            self.exp_offset.setRange(-1000, 1000)
            self.exp_offset.setValue(0.0)
            layout.addWidget(self.exp_offset, 2, 1)

            return widget

        def _create_linear_params(self) -> QWidget:
            """Create linear parameter controls."""
            widget = QWidget()
            layout = QGridLayout(widget)

            layout.addWidget(QLabel("Slope:"), 0, 0)
            self.linear_slope = QDoubleSpinBox()
            self.linear_slope.setRange(-1000, 1000)
            self.linear_slope.setValue(1.0)
            layout.addWidget(self.linear_slope, 0, 1)

            layout.addWidget(QLabel("Intercept:"), 1, 0)
            self.linear_intercept = QDoubleSpinBox()
            self.linear_intercept.setRange(-1000, 1000)
            self.linear_intercept.setValue(0.0)
            layout.addWidget(self.linear_intercept, 1, 1)

            return widget

        def _create_step_params(self) -> QWidget:
            """Create step parameter controls."""
            widget = QWidget()
            layout = QGridLayout(widget)

            layout.addWidget(QLabel("Step Time:"), 0, 0)
            self.step_time = QDoubleSpinBox()
            self.step_time.setRange(-1000, 1000)
            self.step_time.setValue(5.0)
            layout.addWidget(self.step_time, 0, 1)

            layout.addWidget(QLabel("Step Value:"), 1, 0)
            self.step_value = QDoubleSpinBox()
            self.step_value.setRange(-1000, 1000)
            self.step_value.setValue(1.0)
            layout.addWidget(self.step_value, 1, 1)

            layout.addWidget(QLabel("Initial Value:"), 2, 0)
            self.step_initial = QDoubleSpinBox()
            self.step_initial.setRange(-1000, 1000)
            self.step_initial.setValue(0.0)
            layout.addWidget(self.step_initial, 2, 1)

            return widget

        def _create_chirp_params(self) -> QWidget:
            """Create chirp parameter controls."""
            widget = QWidget()
            layout = QGridLayout(widget)

            layout.addWidget(QLabel("Start Freq (Hz):"), 0, 0)
            self.chirp_f0 = QDoubleSpinBox()
            self.chirp_f0.setRange(0.001, 1000)
            self.chirp_f0.setValue(0.5)
            layout.addWidget(self.chirp_f0, 0, 1)

            layout.addWidget(QLabel("End Freq (Hz):"), 1, 0)
            self.chirp_f1 = QDoubleSpinBox()
            self.chirp_f1.setRange(0.001, 1000)
            self.chirp_f1.setValue(5.0)
            layout.addWidget(self.chirp_f1, 1, 1)

            layout.addWidget(QLabel("Amplitude:"), 2, 0)
            self.chirp_amplitude = QDoubleSpinBox()
            self.chirp_amplitude.setRange(-1000, 1000)
            self.chirp_amplitude.setValue(1.0)
            layout.addWidget(self.chirp_amplitude, 2, 1)

            return widget

        def _create_square_params(self) -> QWidget:
            """Create square wave parameter controls."""
            widget = QWidget()
            layout = QGridLayout(widget)

            layout.addWidget(QLabel("Frequency (Hz):"), 0, 0)
            self.square_freq = QDoubleSpinBox()
            self.square_freq.setRange(0.001, 1000)
            self.square_freq.setValue(1.0)
            layout.addWidget(self.square_freq, 0, 1)

            layout.addWidget(QLabel("Amplitude:"), 1, 0)
            self.square_amplitude = QDoubleSpinBox()
            self.square_amplitude.setRange(-1000, 1000)
            self.square_amplitude.setValue(1.0)
            layout.addWidget(self.square_amplitude, 1, 1)

            layout.addWidget(QLabel("Duty Cycle:"), 2, 0)
            self.square_duty = QDoubleSpinBox()
            self.square_duty.setRange(0.01, 0.99)
            self.square_duty.setValue(0.5)
            layout.addWidget(self.square_duty, 2, 1)

            return widget

        def _create_triangle_params(self) -> QWidget:
            """Create triangle wave parameter controls."""
            widget = QWidget()
            layout = QGridLayout(widget)

            layout.addWidget(QLabel("Frequency (Hz):"), 0, 0)
            self.triangle_freq = QDoubleSpinBox()
            self.triangle_freq.setRange(0.001, 1000)
            self.triangle_freq.setValue(1.0)
            layout.addWidget(self.triangle_freq, 0, 1)

            layout.addWidget(QLabel("Amplitude:"), 1, 0)
            self.triangle_amplitude = QDoubleSpinBox()
            self.triangle_amplitude.setRange(-1000, 1000)
            self.triangle_amplitude.setValue(1.0)
            layout.addWidget(self.triangle_amplitude, 1, 1)

            return widget

        def _create_custom_params(self) -> QWidget:
            """Create custom expression controls."""
            widget = QWidget()
            layout = QVBoxLayout(widget)

            layout.addWidget(QLabel("Expression (use 't' for time):"))
            self.custom_expr = QLineEdit()
            self.custom_expr.setPlaceholderText("sin(2*pi*t) + 0.5*cos(4*pi*t)")
            layout.addWidget(self.custom_expr)

            layout.addWidget(QLabel("Available: sin, cos, exp, log, sqrt, pi"))

            return widget

        def _create_fitting_tab(self) -> QWidget:
            """Create function fitting controls."""
            tab = QWidget()
            layout = QVBoxLayout(tab)

            # Fit type selector
            fit_group = QGroupBox("Fit Type")
            fit_layout = QVBoxLayout(fit_group)

            self.fit_type_combo = QComboBox()
            self.fit_type_combo.addItems(
                [
                    "Sinusoid",
                    "Cosine",
                    "Exponential Decay",
                    "Exponential Growth",
                    "Linear",
                    "Polynomial",
                    "Custom",
                ]
            )
            fit_layout.addWidget(self.fit_type_combo)

            # Polynomial order for polynomial fit
            order_row = QHBoxLayout()
            order_row.addWidget(QLabel("Poly Order:"))
            self.fit_poly_order = QSpinBox()
            self.fit_poly_order.setRange(1, 10)
            self.fit_poly_order.setValue(6)
            order_row.addWidget(self.fit_poly_order)
            fit_layout.addLayout(order_row)

            layout.addWidget(fit_group)

            # Custom fit expression
            custom_group = QGroupBox("Custom Fit (optional)")
            custom_layout = QVBoxLayout(custom_group)
            self.fit_custom_expr = QLineEdit()
            self.fit_custom_expr.setPlaceholderText("a*sin(b*t) + c")
            custom_layout.addWidget(self.fit_custom_expr)
            self.fit_custom_params = QLineEdit()
            self.fit_custom_params.setPlaceholderText("a, b, c")
            custom_layout.addWidget(self.fit_custom_params)
            layout.addWidget(custom_group)

            # Fit button
            self.fit_btn = QPushButton("Fit Function")
            self.fit_btn.setStyleSheet("background-color: #0078d4; font-weight: bold;")
            layout.addWidget(self.fit_btn)

            # Auto-fit button
            self.auto_fit_btn = QPushButton("Auto-Detect Best Fit")
            layout.addWidget(self.auto_fit_btn)

            layout.addStretch()
            return tab

        def _create_limits_tab(self) -> QWidget:
            """Create limit/saturation controls."""
            tab = QWidget()
            layout = QVBoxLayout(tab)

            # Saturation controls
            sat_group = QGroupBox("Saturation")
            sat_layout = QGridLayout(sat_group)

            sat_layout.addWidget(QLabel("Lower Limit:"), 0, 0)
            self.sat_lower = QDoubleSpinBox()
            self.sat_lower.setRange(-1000, 1000)
            self.sat_lower.setValue(-1.0)
            sat_layout.addWidget(self.sat_lower, 0, 1)

            sat_layout.addWidget(QLabel("Upper Limit:"), 1, 0)
            self.sat_upper = QDoubleSpinBox()
            self.sat_upper.setRange(-1000, 1000)
            self.sat_upper.setValue(1.0)
            sat_layout.addWidget(self.sat_upper, 1, 1)

            sat_layout.addWidget(QLabel("Mode:"), 2, 0)
            self.sat_mode_combo = QComboBox()
            self.sat_mode_combo.addItems(
                [
                    "Hard",
                    "Soft",
                    "Tanh",
                    "Sigmoid",
                    "Atan",
                    "Cubic",
                    "Exponential",
                ]
            )
            self.sat_mode_combo.setCurrentIndex(2)  # Tanh default
            sat_layout.addWidget(self.sat_mode_combo, 2, 1)

            sat_layout.addWidget(QLabel("Smoothness:"), 3, 0)
            self.sat_smoothness = QDoubleSpinBox()
            self.sat_smoothness.setRange(0.1, 100)
            self.sat_smoothness.setValue(1.0)
            sat_layout.addWidget(self.sat_smoothness, 3, 1)

            layout.addWidget(sat_group)

            # Apply saturation button
            self.apply_sat_btn = QPushButton("Apply Saturation")
            self.apply_sat_btn.setStyleSheet(
                "background-color: #0078d4; font-weight: bold;"
            )
            layout.addWidget(self.apply_sat_btn)

            # Preview checkbox
            self.sat_preview_check = QCheckBox("Live Preview")
            self.sat_preview_check.setChecked(True)
            layout.addWidget(self.sat_preview_check)

            layout.addStretch()
            return tab

        def _create_calculus_tab(self) -> QWidget:
            """Create differentiation and integration controls."""
            tab = QWidget()
            layout = QVBoxLayout(tab)

            # Differentiation group
            diff_group = QGroupBox("Differentiation")
            diff_layout = QVBoxLayout(diff_group)

            order_row = QHBoxLayout()
            order_row.addWidget(QLabel("Order:"))
            self.diff_order = QSpinBox()
            self.diff_order.setRange(1, 4)
            self.diff_order.setValue(1)
            order_row.addWidget(self.diff_order)
            diff_layout.addLayout(order_row)

            self.show_derivative_btn = QPushButton("Show Derivative")
            diff_layout.addWidget(self.show_derivative_btn)

            # Tangent line controls
            tangent_row = QHBoxLayout()
            tangent_row.addWidget(QLabel("Tangent at t="))
            self.tangent_t_spin = QDoubleSpinBox()
            self.tangent_t_spin.setRange(-1000, 1000)
            self.tangent_t_spin.setValue(5.0)
            tangent_row.addWidget(self.tangent_t_spin)
            diff_layout.addLayout(tangent_row)

            # Tangent slider
            self.tangent_slider = QSlider(Qt.Orientation.Horizontal)
            self.tangent_slider.setRange(0, 100)
            self.tangent_slider.setValue(50)
            diff_layout.addWidget(self.tangent_slider)

            self.show_tangent_check = QCheckBox("Show Tangent Line")
            diff_layout.addWidget(self.show_tangent_check)

            layout.addWidget(diff_group)

            # Integration group
            int_group = QGroupBox("Integration")
            int_layout = QVBoxLayout(int_group)

            bounds_row = QHBoxLayout()
            bounds_row.addWidget(QLabel("From:"))
            self.int_lower = QDoubleSpinBox()
            self.int_lower.setRange(-1000, 1000)
            self.int_lower.setValue(0.0)
            bounds_row.addWidget(self.int_lower)
            bounds_row.addWidget(QLabel("To:"))
            self.int_upper = QDoubleSpinBox()
            self.int_upper.setRange(-1000, 1000)
            self.int_upper.setValue(10.0)
            bounds_row.addWidget(self.int_upper)
            int_layout.addLayout(bounds_row)

            # Bounds sliders
            self.int_lower_slider = QSlider(Qt.Orientation.Horizontal)
            self.int_lower_slider.setRange(0, 100)
            self.int_lower_slider.setValue(0)
            int_layout.addWidget(self.int_lower_slider)

            self.int_upper_slider = QSlider(Qt.Orientation.Horizontal)
            self.int_upper_slider.setRange(0, 100)
            self.int_upper_slider.setValue(100)
            int_layout.addWidget(self.int_upper_slider)

            self.show_integral_btn = QPushButton("Show Integral")
            int_layout.addWidget(self.show_integral_btn)

            self.integral_value_label = QLabel("Integral: --")
            int_layout.addWidget(self.integral_value_label)

            layout.addWidget(int_group)

            layout.addStretch()
            return tab

        def _create_filters_tab(self) -> QWidget:
            """Create filter controls."""
            tab = QWidget()
            layout = QVBoxLayout(tab)

            # Filter type
            type_group = QGroupBox("Filter Type")
            type_layout = QVBoxLayout(type_group)

            self.filter_design_combo = QComboBox()
            self.filter_design_combo.addItems(
                [
                    "Butterworth",
                    "Chebyshev I",
                    "Chebyshev II",
                    "Elliptic",
                    "Bessel",
                    "Moving Average",
                    "Savitzky-Golay",
                    "Median",
                    "Gaussian",
                ]
            )
            type_layout.addWidget(self.filter_design_combo)

            self.filter_type_combo = QComboBox()
            self.filter_type_combo.addItems(
                [
                    "Lowpass",
                    "Highpass",
                    "Bandpass",
                    "Bandstop",
                ]
            )
            type_layout.addWidget(self.filter_type_combo)

            layout.addWidget(type_group)

            # Filter parameters
            param_group = QGroupBox("Parameters")
            param_layout = QGridLayout(param_group)

            param_layout.addWidget(QLabel("Cutoff (Hz):"), 0, 0)
            self.filter_cutoff = QDoubleSpinBox()
            self.filter_cutoff.setRange(0.001, 10000)
            self.filter_cutoff.setValue(10.0)
            param_layout.addWidget(self.filter_cutoff, 0, 1)

            param_layout.addWidget(QLabel("Cutoff 2 (Hz):"), 1, 0)
            self.filter_cutoff2 = QDoubleSpinBox()
            self.filter_cutoff2.setRange(0.001, 10000)
            self.filter_cutoff2.setValue(20.0)
            param_layout.addWidget(self.filter_cutoff2, 1, 1)

            param_layout.addWidget(QLabel("Order:"), 2, 0)
            self.filter_order = QSpinBox()
            self.filter_order.setRange(1, 10)
            self.filter_order.setValue(4)
            param_layout.addWidget(self.filter_order, 2, 1)

            param_layout.addWidget(QLabel("Window Size:"), 3, 0)
            self.filter_window = QSpinBox()
            self.filter_window.setRange(3, 101)
            self.filter_window.setValue(11)
            self.filter_window.setSingleStep(2)
            param_layout.addWidget(self.filter_window, 3, 1)

            layout.addWidget(param_group)

            # Apply filter button
            self.apply_filter_btn = QPushButton("Apply Filter")
            self.apply_filter_btn.setStyleSheet(
                "background-color: #0078d4; font-weight: bold;"
            )
            layout.addWidget(self.apply_filter_btn)

            self.show_freq_response_btn = QPushButton("Show Frequency Response")
            layout.addWidget(self.show_freq_response_btn)

            layout.addStretch()
            return tab

        def _create_noise_tab(self) -> QWidget:
            """Create noise controls."""
            tab = QWidget()
            layout = QVBoxLayout(tab)

            # Noise type
            type_group = QGroupBox("Noise Type")
            type_layout = QVBoxLayout(type_group)

            self.noise_type_combo = QComboBox()
            self.noise_type_combo.addItems(
                [
                    "White (Gaussian)",
                    "Pink (1/f)",
                    "Brown (Brownian)",
                    "Blue",
                    "Violet",
                    "Uniform",
                    "Impulse",
                    "Periodic (60Hz)",
                ]
            )
            type_layout.addWidget(self.noise_type_combo)

            layout.addWidget(type_group)

            # Noise parameters
            param_group = QGroupBox("Parameters")
            param_layout = QGridLayout(param_group)

            param_layout.addWidget(QLabel("SNR (dB):"), 0, 0)
            self.noise_snr = QDoubleSpinBox()
            self.noise_snr.setRange(-20, 100)
            self.noise_snr.setValue(20.0)
            param_layout.addWidget(self.noise_snr, 0, 1)

            param_layout.addWidget(QLabel("Amplitude:"), 1, 0)
            self.noise_amplitude = QDoubleSpinBox()
            self.noise_amplitude.setRange(0, 1000)
            self.noise_amplitude.setValue(0.1)
            self.noise_amplitude.setDecimals(4)
            param_layout.addWidget(self.noise_amplitude, 1, 1)

            self.noise_use_snr = QCheckBox("Use SNR (not amplitude)")
            self.noise_use_snr.setChecked(True)
            param_layout.addWidget(self.noise_use_snr, 2, 0, 1, 2)

            layout.addWidget(param_group)

            # Add noise button
            self.add_noise_btn = QPushButton("Add Noise")
            self.add_noise_btn.setStyleSheet(
                "background-color: #0078d4; font-weight: bold;"
            )
            layout.addWidget(self.add_noise_btn)

            self.reset_signal_btn = QPushButton("Reset to Original")
            layout.addWidget(self.reset_signal_btn)

            layout.addStretch()
            return tab

        def _create_import_tab(self) -> QWidget:
            """Create import controls."""
            tab = QWidget()
            layout = QVBoxLayout(tab)

            # File import
            import_group = QGroupBox("Import from File")
            import_layout = QVBoxLayout(import_group)

            file_row = QHBoxLayout()
            self.import_path = QLineEdit()
            self.import_path.setPlaceholderText("Select CSV file...")
            file_row.addWidget(self.import_path)
            self.browse_btn = QPushButton("Browse")
            file_row.addWidget(self.browse_btn)
            import_layout.addLayout(file_row)

            col_row = QHBoxLayout()
            col_row.addWidget(QLabel("Time Col:"))
            self.time_col_spin = QSpinBox()
            self.time_col_spin.setRange(0, 100)
            self.time_col_spin.setValue(0)
            col_row.addWidget(self.time_col_spin)
            col_row.addWidget(QLabel("Value Col:"))
            self.value_col_spin = QSpinBox()
            self.value_col_spin.setRange(0, 100)
            self.value_col_spin.setValue(1)
            col_row.addWidget(self.value_col_spin)
            import_layout.addLayout(col_row)

            self.import_btn = QPushButton("Import Signal")
            self.import_btn.setStyleSheet(
                "background-color: #0078d4; font-weight: bold;"
            )
            import_layout.addWidget(self.import_btn)

            layout.addWidget(import_group)

            layout.addStretch()
            return tab

        def _create_right_panel(self) -> QWidget:
            """Create the right plot panel."""
            panel = QWidget()
            layout = QVBoxLayout(panel)

            # Main plot canvas
            self.canvas = MplCanvas(self, width=8, height=5, dpi=100)
            self.canvas.setup_dark_theme()

            # Toolbar
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setStyleSheet("background-color: #333;")

            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, 1)

            # Secondary plot for derivative/integral
            self.canvas2 = MplCanvas(self, width=8, height=3, dpi=100)
            self.canvas2.setup_dark_theme()
            layout.addWidget(self.canvas2)

            return panel

        def _setup_connections(self) -> None:
            """Setup signal-slot connections."""
            # Generation
            self.signal_type_combo.currentIndexChanged.connect(
                self.param_stack.setCurrentIndex
            )
            self.generate_btn.clicked.connect(self._generate_signal)

            # Fitting
            self.fit_btn.clicked.connect(self._fit_function)
            self.auto_fit_btn.clicked.connect(self._auto_fit)

            # Limits
            self.apply_sat_btn.clicked.connect(self._apply_saturation)
            self.sat_preview_check.stateChanged.connect(self._update_saturation_preview)

            # Calculus
            self.show_derivative_btn.clicked.connect(self._show_derivative)
            self.show_integral_btn.clicked.connect(self._show_integral)
            self.tangent_slider.valueChanged.connect(self._update_tangent_position)
            self.show_tangent_check.stateChanged.connect(self._toggle_tangent)
            self.int_lower_slider.valueChanged.connect(self._update_integral_bounds)
            self.int_upper_slider.valueChanged.connect(self._update_integral_bounds)

            # Filters
            self.apply_filter_btn.clicked.connect(self._apply_filter)
            self.show_freq_response_btn.clicked.connect(self._show_frequency_response)

            # Noise
            self.add_noise_btn.clicked.connect(self._add_noise)
            self.reset_signal_btn.clicked.connect(self._reset_signal)

            # Import
            self.browse_btn.clicked.connect(self._browse_file)
            self.import_btn.clicked.connect(self._import_signal)

            # Output
            self.apply_btn.clicked.connect(self._apply_to_joint)
            self.export_btn.clicked.connect(self._export_signal)

        def _generate_default_signal(self) -> None:
            """Generate a default signal to start with."""
            t = np.linspace(0, 10, 1000)
            self.current_signal = SignalGenerator.sinusoid(
                t, amplitude=1.0, frequency=1.0, name="default"
            )
            self.original_signal = self.current_signal.copy()
            self._update_plot()

        def _generate_signal(self) -> None:
            """Generate signal based on current settings."""
            t = np.linspace(
                self.t_start_spin.value(),
                self.t_end_spin.value(),
                self.n_points_spin.value(),
            )

            signal_type = self.signal_type_combo.currentText()

            try:
                if signal_type == "Sinusoid":
                    self.current_signal = SignalGenerator.sinusoid(
                        t,
                        amplitude=self.sin_amplitude.value(),
                        frequency=self.sin_frequency.value(),
                        phase=self.sin_phase.value(),
                        offset=self.sin_offset.value(),
                    )
                elif signal_type == "Cosine":
                    self.current_signal = SignalGenerator.cosine(
                        t,
                        amplitude=self.sin_amplitude.value(),
                        frequency=self.sin_frequency.value(),
                        phase=self.sin_phase.value(),
                        offset=self.sin_offset.value(),
                    )
                elif signal_type == "Polynomial":
                    coeffs_str = self.poly_coeffs_input.text()
                    coeffs = [float(c.strip()) for c in coeffs_str.split(",")]
                    self.current_signal = SignalGenerator.polynomial(t, coeffs)
                elif signal_type == "Exponential":
                    self.current_signal = SignalGenerator.exponential(
                        t,
                        amplitude=self.exp_amplitude.value(),
                        decay_rate=self.exp_decay.value(),
                        offset=self.exp_offset.value(),
                    )
                elif signal_type == "Linear":
                    self.current_signal = SignalGenerator.linear(
                        t,
                        slope=self.linear_slope.value(),
                        intercept=self.linear_intercept.value(),
                    )
                elif signal_type == "Step":
                    self.current_signal = SignalGenerator.step(
                        t,
                        step_time=self.step_time.value(),
                        step_value=self.step_value.value(),
                        initial_value=self.step_initial.value(),
                    )
                elif signal_type == "Chirp":
                    self.current_signal = SignalGenerator.chirp(
                        t,
                        f0=self.chirp_f0.value(),
                        f1=self.chirp_f1.value(),
                        amplitude=self.chirp_amplitude.value(),
                    )
                elif signal_type == "Square":
                    self.current_signal = SignalGenerator.square(
                        t,
                        frequency=self.square_freq.value(),
                        amplitude=self.square_amplitude.value(),
                        duty_cycle=self.square_duty.value(),
                    )
                elif signal_type == "Triangle":
                    self.current_signal = SignalGenerator.triangle(
                        t,
                        frequency=self.triangle_freq.value(),
                        amplitude=self.triangle_amplitude.value(),
                    )
                elif signal_type == "Custom":
                    expr = self.custom_expr.text()
                    if expr:
                        # Safe evaluation
                        safe_dict = {
                            "sin": np.sin,
                            "cos": np.cos,
                            "tan": np.tan,
                            "exp": np.exp,
                            "log": np.log,
                            "sqrt": np.sqrt,
                            "pi": np.pi,
                            "t": t,
                        }
                        values = eval(
                            expr, {"__builtins__": {}}, safe_dict
                        )  # noqa: S307
                        self.current_signal = Signal(t, values, name="custom")
                    else:
                        return

                self.original_signal = self.current_signal.copy()
                self._update_plot()
                self._log(f"Generated {signal_type} signal")

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to generate signal: {e}")

        def _fit_function(self) -> None:
            """Fit a function to the current signal."""
            if self.current_signal is None:
                return

            fit_type = self.fit_type_combo.currentText()
            fitter = FunctionFitter()

            try:
                if fit_type == "Sinusoid":
                    result = fitter.fit_sinusoid(self.current_signal)
                elif fit_type == "Cosine":
                    result = fitter.fit_cosine(self.current_signal)
                elif fit_type == "Exponential Decay":
                    result = fitter.fit_exponential_decay(self.current_signal)
                elif fit_type == "Exponential Growth":
                    result = fitter.fit_exponential_growth(self.current_signal)
                elif fit_type == "Linear":
                    result = fitter.fit_linear(self.current_signal)
                elif fit_type == "Polynomial":
                    result = fitter.fit_polynomial(
                        self.current_signal,
                        order=self.fit_poly_order.value(),
                    )
                elif fit_type == "Custom":
                    expr = self.fit_custom_expr.text()
                    params_str = self.fit_custom_params.text()
                    if expr and params_str:
                        params = [p.strip() for p in params_str.split(",")]
                        result = fitter.fit_custom_expression(
                            self.current_signal, expr, params
                        )
                    else:
                        return
                else:
                    return

                # Display results
                self._log(
                    f"Fit: {fit_type}\n"
                    f"R^2: {result.r_squared:.4f}\n"
                    f"RMSE: {result.rmse:.4f}\n"
                    f"Parameters: {result.parameters}"
                )

                # Plot fitted curve
                self._update_plot(fitted_signal=result.fitted_signal)

            except Exception as e:
                QMessageBox.warning(self, "Fit Error", f"Failed to fit: {e}")

        def _auto_fit(self) -> None:
            """Automatically find the best fit."""
            if self.current_signal is None:
                return

            try:
                fitter = FunctionFitter()
                best_type, result = fitter.auto_fit(self.current_signal)

                self._log(
                    f"Best fit: {best_type}\n"
                    f"R^2: {result.r_squared:.4f}\n"
                    f"Parameters: {result.parameters}"
                )

                self._update_plot(fitted_signal=result.fitted_signal)

            except Exception as e:
                QMessageBox.warning(self, "Auto-fit Error", f"Failed: {e}")

        def _apply_saturation(self) -> None:
            """Apply saturation to the signal."""
            if self.current_signal is None:
                return

            mode_map = {
                "Hard": SaturationMode.HARD,
                "Soft": SaturationMode.SOFT,
                "Tanh": SaturationMode.TANH,
                "Sigmoid": SaturationMode.SIGMOID,
                "Atan": SaturationMode.ATAN,
                "Cubic": SaturationMode.CUBIC,
                "Exponential": SaturationMode.EXPONENTIAL,
            }

            mode = mode_map[self.sat_mode_combo.currentText()]

            self.current_signal = apply_saturation(
                self.current_signal,
                lower=self.sat_lower.value(),
                upper=self.sat_upper.value(),
                mode=mode,
                smoothness=self.sat_smoothness.value(),
            )

            self._update_plot()
            self._log(f"Applied {mode.value} saturation")

        def _update_saturation_preview(self) -> None:
            """Update saturation preview if enabled."""
            if not self.original_signal:
                return

            if self.sat_preview_check.isChecked():
                # Show preview without modifying current signal
                mode_map = {
                    "Hard Clip": SaturationMode.HARD,
                    "Soft Clip (tanh)": SaturationMode.SOFT_TANH,
                    "Soft Clip (sigmoid)": SaturationMode.SOFT_SIGMOID,
                    "Polynomial": SaturationMode.POLYNOMIAL,
                }
                mode = mode_map.get(
                    self.sat_mode_combo.currentText(), SaturationMode.HARD
                )

                # Create preview signal
                preview = apply_saturation(
                    (
                        self.current_signal.copy()
                        if self.current_signal
                        else self.original_signal.copy()
                    ),
                    lower=self.sat_lower.value(),
                    upper=self.sat_upper.value(),
                    mode=mode,
                    smoothness=self.sat_smoothness.value(),
                )

                # Show on secondary plot
                self._update_secondary_plot(preview, "Saturation Preview")
            else:
                # Clear preview
                self.canvas2.axes.clear()
                self.canvas2.draw()

        def _show_derivative(self) -> None:
            """Show the derivative of the current signal."""
            if self.current_signal is None:
                return

            diff = Differentiator()
            self.derivative_signal = diff.differentiate(
                self.current_signal,
                order=self.diff_order.value(),
            )

            self._update_secondary_plot(self.derivative_signal, "Derivative")

        def _show_integral(self) -> None:
            """Show the integral of the current signal."""
            if self.current_signal is None:
                return

            integrator = Integrator()
            result = integrator.integrate(
                self.current_signal,
                lower_bound=self.int_lower.value(),
                upper_bound=self.int_upper.value(),
            )

            self.integral_signal = result.cumulative_signal
            self.integral_value_label.setText(f"Integral: {result.value:.4f}")

            self._update_secondary_plot(self.integral_signal, "Integral")

        def _update_tangent_position(self, value: int) -> None:
            """Update tangent line position from slider."""
            if self.current_signal is None:
                return

            t_range = self.current_signal.time[-1] - self.current_signal.time[0]
            t_point = self.current_signal.time[0] + (value / 100) * t_range

            self.tangent_t_spin.setValue(t_point)

            if self.show_tangent_check.isChecked():
                self._update_plot()

        def _toggle_tangent(self, state: int) -> None:
            """Toggle tangent line display."""
            self._update_plot()

        def _update_integral_bounds(self) -> None:
            """Update integral bounds from sliders."""
            if self.current_signal is None:
                return

            t_range = self.current_signal.time[-1] - self.current_signal.time[0]
            t0 = self.current_signal.time[0]

            lower = t0 + (self.int_lower_slider.value() / 100) * t_range
            upper = t0 + (self.int_upper_slider.value() / 100) * t_range

            self.int_lower.setValue(lower)
            self.int_upper.setValue(upper)

        def _apply_filter(self) -> None:
            """Apply filter to the signal."""
            if self.current_signal is None:
                return

            design = self.filter_design_combo.currentText()
            filter_type = self.filter_type_combo.currentText().lower()

            try:
                if design in ("Moving Average", "Savitzky-Golay", "Median", "Gaussian"):
                    window = self.filter_window.value()
                    if window % 2 == 0:
                        window += 1

                    if design == "Moving Average":
                        self.current_signal = apply_moving_average(
                            self.current_signal, window
                        )
                    elif design == "Savitzky-Golay":
                        self.current_signal = apply_savgol(
                            self.current_signal, window, 3
                        )
                    elif design == "Median":
                        from src.shared.python.signal_toolkit.filters import (
                            apply_median_filter,
                        )

                        self.current_signal = apply_median_filter(
                            self.current_signal, window
                        )
                    elif design == "Gaussian":
                        from src.shared.python.signal_toolkit.filters import (
                            apply_gaussian_smoothing,
                        )

                        self.current_signal = apply_gaussian_smoothing(
                            self.current_signal, window / 3
                        )
                else:
                    # IIR filters
                    fs = self.current_signal.fs
                    cutoff = self.filter_cutoff.value()
                    order = self.filter_order.value()

                    if filter_type in ("bandpass", "bandstop"):
                        cutoff = (cutoff, self.filter_cutoff2.value())

                    ft = FilterType(filter_type)

                    if design == "Butterworth":
                        spec = FilterDesigner.butterworth(ft, cutoff, fs, order)
                    elif design == "Chebyshev I":
                        spec = FilterDesigner.chebyshev1(ft, cutoff, fs, order)
                    elif design == "Chebyshev II":
                        spec = FilterDesigner.chebyshev2(ft, cutoff, fs, order)
                    elif design == "Elliptic":
                        spec = FilterDesigner.elliptic(ft, cutoff, fs, order)
                    elif design == "Bessel":
                        spec = FilterDesigner.bessel(ft, cutoff, fs, order)
                    else:
                        return

                    self.current_signal = apply_filter(self.current_signal, spec)

                self._update_plot()
                self._log(f"Applied {design} {filter_type} filter")

            except Exception as e:
                QMessageBox.warning(self, "Filter Error", f"Failed: {e}")

        def _show_frequency_response(self) -> None:
            """Show frequency response of the current filter settings."""
            if self.current_signal is None:
                QMessageBox.information(
                    self,
                    "No Signal",
                    "Please generate or load a signal first.",
                )
                return

            design = self.filter_design_combo.currentText()

            # Non-IIR filters don't have a traditional frequency response
            if design in ("Moving Average", "Savitzky-Golay", "Median", "Gaussian"):
                QMessageBox.information(
                    self,
                    "Frequency Response",
                    f"{design} filters are FIR/smoothing filters.\n"
                    "Use IIR filter designs (Butterworth, Chebyshev, etc.) "
                    "to view frequency response.",
                )
                return

            try:
                from scipy import signal as scipy_signal

                filter_type = self.filter_type_combo.currentText().lower()
                fs = self.current_signal.fs
                cutoff = self.filter_cutoff.value()
                order = self.filter_order.value()

                if filter_type in ("bandpass", "bandstop"):
                    cutoff = (cutoff, self.filter_cutoff2.value())

                ft = FilterType(filter_type)

                # Get filter spec
                if design == "Butterworth":
                    spec = FilterDesigner.butterworth(ft, cutoff, fs, order)
                elif design == "Chebyshev I":
                    spec = FilterDesigner.chebyshev1(ft, cutoff, fs, order)
                elif design == "Chebyshev II":
                    spec = FilterDesigner.chebyshev2(ft, cutoff, fs, order)
                elif design == "Elliptic":
                    spec = FilterDesigner.elliptic(ft, cutoff, fs, order)
                elif design == "Bessel":
                    spec = FilterDesigner.bessel(ft, cutoff, fs, order)
                else:
                    return

                # Calculate frequency response
                w, h = scipy_signal.freqz(spec.b_coeffs, spec.a_coeffs, fs=fs)

                # Plot on secondary canvas
                self.canvas2.axes.clear()
                self.canvas2.setup_dark_theme()

                self.canvas2.axes.semilogy(w, np.abs(h), color="#4ecdc4", linewidth=1.5)
                self.canvas2.axes.set_title("Frequency Response", fontsize=10)
                self.canvas2.axes.set_xlabel("Frequency (Hz)")
                self.canvas2.axes.set_ylabel("Magnitude")
                self.canvas2.axes.grid(True, alpha=0.3)
                self.canvas2.draw()

                self._log(f"Showing frequency response for {design} {filter_type}")

            except Exception as e:
                QMessageBox.warning(
                    self, "Error", f"Failed to compute frequency response: {e}"
                )

        def _add_noise(self) -> None:
            """Add noise to the signal."""
            if self.current_signal is None:
                return

            noise_map = {
                "White (Gaussian)": NoiseType.WHITE,
                "Pink (1/f)": NoiseType.PINK,
                "Brown (Brownian)": NoiseType.BROWN,
                "Blue": NoiseType.BLUE,
                "Violet": NoiseType.VIOLET,
                "Uniform": NoiseType.UNIFORM,
                "Impulse": NoiseType.IMPULSE,
                "Periodic (60Hz)": NoiseType.PERIODIC,
            }

            noise_type = noise_map[self.noise_type_combo.currentText()]

            if self.noise_use_snr.isChecked():
                self.current_signal = add_noise_to_signal(
                    self.current_signal,
                    noise_type=noise_type,
                    snr_db=self.noise_snr.value(),
                )
            else:
                self.current_signal = add_noise_to_signal(
                    self.current_signal,
                    noise_type=noise_type,
                    amplitude=self.noise_amplitude.value(),
                )

            self._update_plot()
            self._log(f"Added {noise_type.value} noise")

        def _reset_signal(self) -> None:
            """Reset to original signal."""
            if self.original_signal:
                self.current_signal = self.original_signal.copy()
                self._update_plot()
                self._log("Reset to original signal")

        def _browse_file(self) -> None:
            """Browse for a file to import."""
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Signal",
                "",
                "CSV Files (*.csv);;All Files (*)",
            )
            if path:
                self.import_path.setText(path)

        def _import_signal(self) -> None:
            """Import signal from file."""
            path = self.import_path.text()
            if not path:
                return

            try:
                result = SignalImporter.from_csv(
                    path,
                    time_column=self.time_col_spin.value(),
                    value_columns=self.value_col_spin.value(),
                )

                if isinstance(result, list):
                    self.current_signal = result[0]
                else:
                    self.current_signal = result

                self.original_signal = self.current_signal.copy()
                self._update_plot()
                self._log(f"Imported signal from {Path(path).name}")

            except Exception as e:
                QMessageBox.warning(self, "Import Error", f"Failed: {e}")

        def _apply_to_joint(self) -> None:
            """Apply signal to selected joint."""
            if self.current_signal is None:
                return

            joint = self.joint_combo.currentText()

            # Fit polynomial to get coefficients for control system
            fitter = FunctionFitter()
            result = fitter.fit_polynomial(self.current_signal, order=6)

            # Get coefficients in [c0, c1, c2, ...] order
            coeffs = [result.parameters.get(f"c{i}", 0.0) for i in range(7)]

            self.signal_generated.emit(joint, coeffs)
            self._log(f"Applied to {joint}: {coeffs}")

        def _export_signal(self) -> None:
            """Export current signal to file."""
            if self.current_signal is None:
                return

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Signal",
                "",
                "CSV Files (*.csv);;JSON Files (*.json)",
            )

            if path:
                try:
                    if path.endswith(".json"):
                        SignalExporter.to_json(self.current_signal, path)
                    else:
                        SignalExporter.to_csv(self.current_signal, path)
                    self._log(f"Exported to {Path(path).name}")
                except Exception as e:
                    QMessageBox.warning(self, "Export Error", f"Failed: {e}")

        def _update_plot(
            self,
            fitted_signal: Signal | None = None,
        ) -> None:
            """Update the main plot."""
            self.canvas.axes.clear()
            self.canvas.setup_dark_theme()

            if self.current_signal is None:
                self.canvas.draw()
                return

            # Plot current signal
            self.canvas.axes.plot(
                self.current_signal.time,
                self.current_signal.values,
                color="#4da6ff",
                linewidth=1.5,
                label="Signal",
            )

            # Plot fitted signal if provided
            if fitted_signal:
                self.canvas.axes.plot(
                    fitted_signal.time,
                    fitted_signal.values,
                    color="#ff6b6b",
                    linewidth=2,
                    linestyle="--",
                    label="Fit",
                )

            # Plot tangent line if enabled
            if self.show_tangent_check.isChecked():
                tangent = compute_tangent_line(
                    self.current_signal,
                    self.tangent_t_spin.value(),
                )
                self.canvas.axes.plot(
                    tangent.t_range,
                    tangent.line_values,
                    color="#ffd93d",
                    linewidth=2,
                    label=f"Tangent (slope={tangent.slope:.3f})",
                )
                self.canvas.axes.scatter(
                    [tangent.t_point],
                    [tangent.y_point],
                    color="#ffd93d",
                    s=50,
                    zorder=5,
                )

            self.canvas.axes.set_xlabel("Time")
            self.canvas.axes.set_ylabel("Value")
            self.canvas.axes.set_title(self.current_signal.name)
            self.canvas.axes.legend(loc="upper right")

            self.canvas.draw()

        def _update_secondary_plot(
            self,
            signal: Signal,
            title: str,
        ) -> None:
            """Update the secondary plot."""
            self.canvas2.axes.clear()
            self.canvas2.setup_dark_theme()

            self.canvas2.axes.plot(
                signal.time,
                signal.values,
                color="#6bcb77",
                linewidth=1.5,
            )

            self.canvas2.axes.set_xlabel("Time")
            self.canvas2.axes.set_ylabel("Value")
            self.canvas2.axes.set_title(title)

            self.canvas2.draw()

        def _log(self, message: str) -> None:
            """Log a message to the result text area."""
            self.result_text.append(message)

        def set_joints(self, joints: list[str]) -> None:
            """Set the list of available joints."""
            self.joint_names = joints
            self.joint_combo.clear()
            self.joint_combo.addItems(joints)

    def main() -> None:
        """Run the widget as a standalone application."""
        app = QApplication(sys.argv)
        window = SignalToolkitWidget()
        window.show()
        sys.exit(app.exec())

else:
    # Stub class when dependencies are not available
    class SignalToolkitWidget:  # type: ignore[no-redef]
        """Stub class when PyQt6 or matplotlib is not available."""

        def __init__(self, *args, **kwargs) -> None:
            msg = "SignalToolkitWidget requires PyQt6 and matplotlib"
            raise ImportError(msg)

    def main() -> None:
        """Stub main function."""
        print("SignalToolkitWidget requires PyQt6 and matplotlib")


if __name__ == "__main__":
    main()
