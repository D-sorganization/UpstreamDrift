"""Signal Toolkit Widget – UI construction mixin.

Extracts all ``_create_*`` methods that build the left-panel tabs
(generation, fitting, limits, calculus, filters, noise, import),
the right plot panel, the joint/output group, and the result area.
"""

from __future__ import annotations

from typing import Any

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.theme.style_constants import Styles

try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SignalToolkitUIMixin:
    """Mixin providing all UI-construction helpers for SignalToolkitWidget."""

    # ------------------------------------------------------------------
    # Top-level layout
    # ------------------------------------------------------------------

    def _setup_ui(self: Any) -> None:
        """Setup the user interface."""
        from PyQt6.QtWidgets import QHBoxLayout, QSplitter

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------

    def _create_left_panel(self: Any) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        panel.setMinimumWidth(350)
        panel.setMaximumWidth(500)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(10)

        tabs = QTabWidget()
        tabs.addTab(self._create_generation_tab(), "Generate")
        tabs.addTab(self._create_fitting_tab(), "Fit")
        tabs.addTab(self._create_limits_tab(), "Limits")
        tabs.addTab(self._create_calculus_tab(), "Calculus")
        tabs.addTab(self._create_filters_tab(), "Filters")
        tabs.addTab(self._create_noise_tab(), "Noise")
        tabs.addTab(self._create_import_tab(), "Import")

        content_layout.addWidget(tabs)

        # Joint / output group
        joint_group = QGroupBox("Output")
        joint_layout = QVBoxLayout(joint_group)

        joint_row = QHBoxLayout()
        joint_row.addWidget(QLabel("Joint:"))
        self.joint_combo = QComboBox()
        self.joint_combo.addItems(["Joint 1", "Joint 2", "Joint 3"])
        joint_row.addWidget(self.joint_combo, 1)
        joint_layout.addLayout(joint_row)

        self.apply_btn = QPushButton("Apply to Joint")
        self.apply_btn.setStyleSheet(Styles.BTN_ACTION_GREEN)
        joint_layout.addWidget(self.apply_btn)

        self.export_btn = QPushButton("Export Signal")
        joint_layout.addWidget(self.export_btn)

        content_layout.addWidget(joint_group)

        # Result area
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

    # ------------------------------------------------------------------
    # Generation tab
    # ------------------------------------------------------------------

    def _create_generation_tab(self: Any) -> QWidget:
        """Create signal generation controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

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

        self.param_stack = QStackedWidget()
        self.param_stack.addWidget(self._create_sinusoid_params())
        self.param_stack.addWidget(self._create_sinusoid_params())  # cosine
        self.param_stack.addWidget(self._create_polynomial_params())
        self.param_stack.addWidget(self._create_exponential_params())
        self.param_stack.addWidget(self._create_linear_params())
        self.param_stack.addWidget(self._create_step_params())
        self.param_stack.addWidget(self._create_chirp_params())
        self.param_stack.addWidget(self._create_square_params())
        self.param_stack.addWidget(self._create_triangle_params())
        self.param_stack.addWidget(self._create_custom_params())
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

        self.generate_btn = QPushButton("Generate Signal")
        self.generate_btn.setStyleSheet(Styles.BTN_ACTION_BLUE)
        layout.addWidget(self.generate_btn)
        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Parameter sub-widgets
    # ------------------------------------------------------------------

    def _create_sinusoid_params(self: Any) -> QWidget:
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

    def _create_polynomial_params(self: Any) -> QWidget:
        """Create polynomial parameter controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Coefficients (c0, c1, c2, ...):"))
        self.poly_coeffs_input = QLineEdit()
        self.poly_coeffs_input.setPlaceholderText("0, 0, 1")
        self.poly_coeffs_input.setText("0, 1, 0.5")
        layout.addWidget(self.poly_coeffs_input)

        layout.addWidget(QLabel("Order:"))
        self.poly_order_spin = QSpinBox()
        self.poly_order_spin.setRange(1, 10)
        self.poly_order_spin.setValue(6)
        layout.addWidget(self.poly_order_spin)

        return widget

    def _create_exponential_params(self: Any) -> QWidget:
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

    def _create_linear_params(self: Any) -> QWidget:
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

    def _create_step_params(self: Any) -> QWidget:
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

    def _create_chirp_params(self: Any) -> QWidget:
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

    def _create_square_params(self: Any) -> QWidget:
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

    def _create_triangle_params(self: Any) -> QWidget:
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

    def _create_custom_params(self: Any) -> QWidget:
        """Create custom expression controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Expression (use 't' for time):"))
        self.custom_expr = QLineEdit()
        self.custom_expr.setPlaceholderText("sin(2*pi*t) + 0.5*cos(4*pi*t)")
        layout.addWidget(self.custom_expr)

        layout.addWidget(QLabel("Available: sin, cos, exp, log, sqrt, pi"))
        return widget

    # ------------------------------------------------------------------
    # Fitting tab
    # ------------------------------------------------------------------

    def _create_fitting_tab(self: Any) -> QWidget:
        """Create function fitting controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

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

        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Poly Order:"))
        self.fit_poly_order = QSpinBox()
        self.fit_poly_order.setRange(1, 10)
        self.fit_poly_order.setValue(6)
        order_row.addWidget(self.fit_poly_order)
        fit_layout.addLayout(order_row)
        layout.addWidget(fit_group)

        custom_group = QGroupBox("Custom Fit (optional)")
        custom_layout = QVBoxLayout(custom_group)
        self.fit_custom_expr = QLineEdit()
        self.fit_custom_expr.setPlaceholderText("a*sin(b*t) + c")
        custom_layout.addWidget(self.fit_custom_expr)
        self.fit_custom_params = QLineEdit()
        self.fit_custom_params.setPlaceholderText("a, b, c")
        custom_layout.addWidget(self.fit_custom_params)
        layout.addWidget(custom_group)

        self.fit_btn = QPushButton("Fit Function")
        self.fit_btn.setStyleSheet("background-color: #0078d4; font-weight: bold;")
        layout.addWidget(self.fit_btn)

        self.auto_fit_btn = QPushButton("Auto-Detect Best Fit")
        layout.addWidget(self.auto_fit_btn)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Limits tab
    # ------------------------------------------------------------------

    def _create_limits_tab(self: Any) -> QWidget:
        """Create limit/saturation controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

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
        self.sat_mode_combo.setCurrentIndex(2)
        sat_layout.addWidget(self.sat_mode_combo, 2, 1)

        sat_layout.addWidget(QLabel("Smoothness:"), 3, 0)
        self.sat_smoothness = QDoubleSpinBox()
        self.sat_smoothness.setRange(0.1, 100)
        self.sat_smoothness.setValue(1.0)
        sat_layout.addWidget(self.sat_smoothness, 3, 1)

        layout.addWidget(sat_group)

        self.apply_sat_btn = QPushButton("Apply Saturation")
        self.apply_sat_btn.setStyleSheet(
            "background-color: #0078d4; font-weight: bold;"
        )
        layout.addWidget(self.apply_sat_btn)

        self.sat_preview_check = QCheckBox("Live Preview")
        self.sat_preview_check.setChecked(True)
        layout.addWidget(self.sat_preview_check)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Calculus tab
    # ------------------------------------------------------------------

    def _create_calculus_tab(self: Any) -> QWidget:
        """Create differentiation and integration controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Differentiation
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

        tangent_row = QHBoxLayout()
        tangent_row.addWidget(QLabel("Tangent at t="))
        self.tangent_t_spin = QDoubleSpinBox()
        self.tangent_t_spin.setRange(-1000, 1000)
        self.tangent_t_spin.setValue(5.0)
        tangent_row.addWidget(self.tangent_t_spin)
        diff_layout.addLayout(tangent_row)

        self.tangent_slider = QSlider(Qt.Orientation.Horizontal)
        self.tangent_slider.setRange(0, 100)
        self.tangent_slider.setValue(50)
        diff_layout.addWidget(self.tangent_slider)

        self.show_tangent_check = QCheckBox("Show Tangent Line")
        diff_layout.addWidget(self.show_tangent_check)

        layout.addWidget(diff_group)

        # Integration
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

    # ------------------------------------------------------------------
    # Filters tab
    # ------------------------------------------------------------------

    def _create_filters_tab(self: Any) -> QWidget:
        """Create filter controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

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
        self.filter_type_combo.addItems(["Lowpass", "Highpass", "Bandpass", "Bandstop"])
        type_layout.addWidget(self.filter_type_combo)

        layout.addWidget(type_group)

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

        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.setStyleSheet(
            "background-color: #0078d4; font-weight: bold;"
        )
        layout.addWidget(self.apply_filter_btn)

        self.show_freq_response_btn = QPushButton("Show Frequency Response")
        layout.addWidget(self.show_freq_response_btn)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Noise tab
    # ------------------------------------------------------------------

    def _create_noise_tab(self: Any) -> QWidget:
        """Create noise controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

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

        self.add_noise_btn = QPushButton("Add Noise")
        self.add_noise_btn.setStyleSheet(
            "background-color: #0078d4; font-weight: bold;"
        )
        layout.addWidget(self.add_noise_btn)

        self.reset_signal_btn = QPushButton("Reset to Original")
        layout.addWidget(self.reset_signal_btn)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Import tab
    # ------------------------------------------------------------------

    def _create_import_tab(self: Any) -> QWidget:
        """Create import controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

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
        self.import_btn.setStyleSheet("background-color: #0078d4; font-weight: bold;")
        import_layout.addWidget(self.import_btn)

        layout.addWidget(import_group)
        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Right panel (plots)
    # ------------------------------------------------------------------

    def _create_right_panel(self: Any) -> QWidget:
        """Create the right plot panel."""
        from .signal_toolkit_widget import MplCanvas

        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.canvas = MplCanvas(self, width=8, height=5, dpi=100)
        self.canvas.setup_dark_theme()

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #333;")

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.canvas2 = MplCanvas(self, width=8, height=3, dpi=100)
        self.canvas2.setup_dark_theme()
        layout.addWidget(self.canvas2)

        return panel
