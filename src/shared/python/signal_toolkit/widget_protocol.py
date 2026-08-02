"""Protocol defining the interface between widget mixins.

This module solves the type: ignore proliferation in widget mixins by providing
a Protocol class that describes the shared attributes accessed across
UISetupMixin, ProcessingMixin, and PlottingMixin. Each mixin can reference
this protocol via TYPE_CHECKING imports for static analysis without runtime overhead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QLabel,
        QLineEdit,
        QSlider,
        QSpinBox,
        QTextEdit,
    )

    from .core import Signal


class WidgetProtocol(Protocol):
    """Protocol describing shared attributes for SignalToolkitWidget mixins.

    This allows mypy to verify cross-mixin attribute access without
    ``# type: ignore[attr-defined]`` annotations.
    """

    # --- Canvases ---
    canvas: Any  # MplCanvas
    canvas2: Any  # MplCanvas

    # --- State ---
    current_signal: Signal | None
    original_signal: Signal | None
    derivative_signal: Signal | None
    integral_signal: Signal | None

    # --- Signals ---
    signal_generated: Any
    signal_updated: Any

    # --- UI controls (type stubs for static checking) ---
    show_tangent_check: QCheckBox
    tangent_t_spin: QDoubleSpinBox
    result_text: QTextEdit
    joint_combo: QComboBox
    filter_design_combo: QComboBox
    filter_type_combo: QComboBox
    filter_cutoff: QDoubleSpinBox
    filter_cutoff2: QDoubleSpinBox
    filter_order: QSpinBox
    filter_window: QSpinBox
    noise_type_combo: QComboBox
    noise_snr: QDoubleSpinBox
    noise_amplitude: QDoubleSpinBox
    noise_use_snr: QCheckBox
    sat_lower: QDoubleSpinBox
    sat_upper: QDoubleSpinBox
    sat_mode_combo: QComboBox
    sat_smoothness: QDoubleSpinBox
    sat_preview_check: QCheckBox
    diff_order: QSpinBox
    int_lower: QDoubleSpinBox
    int_upper: QDoubleSpinBox
    int_lower_slider: QSlider
    int_upper_slider: QSlider
    tangent_slider: QSlider
    integral_value_label: QLabel
    import_path: QLineEdit

    # --- Default time array ---
    t_default: np.ndarray
    joint_names: list[str]

    # --- Series tab (Issue #1279) ---
    series_type_combo: QComboBox
    series_center_spin: QDoubleSpinBox
    series_terms_spin: QSpinBox

    # --- Generation controls ---
    signal_type_combo: QComboBox
    t_start_spin: QDoubleSpinBox
    t_end_spin: QDoubleSpinBox
    n_points_spin: QSpinBox
    sin_amplitude: QDoubleSpinBox
    sin_frequency: QDoubleSpinBox
    sin_phase: QDoubleSpinBox
    sin_offset: QDoubleSpinBox
    poly_coeffs_input: QLineEdit
    exp_amplitude: QDoubleSpinBox
    exp_decay: QDoubleSpinBox
    exp_offset: QDoubleSpinBox
    linear_slope: QDoubleSpinBox
    linear_intercept: QDoubleSpinBox
    step_time: QDoubleSpinBox
    step_value: QDoubleSpinBox
    step_initial: QDoubleSpinBox
    chirp_f0: QDoubleSpinBox
    chirp_f1: QDoubleSpinBox
    chirp_amplitude: QDoubleSpinBox
    square_freq: QDoubleSpinBox
    square_amplitude: QDoubleSpinBox
    square_duty: QDoubleSpinBox
    triangle_freq: QDoubleSpinBox
    triangle_amplitude: QDoubleSpinBox
    custom_expr: QLineEdit

    # --- Fitting controls ---
    fit_type_combo: QComboBox
    fit_poly_order: QSpinBox
    fit_custom_expr: QLineEdit
    fit_custom_params: QLineEdit

    # --- Import controls ---
    time_col_spin: QSpinBox
    value_col_spin: QSpinBox

    # --- Methods ---
    def _update_plot(self, fitted_signal: Signal | None = None) -> None: ...
    def _update_secondary_plot(self, signal: Signal, title: str) -> None: ...
    def _update_frequency_response_plot(
        self,
        frequencies: np.ndarray,
        magnitude: np.ndarray,
        phase: np.ndarray,
        title: str,
    ) -> None: ...
    def _log(self, message: str) -> None: ...
