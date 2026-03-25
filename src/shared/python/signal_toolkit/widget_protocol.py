"""Protocol defining the expected host interface for Signal Toolkit mixins.

All three mixin classes (UISetupMixin, ProcessingMixin, PlottingMixin) access
attributes that are only defined on the concrete host class
(SignalToolkitWidget). Without a shared type declaration mypy cannot verify
those accesses and every callsite needed a ``# type: ignore[attr-defined]``
comment.

This module defines ``_SignalToolkitHost``, a ``typing.Protocol`` that
enumerates every attribute the mixins depend on. Annotating ``self`` as this
protocol inside the mixin methods eliminates the suppressions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Heavy GUI imports are only needed for type-checking.
    # pyqtBoundSignal is the instance-level type for pyqtSignal attributes
    # (pyqtSignal is only the class-level descriptor; instances expose emit()).
    from PyQt6.QtCore import pyqtBoundSignal as PyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QLabel,
        QLineEdit,
        QSlider,
        QSpinBox,
        QStackedWidget,
        QTextEdit,
    )

    from .core import Signal

    # MplCanvas is conditionally defined in widget.py; import only for typing.
    from .widget import MplCanvas


class _SignalToolkitHost(Protocol):
    """Protocol listing every attribute that the three mixin classes access.

    Concrete implementation is ``SignalToolkitWidget`` in ``widget.py``.
    """

    # ------------------------------------------------------------------ #
    # Qt signals                                                           #
    # ------------------------------------------------------------------ #
    signal_generated: PyqtSignal
    signal_updated: PyqtSignal

    # ------------------------------------------------------------------ #
    # Runtime state                                                        #
    # ------------------------------------------------------------------ #
    current_signal: Signal | None
    original_signal: Signal | None
    derivative_signal: Signal | None
    integral_signal: Signal | None
    joint_names: list[str]

    # ------------------------------------------------------------------ #
    # Plot canvases                                                        #
    # ------------------------------------------------------------------ #
    canvas: MplCanvas
    canvas2: MplCanvas

    # ------------------------------------------------------------------ #
    # Generation widgets                                                   #
    # ------------------------------------------------------------------ #
    signal_type_combo: QComboBox
    param_stack: QStackedWidget
    t_start_spin: QDoubleSpinBox
    t_end_spin: QDoubleSpinBox
    n_points_spin: QSpinBox

    # Sinusoid / Cosine
    sin_amplitude: QDoubleSpinBox
    sin_frequency: QDoubleSpinBox
    sin_phase: QDoubleSpinBox
    sin_offset: QDoubleSpinBox

    # Polynomial
    poly_coeffs_input: QLineEdit
    poly_order_spin: QSpinBox

    # Exponential
    exp_amplitude: QDoubleSpinBox
    exp_decay: QDoubleSpinBox
    exp_offset: QDoubleSpinBox

    # Linear
    linear_slope: QDoubleSpinBox
    linear_intercept: QDoubleSpinBox

    # Step
    step_time: QDoubleSpinBox
    step_value: QDoubleSpinBox
    step_initial: QDoubleSpinBox

    # Chirp
    chirp_f0: QDoubleSpinBox
    chirp_f1: QDoubleSpinBox
    chirp_amplitude: QDoubleSpinBox

    # Square
    square_freq: QDoubleSpinBox
    square_amplitude: QDoubleSpinBox
    square_duty: QDoubleSpinBox

    # Triangle
    triangle_freq: QDoubleSpinBox
    triangle_amplitude: QDoubleSpinBox

    # Custom expression
    custom_expr: QLineEdit

    # ------------------------------------------------------------------ #
    # Fitting widgets                                                      #
    # ------------------------------------------------------------------ #
    fit_type_combo: QComboBox
    fit_poly_order: QSpinBox
    fit_custom_expr: QLineEdit
    fit_custom_params: QLineEdit

    # ------------------------------------------------------------------ #
    # Saturation / limits widgets                                          #
    # ------------------------------------------------------------------ #
    sat_lower: QDoubleSpinBox
    sat_upper: QDoubleSpinBox
    sat_mode_combo: QComboBox
    sat_smoothness: QDoubleSpinBox
    sat_preview_check: QCheckBox

    # ------------------------------------------------------------------ #
    # Calculus widgets                                                     #
    # ------------------------------------------------------------------ #
    diff_order: QSpinBox
    tangent_t_spin: QDoubleSpinBox
    tangent_slider: QSlider
    show_tangent_check: QCheckBox
    int_lower: QDoubleSpinBox
    int_upper: QDoubleSpinBox
    int_lower_slider: QSlider
    int_upper_slider: QSlider
    integral_value_label: QLabel

    # ------------------------------------------------------------------ #
    # Filter widgets                                                       #
    # ------------------------------------------------------------------ #
    filter_design_combo: QComboBox
    filter_type_combo: QComboBox
    filter_cutoff: QDoubleSpinBox
    filter_cutoff2: QDoubleSpinBox
    filter_order: QSpinBox
    filter_window: QSpinBox

    # ------------------------------------------------------------------ #
    # Noise widgets                                                        #
    # ------------------------------------------------------------------ #
    noise_type_combo: QComboBox
    noise_snr: QDoubleSpinBox
    noise_amplitude: QDoubleSpinBox
    noise_use_snr: QCheckBox

    # ------------------------------------------------------------------ #
    # Import widgets                                                       #
    # ------------------------------------------------------------------ #
    import_path: QLineEdit
    time_col_spin: QSpinBox
    value_col_spin: QSpinBox

    # ------------------------------------------------------------------ #
    # Output / info widgets                                                #
    # ------------------------------------------------------------------ #
    joint_combo: QComboBox
    result_text: QTextEdit

    # ------------------------------------------------------------------ #
    # Methods the mixins call on each other                                #
    # ------------------------------------------------------------------ #
    def _update_plot(self, fitted_signal: Signal | None = None) -> None: ...
    def _update_secondary_plot(self, signal: Signal, title: str) -> None: ...
    def _log(self, message: str) -> None: ...

    # ------------------------------------------------------------------ #
    # Processing methods (ProcessingMixin) referenced by UISetupMixin     #
    # ------------------------------------------------------------------ #
    def _generate_signal(self) -> None: ...
    def _fit_function(self) -> None: ...
    def _auto_fit(self) -> None: ...
    def _apply_saturation(self) -> None: ...
    def _update_saturation_preview(self) -> None: ...
    def _show_derivative(self) -> None: ...
    def _show_integral(self) -> None: ...
    def _update_tangent_position(self, value: int) -> None: ...
    def _toggle_tangent(self, state: int) -> None: ...
    def _update_integral_bounds(self) -> None: ...
    def _apply_filter(self) -> None: ...
    def _show_frequency_response(self) -> None: ...
    def _add_noise(self) -> None: ...
    def _reset_signal(self) -> None: ...
    def _browse_file(self) -> None: ...
    def _import_signal(self) -> None: ...
    def _apply_to_joint(self) -> None: ...
    def _export_signal(self) -> None: ...
