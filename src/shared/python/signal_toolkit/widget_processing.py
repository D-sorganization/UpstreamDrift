# ruff: noqa: E501
# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""Signal Toolkit Widget Processing Mixin.

Contains all signal generation, fitting, filtering, noise,
import/export, and calculus methods.  Undo/Redo support is provided
via a signal history stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QWidget,
)

from shared.python.safe_eval import safe_eval

from .calculus import (
    Differentiator,
    Integrator,
)
from .core import Signal, SignalGenerator
from .filters import (
    FilterDesigner,
    FilterSpec,
    FilterType,
    apply_filter,
    apply_moving_average,
    apply_savgol,
)
from .fitting import FunctionFitter
from .io import SignalExporter, SignalImporter
from .limits import SaturationMode, apply_saturation
from .noise import NoiseType, add_noise_to_signal
from .series import SeriesExpansion
from .widget_protocol import WidgetProtocol

logger = logging.getLogger(__name__)

# Maximum undo history depth
_MAX_HISTORY = 50


class SignalGenerationError(ValueError):
    """Raised when signal generation cannot produce a valid signal."""


class ProcessingMixin:
    """Mixin providing signal processing methods for SignalToolkitWidget."""

    # ------------------------------------------------------------------
    # Undo / Redo helpers (Issue #1276)
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        """Push the current signal state onto the undo stack."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return
        if not hasattr(self, "_undo_stack"):
            self._undo_stack: list[Signal] = []
        if not hasattr(self, "_redo_stack"):
            self._redo_stack: list[Signal] = []
        self._undo_stack.append(w.current_signal.copy())
        if len(self._undo_stack) > _MAX_HISTORY:
            self._undo_stack.pop(0)
        # Any new action clears the redo stack
        self._redo_stack.clear()

    def undo(self) -> None:
        """Undo the last processing operation."""
        w = cast(WidgetProtocol, self)
        if not hasattr(self, "_undo_stack") or not self._undo_stack:
            return
        if w.current_signal is not None:
            if not hasattr(self, "_redo_stack"):
                self._redo_stack = []
            self._redo_stack.append(w.current_signal.copy())
        w.current_signal = self._undo_stack.pop()
        w._update_plot()
        w._log("Undo")

    def redo(self) -> None:
        """Redo the last undone processing operation."""
        w = cast(WidgetProtocol, self)
        if not hasattr(self, "_redo_stack") or not self._redo_stack:
            return
        if w.current_signal is not None:
            if not hasattr(self, "_undo_stack"):
                self._undo_stack = []
            self._undo_stack.append(w.current_signal.copy())
        w.current_signal = self._redo_stack.pop()
        w._update_plot()
        w._log("Redo")

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_default_signal(self) -> None:
        """Generate a default signal to start with."""
        w = cast(WidgetProtocol, self)
        t = np.linspace(0, 10, 1000)
        w.current_signal = SignalGenerator.sinusoid(
            t, amplitude=1.0, frequency=1.0, name="default"
        )
        w.original_signal = w.current_signal.copy()
        w._update_plot()

    def _generate_signal(self) -> None:
        """Generate signal based on current settings."""
        w = cast(WidgetProtocol, self)
        t = np.linspace(
            w.t_start_spin.value(),
            w.t_end_spin.value(),
            w.n_points_spin.value(),
        )

        signal_type = w.signal_type_combo.currentText()

        try:
            self._generate_signal_or_raise(w, t, signal_type)
        except SignalGenerationError as exc:
            self._report_generation_error(str(exc))

    def _generate_signal_or_raise(
        self,
        w: WidgetProtocol,
        t: np.ndarray,
        signal_type: str,
    ) -> None:
        """Generate the selected signal without opening GUI dialogs."""
        try:
            if signal_type == "Sinusoid":
                w.current_signal = SignalGenerator.sinusoid(
                    t,
                    amplitude=w.sin_amplitude.value(),
                    frequency=w.sin_frequency.value(),
                    phase=w.sin_phase.value(),
                    offset=w.sin_offset.value(),
                )
            elif signal_type == "Cosine":
                w.current_signal = SignalGenerator.cosine(
                    t,
                    amplitude=w.sin_amplitude.value(),
                    frequency=w.sin_frequency.value(),
                    phase=w.sin_phase.value(),
                    offset=w.sin_offset.value(),
                )
            elif signal_type == "Polynomial":
                coeffs_str = w.poly_coeffs_input.text()
                coeffs = [float(c.strip()) for c in coeffs_str.split(",")]
                w.current_signal = SignalGenerator.polynomial(t, coeffs)
            elif signal_type == "Exponential":
                w.current_signal = SignalGenerator.exponential(
                    t,
                    amplitude=w.exp_amplitude.value(),
                    decay_rate=w.exp_decay.value(),
                    offset=w.exp_offset.value(),
                )
            elif signal_type == "Linear":
                w.current_signal = SignalGenerator.linear(
                    t,
                    slope=w.linear_slope.value(),
                    intercept=w.linear_intercept.value(),
                )
            elif signal_type == "Step":
                w.current_signal = SignalGenerator.step(
                    t,
                    step_time=w.step_time.value(),
                    step_value=w.step_value.value(),
                    initial_value=w.step_initial.value(),
                )
            elif signal_type == "Chirp":
                w.current_signal = SignalGenerator.chirp(
                    t,
                    f0=w.chirp_f0.value(),
                    f1=w.chirp_f1.value(),
                    amplitude=w.chirp_amplitude.value(),
                )
            elif signal_type == "Square":
                w.current_signal = SignalGenerator.square(
                    t,
                    frequency=w.square_freq.value(),
                    amplitude=w.square_amplitude.value(),
                    duty_cycle=w.square_duty.value(),
                )
            elif signal_type == "Triangle":
                w.current_signal = SignalGenerator.triangle(
                    t,
                    frequency=w.triangle_freq.value(),
                    amplitude=w.triangle_amplitude.value(),
                )
            elif signal_type == "Custom":
                expr = w.custom_expr.text()
                if not expr:
                    return
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
                values = safe_eval(expr, safe_dict)
                w.current_signal = Signal(t, values, name="custom")
            else:
                return

            w.original_signal = w.current_signal.copy()
            w._update_plot()
            w._log(f"Generated {signal_type} signal")

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            raise SignalGenerationError(f"Failed to generate signal: {e}") from e

    def _report_generation_error(self, message: str) -> None:
        """Report signal-generation errors without coupling to QMessageBox."""
        handler = getattr(self, "show_generation_error", None)
        if callable(handler):
            handler(message)
            return
        logger.warning(message)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_function(self) -> None:
        """Fit a function to the current signal."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        fit_type = w.fit_type_combo.currentText()
        fitter = FunctionFitter()

        try:
            if fit_type == "Sinusoid":
                result = fitter.fit_sinusoid(w.current_signal)
            elif fit_type == "Cosine":
                result = fitter.fit_cosine(w.current_signal)
            elif fit_type == "Exponential Decay":
                result = fitter.fit_exponential_decay(w.current_signal)
            elif fit_type == "Exponential Growth":
                result = fitter.fit_exponential_growth(w.current_signal)
            elif fit_type == "Linear":
                result = fitter.fit_linear(w.current_signal)
            elif fit_type == "Polynomial":
                result = fitter.fit_polynomial(
                    w.current_signal,
                    order=w.fit_poly_order.value(),
                )
            elif fit_type == "Custom":
                expr = w.fit_custom_expr.text()
                params_str = w.fit_custom_params.text()
                if expr and params_str:
                    params = [p.strip() for p in params_str.split(",")]
                    result = fitter.fit_custom_expression(
                        w.current_signal, expr, params
                    )
                else:
                    return
            else:
                return

            # Display results
            w._log(
                f"Fit: {fit_type}\n"
                f"R^2: {result.r_squared:.4f}\n"
                f"RMSE: {result.rmse:.4f}\n"
                f"Parameters: {result.parameters}"
            )

            # Plot fitted curve
            w._update_plot(fitted_signal=result.fitted_signal)

        except (KeyError, ValueError, TypeError) as e:
            QMessageBox.warning(self, "Fit Error", f"Failed to fit: {e}")  # type: ignore[arg-type]

    def _auto_fit(self) -> None:
        """Automatically find the best fit."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        try:
            fitter = FunctionFitter()
            best_type, result = fitter.auto_fit(w.current_signal)

            w._log(
                f"Best fit: {best_type}\n"
                f"R^2: {result.r_squared:.4f}\n"
                f"Parameters: {result.parameters}"
            )

            w._update_plot(fitted_signal=result.fitted_signal)

        except (ValueError, TypeError, RuntimeError) as e:
            QMessageBox.warning(self, "Auto-fit Error", f"Failed: {e}")  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Limits / Saturation
    # ------------------------------------------------------------------

    def _apply_saturation(self) -> None:
        """Apply saturation to the signal."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        self._push_undo()

        mode_map = {
            "Hard": SaturationMode.HARD,
            "Soft": SaturationMode.SOFT,
            "Tanh": SaturationMode.TANH,
            "Sigmoid": SaturationMode.SIGMOID,
            "Atan": SaturationMode.ATAN,
            "Cubic": SaturationMode.CUBIC,
            "Exponential": SaturationMode.EXPONENTIAL,
        }

        mode = mode_map[w.sat_mode_combo.currentText()]

        w.current_signal = apply_saturation(
            w.current_signal,
            lower=w.sat_lower.value(),
            upper=w.sat_upper.value(),
            mode=mode,
            smoothness=w.sat_smoothness.value(),
        )

        w._update_plot()
        w._log(f"Applied {mode.value} saturation")

    def _update_saturation_preview(self) -> None:
        """Update saturation preview if enabled."""
        w = cast(WidgetProtocol, self)
        if not w.original_signal:
            return

        if w.sat_preview_check.isChecked():
            mode_map = {
                "Hard": SaturationMode.HARD,
                "Soft": SaturationMode.SOFT,
                "Tanh": SaturationMode.TANH,
                "Sigmoid": SaturationMode.SIGMOID,
                "Atan": SaturationMode.ATAN,
                "Cubic": SaturationMode.CUBIC,
                "Exponential": SaturationMode.EXPONENTIAL,
            }
            mode = mode_map.get(
                w.sat_mode_combo.currentText(),
                SaturationMode.HARD,
            )

            # Create preview signal
            preview = apply_saturation(
                (
                    w.current_signal.copy()
                    if w.current_signal
                    else w.original_signal.copy()
                ),
                lower=w.sat_lower.value(),
                upper=w.sat_upper.value(),
                mode=mode,
                smoothness=w.sat_smoothness.value(),
            )

            # Show on secondary plot
            w._update_secondary_plot(preview, "Saturation Preview")
        else:
            # Clear preview
            w.canvas2.axes.clear()
            w.canvas2.draw()

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def _show_derivative(self) -> None:
        """Show the derivative of the current signal."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        diff = Differentiator()
        derivative = diff.differentiate(
            w.current_signal,
            order=w.diff_order.value(),
        )
        w.derivative_signal = derivative

        w._update_secondary_plot(derivative, "Derivative")

    def _show_integral(self) -> None:
        """Show the integral of the current signal."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        integrator = Integrator()
        result = integrator.integrate(
            w.current_signal,
            lower_bound=w.int_lower.value(),
            upper_bound=w.int_upper.value(),
        )

        w.integral_signal = result.cumulative_signal
        w.integral_value_label.setText(f"Integral: {result.value:.4f}")

        if result.cumulative_signal is not None:
            w._update_secondary_plot(result.cumulative_signal, "Integral")

    def _export_calculus_result(self) -> None:
        """Export the derivative or integral signal to a file (Issue #1281)."""
        w = cast(WidgetProtocol, self)
        signal_to_export = w.derivative_signal or w.integral_signal
        if signal_to_export is None:
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Nothing to Export",
                "Compute a derivative or integral first.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,  # type: ignore[arg-type]
            "Export Calculus Result",
            "",
            "CSV Files (*.csv);;JSON Files (*.json)",
        )

        if path:
            try:
                if path.endswith(".json"):
                    SignalExporter.to_json(signal_to_export, path)
                else:
                    SignalExporter.to_csv(signal_to_export, path)
                w._log(f"Exported calculus result to {Path(path).name}")
            except (PermissionError, OSError) as e:
                QMessageBox.warning(self, "Export Error", f"Failed: {e}")  # type: ignore[arg-type]

    def _update_tangent_position(self, value: int) -> None:
        """Update tangent line position from slider."""
        if value is None:
            raise ValueError("value must be provided")
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        t_range = w.current_signal.time[-1] - w.current_signal.time[0]
        t_point = w.current_signal.time[0] + (value / 100) * t_range

        w.tangent_t_spin.setValue(t_point)

        if w.show_tangent_check.isChecked():
            w._update_plot()

    def _toggle_tangent(self, state: int) -> None:
        """Toggle tangent line display."""
        w = cast(WidgetProtocol, self)
        w._update_plot()

    def _update_integral_bounds(self) -> None:
        """Update integral bounds from sliders."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        t_range = w.current_signal.time[-1] - w.current_signal.time[0]
        t0 = w.current_signal.time[0]

        lower = t0 + (w.int_lower_slider.value() / 100) * t_range
        upper = t0 + (w.int_upper_slider.value() / 100) * t_range

        w.int_lower.setValue(lower)
        w.int_upper.setValue(upper)

    # ------------------------------------------------------------------
    # Series (Issue #1279)
    # ------------------------------------------------------------------

    def _compute_series(self) -> None:
        """Compute Taylor/Maclaurin series approximation of the signal."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        series_type = w.series_type_combo.currentText()
        center = w.series_center_spin.value() if series_type == "Taylor" else 0.0
        n_terms = w.series_terms_spin.value()

        try:
            # Use interpolation to create a callable from the signal
            from scipy.interpolate import interp1d

            f_interp = interp1d(
                w.current_signal.time,
                w.current_signal.values,
                kind="cubic",
                fill_value="extrapolate",
            )

            expansion = SeriesExpansion(max_terms=n_terms)
            result = expansion.get_series_result(f_interp, center, n_terms)

            # Evaluate the series approximation over the signal's time range
            approx_values = result.function(w.current_signal.time)
            approx_signal = Signal(
                time=w.current_signal.time,
                values=np.asarray(approx_values, dtype=float),
                name=f"{series_type} Series ({n_terms} terms)",
            )

            # Show on secondary plot
            w._update_secondary_plot(
                approx_signal,
                f"{series_type} Series (center={center:.2f}, {n_terms} terms)",
            )

            # Log coefficients
            coeffs_str = ", ".join(f"{c:.4f}" for c in result.coefficients[:n_terms])
            w._log(
                f"{series_type} Series at a={center:.2f}:\nCoefficients: [{coeffs_str}]"
            )

        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, "Series Error", f"Failed: {e}")  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _get_filter_spec(self) -> FilterSpec | None:
        """Build a FilterSpec from the current UI settings.

        Returns:
            FilterSpec if IIR design is selected, else None.
        """
        w = cast(WidgetProtocol, self)
        design = w.filter_design_combo.currentText()
        filter_type = w.filter_type_combo.currentText().lower()

        if design in ("Moving Average", "Savitzky-Golay", "Median", "Gaussian"):
            return None

        fs = w.current_signal.fs if w.current_signal else 100.0
        cutoff: float | tuple[float, float] = w.filter_cutoff.value()
        order = w.filter_order.value()

        if filter_type in ("bandpass", "bandstop"):
            cutoff = (w.filter_cutoff.value(), w.filter_cutoff2.value())

        ft = FilterType(filter_type)

        if design == "Butterworth":
            return FilterDesigner.butterworth(ft, cutoff, fs, order)
        if design == "Chebyshev I":
            return FilterDesigner.chebyshev1(ft, cutoff, fs, order)
        if design == "Chebyshev II":
            return FilterDesigner.chebyshev2(ft, cutoff, fs, order)
        if design == "Elliptic":
            return FilterDesigner.elliptic(ft, cutoff, fs, order)
        if design == "Bessel":
            return FilterDesigner.bessel(ft, cutoff, fs, order)
        return None

    def _apply_filter(self) -> None:
        """Apply filter to the signal."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        self._push_undo()

        design = w.filter_design_combo.currentText()
        filter_type = w.filter_type_combo.currentText().lower()

        try:
            if design in ("Moving Average", "Savitzky-Golay", "Median", "Gaussian"):
                window = w.filter_window.value()
                if window % 2 == 0:
                    window += 1

                if design == "Moving Average":
                    w.current_signal = apply_moving_average(w.current_signal, window)
                elif design == "Savitzky-Golay":
                    w.current_signal = apply_savgol(w.current_signal, window, 3)
                elif design == "Median":
                    from .filters import (
                        apply_median_filter,
                    )

                    w.current_signal = apply_median_filter(w.current_signal, window)
                elif design == "Gaussian":
                    from .filters import (
                        apply_gaussian_smoothing,
                    )

                    w.current_signal = apply_gaussian_smoothing(
                        w.current_signal, window / 3
                    )
            else:
                spec = self._get_filter_spec()
                if spec is None:
                    return
                w.current_signal = apply_filter(w.current_signal, spec)

            w._update_plot()
            w._log(f"Applied {design} {filter_type} filter")

        except (ValueError, ImportError) as e:
            QMessageBox.warning(cast("QWidget", self), "Filter Error", f"Failed: {e}")

    def _show_frequency_response(self) -> None:
        """Show frequency response of the current filter settings (Issue #1278).

        Renders a Bode-style magnitude plot on the secondary canvas
        instead of opening a separate matplotlib window.
        """
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "No Signal",
                "Please generate or load a signal first.",
            )
            return

        design = w.filter_design_combo.currentText()

        # Non-IIR filters don't have a traditional frequency response
        if design in ("Moving Average", "Savitzky-Golay", "Median", "Gaussian"):
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Frequency Response",
                f"{design} filters are FIR/smoothing filters.\n"
                "Use IIR filter designs (Butterworth, Chebyshev, etc.) "
                "to view frequency response.",
            )
            return

        try:
            spec = self._get_filter_spec()
            if spec is None:
                return

            # Use FilterSpec.get_frequency_response (the correct API)
            frequencies, magnitude, phase = spec.get_frequency_response(512)

            filter_type = w.filter_type_combo.currentText().lower()
            title = f"{design} {filter_type} — Frequency Response"

            # Render on secondary canvas via the new Bode plot method
            w._update_frequency_response_plot(frequencies, magnitude, phase, title)

            w._log(f"Showing frequency response for {design} {filter_type}")

        except (ValueError, ImportError) as e:
            QMessageBox.warning(
                cast("QWidget", self),
                "Error",
                f"Failed to compute frequency response: {e}",
            )

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def _add_noise(self) -> None:
        """Add noise to the signal."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        self._push_undo()

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

        noise_type = noise_map[w.noise_type_combo.currentText()]

        if w.noise_use_snr.isChecked():
            w.current_signal = add_noise_to_signal(
                w.current_signal,
                noise_type=noise_type,
                snr_db=w.noise_snr.value(),
            )
        else:
            w.current_signal = add_noise_to_signal(
                w.current_signal,
                noise_type=noise_type,
                amplitude=w.noise_amplitude.value(),
            )

        w._update_plot()
        w._log(f"Added {noise_type.value} noise")

    def _reset_signal(self) -> None:
        """Reset to original signal."""
        w = cast(WidgetProtocol, self)
        if w.original_signal:
            self._push_undo()
            w.current_signal = w.original_signal.copy()
            w._update_plot()
            w._log("Reset to original signal")

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def _browse_file(self) -> None:
        """Browse for a file to import."""
        path, _ = QFileDialog.getOpenFileName(
            self,  # type: ignore[arg-type]
            "Import Signal",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            w = cast(WidgetProtocol, self)
            w.import_path.setText(path)

    def _import_signal(self) -> None:
        """Import signal from file."""
        w = cast(WidgetProtocol, self)
        path = w.import_path.text()
        if not path:
            return

        try:
            result = SignalImporter.from_csv(
                path,
                time_column=w.time_col_spin.value(),
                value_columns=w.value_col_spin.value(),
            )

            if isinstance(result, list):
                w.current_signal = result[0]
            else:
                w.current_signal = result

            w.original_signal = w.current_signal.copy()
            w._update_plot()
            w._log(f"Imported signal from {Path(path).name}")

        except (PermissionError, OSError) as e:
            QMessageBox.warning(self, "Import Error", f"Failed: {e}")  # type: ignore[arg-type]

    def _apply_to_joint(self) -> None:
        """Apply signal to selected joint."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        joint = w.joint_combo.currentText()

        # Fit polynomial to get coefficients for control system
        fitter = FunctionFitter()
        result = fitter.fit_polynomial(w.current_signal, order=6)

        # Get coefficients in [c0, c1, c2, ...] order
        coeffs = [result.parameters.get(f"c{i}", 0.0) for i in range(7)]

        w.signal_generated.emit(joint, coeffs)
        w._log(f"Applied to {joint}: {coeffs}")

    def _export_signal(self) -> None:
        """Export current signal to file."""
        w = cast(WidgetProtocol, self)
        if w.current_signal is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,  # type: ignore[arg-type]
            "Export Signal",
            "",
            "CSV Files (*.csv);;JSON Files (*.json)",
        )

        if path:
            try:
                if path.endswith(".json"):
                    SignalExporter.to_json(w.current_signal, path)
                else:
                    SignalExporter.to_csv(w.current_signal, path)
                w._log(f"Exported to {Path(path).name}")
            except (PermissionError, OSError) as e:
                QMessageBox.warning(self, "Export Error", f"Failed: {e}")  # type: ignore[arg-type]

    def load_external_signal(self, signal: Signal) -> None:
        """Load a signal from an external source.

        Allows other widgets (e.g., Function Generator) to send
        signals to the toolkit for analysis, filtering, or fitting.

        Args:
            signal: Signal object to load.
        """
        if signal is None:
            raise ValueError("signal must be provided")
        w = cast(WidgetProtocol, self)
        w.current_signal = signal
        w.original_signal = signal.copy()
        w._update_plot()
        w._log(
            f"Loaded external signal: {signal.name or 'unnamed'} "
            f"({signal.n_samples} samples)"
        )
        w.signal_updated.emit(signal)
