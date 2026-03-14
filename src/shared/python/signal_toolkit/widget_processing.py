"""Signal Toolkit Widget Processing Mixin.

Contains all signal generation, fitting, filtering, noise,
import/export, and calculus methods.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from shared.python.safe_eval import safe_eval

from .calculus import (
    Differentiator,
    Integrator,
)
from .core import Signal, SignalGenerator
from .filters import (
    FilterDesigner,
    FilterType,
    apply_filter,
    apply_moving_average,
    apply_savgol,
)
from .fitting import FunctionFitter
from .io import SignalExporter, SignalImporter
from .limits import SaturationMode, apply_saturation
from .noise import NoiseType, add_noise_to_signal

logger = logging.getLogger(__name__)


class ProcessingMixin:
    """Mixin providing signal processing methods for SignalToolkitWidget."""

    def _generate_default_signal(self) -> None:
        """Generate a default signal to start with."""
        t = np.linspace(0, 10, 1000)
        self.current_signal = SignalGenerator.sinusoid(
            t, amplitude=1.0, frequency=1.0, name="default"
        )
        self.original_signal = self.current_signal.copy()
        self._update_plot()  # type: ignore[attr-defined]

    def _generate_signal(self) -> None:
        """Generate signal based on current settings."""
        t = np.linspace(
            self.t_start_spin.value(),  # type: ignore[attr-defined]
            self.t_end_spin.value(),  # type: ignore[attr-defined]
            self.n_points_spin.value(),  # type: ignore[attr-defined]
        )

        signal_type = self.signal_type_combo.currentText()  # type: ignore[attr-defined]

        try:
            if signal_type == "Sinusoid":
                self.current_signal = SignalGenerator.sinusoid(
                    t,
                    amplitude=self.sin_amplitude.value(),  # type: ignore[attr-defined]
                    frequency=self.sin_frequency.value(),  # type: ignore[attr-defined]
                    phase=self.sin_phase.value(),  # type: ignore[attr-defined]
                    offset=self.sin_offset.value(),  # type: ignore[attr-defined]
                )
            elif signal_type == "Cosine":
                self.current_signal = SignalGenerator.cosine(
                    t,
                    amplitude=self.sin_amplitude.value(),  # type: ignore[attr-defined]
                    frequency=self.sin_frequency.value(),  # type: ignore[attr-defined]
                    phase=self.sin_phase.value(),  # type: ignore[attr-defined]
                    offset=self.sin_offset.value(),  # type: ignore[attr-defined]
                )
            elif signal_type == "Polynomial":
                coeffs_str = self.poly_coeffs_input.text()  # type: ignore[attr-defined]
                coeffs = [float(c.strip()) for c in coeffs_str.split(",")]
                self.current_signal = SignalGenerator.polynomial(t, coeffs)
            elif signal_type == "Exponential":
                self.current_signal = SignalGenerator.exponential(
                    t,
                    amplitude=self.exp_amplitude.value(),  # type: ignore[attr-defined]
                    decay_rate=self.exp_decay.value(),  # type: ignore[attr-defined]
                    offset=self.exp_offset.value(),  # type: ignore[attr-defined]
                )
            elif signal_type == "Linear":
                self.current_signal = SignalGenerator.linear(
                    t,
                    slope=self.linear_slope.value(),  # type: ignore[attr-defined]
                    intercept=self.linear_intercept.value(),  # type: ignore[attr-defined]
                )
            elif signal_type == "Step":
                self.current_signal = SignalGenerator.step(
                    t,
                    step_time=self.step_time.value(),  # type: ignore[attr-defined]
                    step_value=self.step_value.value(),  # type: ignore[attr-defined]
                    initial_value=self.step_initial.value(),  # type: ignore[attr-defined]
                )
            elif signal_type == "Chirp":
                self.current_signal = SignalGenerator.chirp(
                    t,
                    f0=self.chirp_f0.value(),  # type: ignore[attr-defined]
                    f1=self.chirp_f1.value(),  # type: ignore[attr-defined]
                    amplitude=self.chirp_amplitude.value(),  # type: ignore[attr-defined]
                )
            elif signal_type == "Square":
                self.current_signal = SignalGenerator.square(
                    t,
                    frequency=self.square_freq.value(),  # type: ignore[attr-defined]
                    amplitude=self.square_amplitude.value(),  # type: ignore[attr-defined]
                    duty_cycle=self.square_duty.value(),  # type: ignore[attr-defined]
                )
            elif signal_type == "Triangle":
                self.current_signal = SignalGenerator.triangle(
                    t,
                    frequency=self.triangle_freq.value(),  # type: ignore[attr-defined]
                    amplitude=self.triangle_amplitude.value(),  # type: ignore[attr-defined]
                )
            elif signal_type == "Custom":
                expr = self.custom_expr.text()  # type: ignore[attr-defined]
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
                    values = safe_eval(expr, safe_dict)
                    self.current_signal = Signal(t, values, name="custom")
                else:
                    return

            self.original_signal = self.current_signal.copy()
            self._update_plot()  # type: ignore[attr-defined]
            self._log(f"Generated {signal_type} signal")  # type: ignore[attr-defined]

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            QMessageBox.warning(self, "Error", f"Failed to generate signal: {e}")  # type: ignore[arg-type]

    def _fit_function(self) -> None:
        """Fit a function to the current signal."""
        if self.current_signal is None:
            return

        fit_type = self.fit_type_combo.currentText()  # type: ignore[attr-defined]
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
                    order=self.fit_poly_order.value(),  # type: ignore[attr-defined]
                )
            elif fit_type == "Custom":
                expr = self.fit_custom_expr.text()  # type: ignore[attr-defined]
                params_str = self.fit_custom_params.text()  # type: ignore[attr-defined]
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
            self._log(  # type: ignore[attr-defined]
                f"Fit: {fit_type}\n"
                f"R^2: {result.r_squared:.4f}\n"
                f"RMSE: {result.rmse:.4f}\n"
                f"Parameters: {result.parameters}"
            )

            # Plot fitted curve
            self._update_plot(fitted_signal=result.fitted_signal)  # type: ignore[attr-defined]

        except (KeyError, ValueError, TypeError) as e:
            QMessageBox.warning(self, "Fit Error", f"Failed to fit: {e}")  # type: ignore[arg-type]

    def _auto_fit(self) -> None:
        """Automatically find the best fit."""
        if self.current_signal is None:
            return

        try:
            fitter = FunctionFitter()
            best_type, result = fitter.auto_fit(self.current_signal)

            self._log(  # type: ignore[attr-defined]
                f"Best fit: {best_type}\n"
                f"R^2: {result.r_squared:.4f}\n"
                f"Parameters: {result.parameters}"
            )

            self._update_plot(fitted_signal=result.fitted_signal)  # type: ignore[attr-defined]

        except (ValueError, TypeError, RuntimeError) as e:
            QMessageBox.warning(self, "Auto-fit Error", f"Failed: {e}")  # type: ignore[arg-type]

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

        mode = mode_map[self.sat_mode_combo.currentText()]  # type: ignore[attr-defined]

        self.current_signal = apply_saturation(
            self.current_signal,
            lower=self.sat_lower.value(),  # type: ignore[attr-defined]
            upper=self.sat_upper.value(),  # type: ignore[attr-defined]
            mode=mode,
            smoothness=self.sat_smoothness.value(),  # type: ignore[attr-defined]
        )

        self._update_plot()  # type: ignore[attr-defined]
        self._log(f"Applied {mode.value} saturation")  # type: ignore[attr-defined]

    def _update_saturation_preview(self) -> None:
        """Update saturation preview if enabled."""
        if not self.original_signal:
            return

        if self.sat_preview_check.isChecked():  # type: ignore[attr-defined]
            # Show preview without modifying current signal
            mode_map = {
                "Hard Clip": SaturationMode.HARD,
                "Soft Clip (tanh)": SaturationMode.SOFT_TANH,  # type: ignore
                "Soft Clip (sigmoid)": SaturationMode.SOFT_SIGMOID,  # type: ignore
                "Polynomial": SaturationMode.POLYNOMIAL,  # type: ignore
            }
            mode = mode_map.get(
                self.sat_mode_combo.currentText(),  # type: ignore
                SaturationMode.HARD,  # type: ignore[attr-defined]
            )

            # Create preview signal
            preview = apply_saturation(
                (
                    self.current_signal.copy()
                    if self.current_signal
                    else self.original_signal.copy()
                ),
                lower=self.sat_lower.value(),  # type: ignore[attr-defined]
                upper=self.sat_upper.value(),  # type: ignore[attr-defined]
                mode=mode,
                smoothness=self.sat_smoothness.value(),  # type: ignore[attr-defined]
            )

            # Show on secondary plot
            self._update_secondary_plot(preview, "Saturation Preview")  # type: ignore[attr-defined]
        else:
            # Clear preview
            self.canvas2.axes.clear()  # type: ignore[attr-defined]
            self.canvas2.draw()  # type: ignore[attr-defined]

    def _show_derivative(self) -> None:
        """Show the derivative of the current signal."""
        if self.current_signal is None:
            return

        diff = Differentiator()
        self.derivative_signal = diff.differentiate(
            self.current_signal,
            order=self.diff_order.value(),  # type: ignore[attr-defined]
        )

        self._update_secondary_plot(self.derivative_signal, "Derivative")  # type: ignore[attr-defined]

    def _show_integral(self) -> None:
        """Show the integral of the current signal."""
        if self.current_signal is None:
            return

        integrator = Integrator()
        result = integrator.integrate(
            self.current_signal,
            lower_bound=self.int_lower.value(),  # type: ignore[attr-defined]
            upper_bound=self.int_upper.value(),  # type: ignore[attr-defined]
        )

        self.integral_signal = result.cumulative_signal
        self.integral_value_label.setText(f"Integral: {result.value:.4f}")  # type: ignore[attr-defined]

        self._update_secondary_plot(self.integral_signal, "Integral")  # type: ignore[attr-defined]

    def _update_tangent_position(self, value: int) -> None:
        """Update tangent line position from slider."""
        assert value is not None, "value must be provided"
        assert value is not None, "value must be provided"
        if self.current_signal is None:
            return

        t_range = self.current_signal.time[-1] - self.current_signal.time[0]
        t_point = self.current_signal.time[0] + (value / 100) * t_range

        self.tangent_t_spin.setValue(t_point)  # type: ignore[attr-defined]

        if self.show_tangent_check.isChecked():  # type: ignore[attr-defined]
            self._update_plot()  # type: ignore[attr-defined]

    def _toggle_tangent(self, state: int) -> None:
        """Toggle tangent line display."""
        self._update_plot()  # type: ignore[attr-defined]

    def _update_integral_bounds(self) -> None:
        """Update integral bounds from sliders."""
        if self.current_signal is None:
            return

        t_range = self.current_signal.time[-1] - self.current_signal.time[0]
        t0 = self.current_signal.time[0]

        lower = t0 + (self.int_lower_slider.value() / 100) * t_range  # type: ignore[attr-defined]
        upper = t0 + (self.int_upper_slider.value() / 100) * t_range  # type: ignore[attr-defined]

        self.int_lower.setValue(lower)  # type: ignore[attr-defined]
        self.int_upper.setValue(upper)  # type: ignore[attr-defined]

    def _apply_filter(self) -> None:
        """Apply filter to the signal."""
        if self.current_signal is None:
            return

        design = self.filter_design_combo.currentText()  # type: ignore[attr-defined]
        filter_type = self.filter_type_combo.currentText().lower()  # type: ignore[attr-defined]

        try:
            if design in ("Moving Average", "Savitzky-Golay", "Median", "Gaussian"):
                window = self.filter_window.value()  # type: ignore[attr-defined]
                if window % 2 == 0:
                    window += 1

                if design == "Moving Average":
                    self.current_signal = apply_moving_average(
                        self.current_signal, window
                    )
                elif design == "Savitzky-Golay":
                    self.current_signal = apply_savgol(self.current_signal, window, 3)
                elif design == "Median":
                    from .filters import (
                        apply_median_filter,
                    )

                    self.current_signal = apply_median_filter(
                        self.current_signal, window
                    )
                elif design == "Gaussian":
                    from .filters import (
                        apply_gaussian_smoothing,
                    )

                    self.current_signal = apply_gaussian_smoothing(
                        self.current_signal, window / 3
                    )
            else:
                # IIR filters
                fs = self.current_signal.fs
                cutoff = self.filter_cutoff.value()  # type: ignore[attr-defined]
                order = self.filter_order.value()  # type: ignore[attr-defined]

                if filter_type in ("bandpass", "bandstop"):
                    cutoff = (cutoff, self.filter_cutoff2.value())  # type: ignore[attr-defined]

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

            self._update_plot()  # type: ignore[attr-defined]
            self._log(f"Applied {design} {filter_type} filter")  # type: ignore[attr-defined]

        except ImportError as e:
            QMessageBox.warning(self, "Filter Error", f"Failed: {e}")  # type: ignore[arg-type]

    def _show_frequency_response(self) -> None:
        """Show frequency response of the current filter settings."""
        if self.current_signal is None:
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "No Signal",
                "Please generate or load a signal first.",
            )
            return

        design = self.filter_design_combo.currentText()  # type: ignore[attr-defined]

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
            from scipy import signal as scipy_signal

            filter_type = self.filter_type_combo.currentText().lower()  # type: ignore[attr-defined]
            fs = self.current_signal.fs
            cutoff = self.filter_cutoff.value()  # type: ignore[attr-defined]
            order = self.filter_order.value()  # type: ignore[attr-defined]

            if filter_type in ("bandpass", "bandstop"):
                cutoff = (cutoff, self.filter_cutoff2.value())  # type: ignore[attr-defined]

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
            w, h = scipy_signal.freqz(spec.b_coeffs, spec.a_coeffs, fs=fs)  # type: ignore

            # Plot on secondary canvas
            self.canvas2.axes.clear()  # type: ignore[attr-defined]
            self.canvas2.setup_dark_theme()  # type: ignore[attr-defined]

            self.canvas2.axes.semilogy(w, np.abs(h), color="#4ecdc4", linewidth=1.5)  # type: ignore[attr-defined]
            self.canvas2.axes.set_title("Frequency Response", fontsize=10)  # type: ignore[attr-defined]
            self.canvas2.axes.set_xlabel("Frequency (Hz)")  # type: ignore[attr-defined]
            self.canvas2.axes.set_ylabel("Magnitude")  # type: ignore[attr-defined]
            self.canvas2.axes.grid(True, alpha=0.3)  # type: ignore[attr-defined]
            self.canvas2.draw()  # type: ignore[attr-defined]

            self._log(f"Showing frequency response for {design} {filter_type}")  # type: ignore[attr-defined]

        except ImportError as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to compute frequency response: {e}",  # type: ignore[arg-type]
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

        noise_type = noise_map[self.noise_type_combo.currentText()]  # type: ignore[attr-defined]

        if self.noise_use_snr.isChecked():  # type: ignore[attr-defined]
            self.current_signal = add_noise_to_signal(
                self.current_signal,
                noise_type=noise_type,
                snr_db=self.noise_snr.value(),  # type: ignore[attr-defined]
            )
        else:
            self.current_signal = add_noise_to_signal(
                self.current_signal,
                noise_type=noise_type,
                amplitude=self.noise_amplitude.value(),  # type: ignore[attr-defined]
            )

        self._update_plot()  # type: ignore[attr-defined]
        self._log(f"Added {noise_type.value} noise")  # type: ignore[attr-defined]

    def _reset_signal(self) -> None:
        """Reset to original signal."""
        if self.original_signal:
            self.current_signal = self.original_signal.copy()
            self._update_plot()  # type: ignore[attr-defined]
            self._log("Reset to original signal")  # type: ignore[attr-defined]

    def _browse_file(self) -> None:
        """Browse for a file to import."""
        path, _ = QFileDialog.getOpenFileName(
            self,  # type: ignore[arg-type]
            "Import Signal",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.import_path.setText(path)  # type: ignore[attr-defined]

    def _import_signal(self) -> None:
        """Import signal from file."""
        path = self.import_path.text()  # type: ignore[attr-defined]
        if not path:
            return

        try:
            result = SignalImporter.from_csv(
                path,
                time_column=self.time_col_spin.value(),  # type: ignore[attr-defined]
                value_columns=self.value_col_spin.value(),  # type: ignore[attr-defined]
            )

            if isinstance(result, list):
                self.current_signal = result[0]
            else:
                self.current_signal = result

            self.original_signal = self.current_signal.copy()
            self._update_plot()  # type: ignore[attr-defined]
            self._log(f"Imported signal from {Path(path).name}")  # type: ignore[attr-defined]

        except (PermissionError, OSError) as e:
            QMessageBox.warning(self, "Import Error", f"Failed: {e}")  # type: ignore[arg-type]

    def _apply_to_joint(self) -> None:
        """Apply signal to selected joint."""
        if self.current_signal is None:
            return

        joint = self.joint_combo.currentText()  # type: ignore[attr-defined]

        # Fit polynomial to get coefficients for control system
        fitter = FunctionFitter()
        result = fitter.fit_polynomial(self.current_signal, order=6)

        # Get coefficients in [c0, c1, c2, ...] order
        coeffs = [result.parameters.get(f"c{i}", 0.0) for i in range(7)]

        self.signal_generated.emit(joint, coeffs)  # type: ignore[attr-defined]
        self._log(f"Applied to {joint}: {coeffs}")  # type: ignore[attr-defined]

    def _export_signal(self) -> None:
        """Export current signal to file."""
        if self.current_signal is None:
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
                    SignalExporter.to_json(self.current_signal, path)
                else:
                    SignalExporter.to_csv(self.current_signal, path)
                self._log(f"Exported to {Path(path).name}")  # type: ignore[attr-defined]
            except (PermissionError, OSError) as e:
                QMessageBox.warning(self, "Export Error", f"Failed: {e}")  # type: ignore[arg-type]

    def load_external_signal(self, signal: Signal) -> None:
        """Load a signal from an external source.

        Allows other widgets (e.g., Function Generator) to send
        signals to the toolkit for analysis, filtering, or fitting.

        Args:
            signal: Signal object to load.
        """
        assert signal is not None, "signal must be provided"
        assert signal is not None, "signal must be provided"
        self.current_signal = signal
        self.original_signal = signal.copy()
        self._update_plot()  # type: ignore[attr-defined]
        self._log(  # type: ignore[attr-defined]
            f"Loaded external signal: {signal.name or 'unnamed'} "
            f"({signal.n_samples} samples)"
        )
        self.signal_updated.emit(signal)  # type: ignore[attr-defined]
