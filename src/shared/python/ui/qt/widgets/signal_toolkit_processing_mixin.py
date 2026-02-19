"""Signal Toolkit Widget – signal processing mixin.

Extracts all methods that *operate* on signals (generate, fit, filter,
noise, calculus, saturation, import/export, apply-to-joint) from
``SignalToolkitWidget``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox

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


class SignalToolkitProcessingMixin:
    """Mixin providing signal-processing logic for ``SignalToolkitWidget``."""

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _setup_connections(self: Any) -> None:
        """Setup signal-slot connections."""
        self.signal_type_combo.currentIndexChanged.connect(
            self.param_stack.setCurrentIndex
        )
        self.generate_btn.clicked.connect(self._generate_signal)

        self.fit_btn.clicked.connect(self._fit_function)
        self.auto_fit_btn.clicked.connect(self._auto_fit)

        self.apply_sat_btn.clicked.connect(self._apply_saturation)
        self.sat_preview_check.stateChanged.connect(self._update_saturation_preview)

        self.show_derivative_btn.clicked.connect(self._show_derivative)
        self.show_integral_btn.clicked.connect(self._show_integral)
        self.tangent_slider.valueChanged.connect(self._update_tangent_position)
        self.show_tangent_check.stateChanged.connect(self._toggle_tangent)
        self.int_lower_slider.valueChanged.connect(self._update_integral_bounds)
        self.int_upper_slider.valueChanged.connect(self._update_integral_bounds)

        self.apply_filter_btn.clicked.connect(self._apply_filter)
        self.show_freq_response_btn.clicked.connect(self._show_frequency_response)

        self.add_noise_btn.clicked.connect(self._add_noise)
        self.reset_signal_btn.clicked.connect(self._reset_signal)

        self.browse_btn.clicked.connect(self._browse_file)
        self.import_btn.clicked.connect(self._import_signal)

        self.apply_btn.clicked.connect(self._apply_to_joint)
        self.export_btn.clicked.connect(self._export_signal)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_default_signal(self: Any) -> None:
        """Generate a default signal to start with."""
        t = np.linspace(0, 10, 1000)
        self.current_signal = SignalGenerator.sinusoid(
            t, amplitude=1.0, frequency=1.0, name="default"
        )
        self.original_signal = self.current_signal.copy()
        self._update_plot()

    def _generate_sinusoid(self: Any, t: np.ndarray) -> Signal:
        """Generate a sinusoid signal from current UI parameters."""
        return SignalGenerator.sinusoid(
            t,
            amplitude=self.sin_amplitude.value(),
            frequency=self.sin_frequency.value(),
            phase=self.sin_phase.value(),
            offset=self.sin_offset.value(),
        )

    def _generate_cosine(self: Any, t: np.ndarray) -> Signal:
        """Generate a cosine signal from current UI parameters."""
        return SignalGenerator.cosine(
            t,
            amplitude=self.sin_amplitude.value(),
            frequency=self.sin_frequency.value(),
            phase=self.sin_phase.value(),
            offset=self.sin_offset.value(),
        )

    def _generate_polynomial(self: Any, t: np.ndarray) -> Signal:
        """Generate a polynomial signal from current UI parameters."""
        coeffs_str = self.poly_coeffs_input.text()
        coeffs = [float(c.strip()) for c in coeffs_str.split(",")]
        return SignalGenerator.polynomial(t, coeffs)

    def _generate_exponential(self: Any, t: np.ndarray) -> Signal:
        """Generate an exponential signal from current UI parameters."""
        return SignalGenerator.exponential(
            t,
            amplitude=self.exp_amplitude.value(),
            decay_rate=self.exp_decay.value(),
            offset=self.exp_offset.value(),
        )

    def _generate_linear(self: Any, t: np.ndarray) -> Signal:
        """Generate a linear signal from current UI parameters."""
        return SignalGenerator.linear(
            t,
            slope=self.linear_slope.value(),
            intercept=self.linear_intercept.value(),
        )

    def _generate_step(self: Any, t: np.ndarray) -> Signal:
        """Generate a step signal from current UI parameters."""
        return SignalGenerator.step(
            t,
            step_time=self.step_time.value(),
            step_value=self.step_value.value(),
            initial_value=self.step_initial.value(),
        )

    def _generate_chirp(self: Any, t: np.ndarray) -> Signal:
        """Generate a chirp signal from current UI parameters."""
        return SignalGenerator.chirp(
            t,
            f0=self.chirp_f0.value(),
            f1=self.chirp_f1.value(),
            amplitude=self.chirp_amplitude.value(),
        )

    def _generate_square(self: Any, t: np.ndarray) -> Signal:
        """Generate a square wave signal from current UI parameters."""
        return SignalGenerator.square(
            t,
            frequency=self.square_freq.value(),
            amplitude=self.square_amplitude.value(),
            duty_cycle=self.square_duty.value(),
        )

    def _generate_triangle(self: Any, t: np.ndarray) -> Signal:
        """Generate a triangle wave signal from current UI parameters."""
        return SignalGenerator.triangle(
            t,
            frequency=self.triangle_freq.value(),
            amplitude=self.triangle_amplitude.value(),
        )

    def _generate_custom(self: Any, t: np.ndarray) -> Signal | None:
        """Generate a custom expression signal from current UI parameters."""
        expr = self.custom_expr.text()
        if not expr:
            return None

        from simpleeval import EvalWithCompoundTypes

        safe_functions = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
        }
        safe_names = {
            "pi": np.pi,
            "t": t,
        }
        evaluator = EvalWithCompoundTypes(
            functions=safe_functions,
            names=safe_names,
        )
        values = evaluator.eval(expr)
        return Signal(t, values, name="custom")

    def _generate_signal(self: Any) -> None:
        """Generate signal based on current settings.

        Dispatches to type-specific generator methods for each signal type.
        """
        t = np.linspace(
            self.t_start_spin.value(),
            self.t_end_spin.value(),
            self.n_points_spin.value(),
        )

        signal_type = self.signal_type_combo.currentText()

        generators: dict[str, Any] = {
            "Sinusoid": self._generate_sinusoid,
            "Cosine": self._generate_cosine,
            "Polynomial": self._generate_polynomial,
            "Exponential": self._generate_exponential,
            "Linear": self._generate_linear,
            "Step": self._generate_step,
            "Chirp": self._generate_chirp,
            "Square": self._generate_square,
            "Triangle": self._generate_triangle,
            "Custom": self._generate_custom,
        }

        try:
            generator = generators.get(signal_type)
            if generator is None:
                return

            result = generator(t)
            if result is None:
                return

            self.current_signal = result
            self.original_signal = self.current_signal.copy()
            self._update_plot()
            self._log(f"Generated {signal_type} signal")

        except (ValueError, TypeError, RuntimeError) as e:
            QMessageBox.warning(self, "Error", f"Failed to generate signal: {e}")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_function(self: Any) -> None:
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

            self._log(
                f"Fit: {fit_type}\n"
                f"R^2: {result.r_squared:.4f}\n"
                f"RMSE: {result.rmse:.4f}\n"
                f"Parameters: {result.parameters}"
            )
            self._update_plot(fitted_signal=result.fitted_signal)

        except (RuntimeError, ValueError, OSError) as e:
            QMessageBox.warning(self, "Fit Error", f"Failed to fit: {e}")

    def _auto_fit(self: Any) -> None:
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
        except (RuntimeError, ValueError, OSError) as e:
            QMessageBox.warning(self, "Auto-fit Error", f"Failed: {e}")

    # ------------------------------------------------------------------
    # Saturation
    # ------------------------------------------------------------------

    def _apply_saturation(self: Any) -> None:
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

    def _update_saturation_preview(self: Any) -> None:
        """Update saturation preview if enabled."""
        if not self.original_signal:
            return

        if self.sat_preview_check.isChecked():
            mode_map = {
                "Hard Clip": SaturationMode.HARD,
                "Soft Clip (tanh)": SaturationMode.TANH,
                "Soft Clip (sigmoid)": SaturationMode.SIGMOID,
                "Polynomial": SaturationMode.SOFT,
            }
            mode = mode_map.get(self.sat_mode_combo.currentText(), SaturationMode.HARD)

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
            self._update_secondary_plot(preview, "Saturation Preview")
        else:
            self.canvas2.axes.clear()
            self.canvas2.draw()

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def _show_derivative(self: Any) -> None:
        """Show the derivative of the current signal."""
        if self.current_signal is None:
            return

        diff = Differentiator()
        self.derivative_signal = diff.differentiate(
            self.current_signal, order=self.diff_order.value()
        )
        self._update_secondary_plot(self.derivative_signal, "Derivative")

    def _show_integral(self: Any) -> None:
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

    def _update_tangent_position(self: Any, value: int) -> None:
        """Update tangent line position from slider."""
        if self.current_signal is None:
            return

        t_range = self.current_signal.time[-1] - self.current_signal.time[0]
        t_point = self.current_signal.time[0] + (value / 100) * t_range
        self.tangent_t_spin.setValue(t_point)

        if self.show_tangent_check.isChecked():
            self._update_plot()

    def _toggle_tangent(self: Any, state: int) -> None:
        """Toggle tangent line display."""
        self._update_plot()

    def _update_integral_bounds(self: Any) -> None:
        """Update integral bounds from sliders."""
        if self.current_signal is None:
            return

        t_range = self.current_signal.time[-1] - self.current_signal.time[0]
        t0 = self.current_signal.time[0]

        lower = t0 + (self.int_lower_slider.value() / 100) * t_range
        upper = t0 + (self.int_upper_slider.value() / 100) * t_range

        self.int_lower.setValue(lower)
        self.int_upper.setValue(upper)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _apply_filter(self: Any) -> None:  # noqa: PLR0912
        """Apply filter to the signal."""
        if self.current_signal is None:
            return

        design = self.filter_design_combo.currentText()
        filter_type = self.filter_type_combo.currentText().lower()

        try:
            if design in (
                "Moving Average",
                "Savitzky-Golay",
                "Median",
                "Gaussian",
            ):
                window = self.filter_window.value()
                if window % 2 == 0:
                    window += 1

                if design == "Moving Average":
                    self.current_signal = apply_moving_average(
                        self.current_signal, window
                    )
                elif design == "Savitzky-Golay":
                    self.current_signal = apply_savgol(self.current_signal, window, 3)
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

        except ImportError as e:
            QMessageBox.warning(self, "Filter Error", f"Failed: {e}")

    def _show_frequency_response(self: Any) -> None:
        """Show frequency response of the current filter settings."""
        if self.current_signal is None:
            QMessageBox.information(
                self,
                "No Signal",
                "Please generate or load a signal first.",
            )
            return

        design = self.filter_design_combo.currentText()

        if design in (
            "Moving Average",
            "Savitzky-Golay",
            "Median",
            "Gaussian",
        ):
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

            w, h = scipy_signal.freqz(spec.b, spec.a, fs=fs)

            self.canvas2.axes.clear()
            self.canvas2.setup_dark_theme()
            self.canvas2.axes.semilogy(w, np.abs(h), color="#4ecdc4", linewidth=1.5)
            self.canvas2.axes.set_title("Frequency Response", fontsize=10)
            self.canvas2.axes.set_xlabel("Frequency (Hz)")
            self.canvas2.axes.set_ylabel("Magnitude")
            self.canvas2.axes.grid(True, alpha=0.3)
            self.canvas2.draw()

            self._log(f"Showing frequency response for {design} {filter_type}")

        except ImportError as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to compute frequency response: {e}",
            )

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def _add_noise(self: Any) -> None:
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

    def _reset_signal(self: Any) -> None:
        """Reset to original signal."""
        if self.original_signal:
            self.current_signal = self.original_signal.copy()
            self._update_plot()
            self._log("Reset to original signal")

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def _browse_file(self: Any) -> None:
        """Browse for a file to import."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Signal",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.import_path.setText(path)

    def _import_signal(self: Any) -> None:
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

        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Failed: {e}")

    def _apply_to_joint(self: Any) -> None:
        """Apply signal to selected joint."""
        if self.current_signal is None:
            return

        joint = self.joint_combo.currentText()

        fitter = FunctionFitter()
        result = fitter.fit_polynomial(self.current_signal, order=6)

        coeffs = [result.parameters.get(f"c{i}", 0.0) for i in range(7)]

        self.signal_generated.emit(joint, coeffs)
        self._log(f"Applied to {joint}: {coeffs}")

    def _export_signal(self: Any) -> None:
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
            except (FileNotFoundError, OSError) as e:
                QMessageBox.warning(self, "Export Error", f"Failed: {e}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _update_plot(
        self: Any,
        fitted_signal: Signal | None = None,
    ) -> None:
        """Update the main plot."""
        self.canvas.axes.clear()
        self.canvas.setup_dark_theme()

        if self.current_signal is None:
            self.canvas.draw()
            return

        self.canvas.axes.plot(
            self.current_signal.time,
            self.current_signal.values,
            color="#4da6ff",
            linewidth=1.5,
            label="Signal",
        )

        if fitted_signal:
            self.canvas.axes.plot(
                fitted_signal.time,
                fitted_signal.values,
                color="#ff6b6b",
                linewidth=2,
                linestyle="--",
                label="Fit",
            )

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
        self: Any,
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

    def _log(self: Any, message: str) -> None:
        """Log a message to the result text area."""
        self.result_text.append(message)

    def set_joints(self: Any, joints: list[str]) -> None:
        """Set the list of available joints."""
        self.joint_names = joints
        self.joint_combo.clear()
        self.joint_combo.addItems(joints)
