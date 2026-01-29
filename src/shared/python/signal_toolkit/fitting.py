"""Function fitting utilities for signal analysis.

This module provides various function fitters including sinusoidal,
exponential, linear, polynomial, and custom function fitting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from src.shared.python.signal_toolkit.core import Signal


@dataclass
class FitResult:
    """Result of a function fitting operation.

    Attributes:
        parameters: Dictionary of fitted parameter names and values.
        covariance: Covariance matrix of the fit (if available).
        r_squared: Coefficient of determination (R^2).
        rmse: Root mean square error.
        fitted_signal: Signal with fitted values.
        residuals: Residuals (original - fitted).
        success: Whether the fit converged successfully.
        message: Fit status message.
    """

    parameters: dict[str, float]
    covariance: np.ndarray | None
    r_squared: float
    rmse: float
    fitted_signal: Signal
    residuals: np.ndarray
    success: bool = True
    message: str = ""

    def get_function_string(self) -> str:
        """Get a string representation of the fitted function."""
        return f"Fitted function with R^2={self.r_squared:.4f}"


class SinusoidFitter:
    """Fits sinusoidal functions to data.

    Fits: y = amplitude * sin(2*pi*frequency*t + phase) + offset
    """

    @staticmethod
    def _model(
        t: np.ndarray,
        amplitude: float,
        frequency: float,
        phase: float,
        offset: float,
    ) -> np.ndarray:
        """Sinusoidal model function."""
        return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    @staticmethod
    def estimate_initial_params(
        t: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Estimate initial parameters using FFT-based frequency estimation.

        Args:
            t: Time array.
            y: Signal values.

        Returns:
            Tuple of (amplitude, frequency, phase, offset) estimates.
        """
        # Offset estimate
        offset = np.mean(y)
        y_centered = y - offset

        # Amplitude estimate
        amplitude = np.std(y_centered) * np.sqrt(2)

        # Frequency estimate using FFT
        n = len(t)
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt

        fft_vals = np.fft.rfft(y_centered)
        freqs = np.fft.rfftfreq(n, dt)

        # Find dominant frequency (skip DC)
        magnitudes = np.abs(fft_vals)
        peak_idx = np.argmax(magnitudes[1:]) + 1
        frequency = freqs[peak_idx]

        # Phase estimate
        phase = np.angle(fft_vals[peak_idx])

        return amplitude, max(frequency, 0.001), phase, offset

    def fit(
        self,
        signal: Signal,
        initial_guess: tuple[float, float, float, float] | None = None,
        bounds: tuple[tuple, tuple] | None = None,
    ) -> FitResult:
        """Fit a sinusoidal function to the signal.

        Args:
            signal: Input signal to fit.
            initial_guess: Optional (amplitude, frequency, phase, offset).
            bounds: Optional bounds ((lower), (upper)) for each parameter.

        Returns:
            FitResult with fitted parameters and statistics.
        """
        t = signal.time - signal.time[0]  # Shift to start at 0
        y = signal.values

        if initial_guess is None:
            initial_guess = self.estimate_initial_params(t, y)

        if bounds is None:
            # Default bounds
            bounds = (
                (0, 0.001, -2 * np.pi, -np.inf),  # Lower bounds
                (np.inf, signal.fs / 2, 2 * np.pi, np.inf),  # Upper bounds
            )

        try:
            popt, pcov = optimize.curve_fit(
                self._model,
                t,
                y,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
            )
            success = True
            message = "Fit converged successfully"
        except Exception as e:
            popt = np.array(initial_guess)
            pcov = None
            success = False
            message = f"Fit failed: {e}"

        # Compute fitted values and statistics
        fitted_values = self._model(t, *popt)
        residuals = y - fitted_values

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # RMSE
        rmse = np.sqrt(np.mean(residuals**2))

        fitted_signal = Signal(
            time=signal.time,
            values=fitted_values,
            name=f"{signal.name}_sinusoid_fit",
            units=signal.units,
        )

        return FitResult(
            parameters={
                "amplitude": popt[0],
                "frequency": popt[1],
                "phase": popt[2],
                "offset": popt[3],
            },
            covariance=pcov,
            r_squared=r_squared,
            rmse=rmse,
            fitted_signal=fitted_signal,
            residuals=residuals,
            success=success,
            message=message,
        )

    def get_function_string(self, params: dict[str, float]) -> str:
        """Get string representation of the fitted function."""
        return (
            f"y = {params['amplitude']:.4f} * sin(2*pi*{params['frequency']:.4f}*t "
            f"+ {params['phase']:.4f}) + {params['offset']:.4f}"
        )


class CosineFitter(SinusoidFitter):
    """Fits cosine functions to data.

    Fits: y = amplitude * cos(2*pi*frequency*t + phase) + offset
    """

    @staticmethod
    def _model(
        t: np.ndarray,
        amplitude: float,
        frequency: float,
        phase: float,
        offset: float,
    ) -> np.ndarray:
        """Cosine model function."""
        return amplitude * np.cos(2 * np.pi * frequency * t + phase) + offset

    def get_function_string(self, params: dict[str, float]) -> str:
        """Get string representation of the fitted function."""
        return (
            f"y = {params['amplitude']:.4f} * cos(2*pi*{params['frequency']:.4f}*t "
            f"+ {params['phase']:.4f}) + {params['offset']:.4f}"
        )


class ExponentialFitter:
    """Fits exponential functions to data.

    Supports multiple exponential forms:
    - Decay: y = amplitude * exp(-decay_rate * t) + offset
    - Growth: y = amplitude * (1 - exp(-growth_rate * t)) + offset
    - General: y = a * exp(b * t) + c
    """

    @staticmethod
    def _decay_model(
        t: np.ndarray,
        amplitude: float,
        decay_rate: float,
        offset: float,
    ) -> np.ndarray:
        """Exponential decay model."""
        return amplitude * np.exp(-decay_rate * t) + offset

    @staticmethod
    def _growth_model(
        t: np.ndarray,
        amplitude: float,
        growth_rate: float,
        offset: float,
    ) -> np.ndarray:
        """Exponential growth (1 - exp) model."""
        return amplitude * (1 - np.exp(-growth_rate * t)) + offset

    @staticmethod
    def _general_model(
        t: np.ndarray,
        a: float,
        b: float,
        c: float,
    ) -> np.ndarray:
        """General exponential model: a * exp(b * t) + c."""
        return a * np.exp(b * t) + c

    def fit_decay(
        self,
        signal: Signal,
        initial_guess: tuple[float, float, float] | None = None,
    ) -> FitResult:
        """Fit exponential decay: y = amplitude * exp(-decay_rate * t) + offset.

        Args:
            signal: Input signal to fit.
            initial_guess: Optional (amplitude, decay_rate, offset).

        Returns:
            FitResult with fitted parameters.
        """
        t = signal.time - signal.time[0]
        y = signal.values

        if initial_guess is None:
            # Estimate initial parameters
            offset = y[-1] if len(y) > 1 else 0
            amplitude = y[0] - offset
            # Estimate decay rate from half-life
            half_idx = np.argmin(np.abs(y - (y[0] + offset) / 2))
            t_half = t[half_idx] if half_idx > 0 else t[-1] / 2
            decay_rate = np.log(2) / max(t_half, 1e-6)
            initial_guess = (amplitude, decay_rate, offset)

        bounds = (
            (-np.inf, 0, -np.inf),  # decay_rate must be positive
            (np.inf, np.inf, np.inf),
        )

        try:
            popt, pcov = optimize.curve_fit(
                self._decay_model,
                t,
                y,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
            )
            success = True
            message = "Fit converged"
        except Exception as e:
            popt = np.array(initial_guess)
            pcov = None
            success = False
            message = f"Fit failed: {e}"

        fitted_values = self._decay_model(t, *popt)
        residuals = y - fitted_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return FitResult(
            parameters={
                "amplitude": popt[0],
                "decay_rate": popt[1],
                "offset": popt[2],
            },
            covariance=pcov,
            r_squared=r_squared,
            rmse=np.sqrt(np.mean(residuals**2)),
            fitted_signal=Signal(
                time=signal.time,
                values=fitted_values,
                name=f"{signal.name}_exp_decay_fit",
            ),
            residuals=residuals,
            success=success,
            message=message,
        )

    def fit_growth(
        self,
        signal: Signal,
        initial_guess: tuple[float, float, float] | None = None,
    ) -> FitResult:
        """Fit exponential growth: y = amplitude * (1 - exp(-rate * t)) + offset.

        Args:
            signal: Input signal to fit.
            initial_guess: Optional (amplitude, growth_rate, offset).

        Returns:
            FitResult with fitted parameters.
        """
        t = signal.time - signal.time[0]
        y = signal.values

        if initial_guess is None:
            offset = y[0]
            amplitude = y[-1] - y[0]
            growth_rate = 1.0 / max(t[-1] / 2, 1e-6)
            initial_guess = (amplitude, growth_rate, offset)

        bounds = (
            (-np.inf, 0, -np.inf),
            (np.inf, np.inf, np.inf),
        )

        try:
            popt, pcov = optimize.curve_fit(
                self._growth_model,
                t,
                y,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
            )
            success = True
            message = "Fit converged"
        except Exception as e:
            popt = np.array(initial_guess)
            pcov = None
            success = False
            message = f"Fit failed: {e}"

        fitted_values = self._growth_model(t, *popt)
        residuals = y - fitted_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return FitResult(
            parameters={
                "amplitude": popt[0],
                "growth_rate": popt[1],
                "offset": popt[2],
            },
            covariance=pcov,
            r_squared=r_squared,
            rmse=np.sqrt(np.mean(residuals**2)),
            fitted_signal=Signal(
                time=signal.time,
                values=fitted_values,
                name=f"{signal.name}_exp_growth_fit",
            ),
            residuals=residuals,
            success=success,
            message=message,
        )


class LinearFitter:
    """Fits linear functions to data.

    Fits: y = slope * t + intercept
    """

    def fit(self, signal: Signal) -> FitResult:
        """Fit a linear function to the signal.

        Args:
            signal: Input signal to fit.

        Returns:
            FitResult with slope and intercept parameters.
        """
        t = signal.time - signal.time[0]
        y = signal.values

        # Use numpy's polyfit for linear regression
        coeffs, residuals_sum, rank, singular, rcond = np.polyfit(
            t, y, deg=1, full=True
        )

        slope, intercept = coeffs
        fitted_values = slope * t + intercept
        residuals = y - fitted_values

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return FitResult(
            parameters={
                "slope": slope,
                "intercept": intercept,
            },
            covariance=None,
            r_squared=r_squared,
            rmse=np.sqrt(np.mean(residuals**2)),
            fitted_signal=Signal(
                time=signal.time,
                values=fitted_values,
                name=f"{signal.name}_linear_fit",
            ),
            residuals=residuals,
            success=True,
            message="Linear fit completed",
        )

    def get_function_string(self, params: dict[str, float]) -> str:
        """Get string representation of the fitted function."""
        return f"y = {params['slope']:.4f} * t + {params['intercept']:.4f}"


class PolynomialFitter:
    """Fits polynomial functions to data.

    Fits: y = c0 + c1*t + c2*t^2 + ... + cn*t^n
    """

    def __init__(self, order: int = 6) -> None:
        """Initialize with polynomial order.

        Args:
            order: Polynomial order (degree).
        """
        self.order = order

    def fit(
        self,
        signal: Signal,
        order: int | None = None,
    ) -> FitResult:
        """Fit a polynomial function to the signal.

        Args:
            signal: Input signal to fit.
            order: Optional polynomial order (overrides default).

        Returns:
            FitResult with polynomial coefficients.
        """
        t = signal.time - signal.time[0]
        y = signal.values

        order = order if order is not None else self.order

        # Need at least order+1 points
        if len(t) < order + 1:
            order = max(0, len(t) - 1)

        # Fit polynomial (numpy returns highest degree first)
        coeffs_high_first = np.polyfit(t, y, order)

        # Convert to low-degree first [c0, c1, c2, ...]
        coeffs = coeffs_high_first[::-1]

        # Evaluate
        poly = np.poly1d(coeffs_high_first)
        fitted_values = poly(t)
        residuals = y - fitted_values

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Create parameters dict
        params = {f"c{i}": c for i, c in enumerate(coeffs)}

        return FitResult(
            parameters=params,
            covariance=None,
            r_squared=r_squared,
            rmse=np.sqrt(np.mean(residuals**2)),
            fitted_signal=Signal(
                time=signal.time,
                values=fitted_values,
                name=f"{signal.name}_poly{order}_fit",
            ),
            residuals=residuals,
            success=True,
            message=f"Polynomial fit (order {order}) completed",
        )

    def get_coefficients_array(self, params: dict[str, float]) -> np.ndarray:
        """Extract coefficient array from parameters dict.

        Returns:
            Array of coefficients [c0, c1, c2, ...].
        """
        max_order = max(int(k[1:]) for k in params.keys())
        coeffs = np.zeros(max_order + 1)
        for k, v in params.items():
            idx = int(k[1:])
            coeffs[idx] = v
        return coeffs


class CustomFunctionFitter:
    """Fits arbitrary custom functions to data.

    Allows users to specify custom functions with named parameters.
    """

    def __init__(
        self,
        func: Callable[..., np.ndarray],
        param_names: list[str],
        expression: str = "",
    ) -> None:
        """Initialize with a custom function.

        Args:
            func: Function of form func(t, param1, param2, ...) -> np.ndarray.
            param_names: List of parameter names (excluding t).
            expression: String representation of the function (for display).
        """
        self.func = func
        self.param_names = param_names
        self.expression = expression

    def fit(
        self,
        signal: Signal,
        initial_guess: list[float] | np.ndarray | None = None,
        bounds: tuple[list[float], list[float]] | None = None,
    ) -> FitResult:
        """Fit the custom function to the signal.

        Args:
            signal: Input signal to fit.
            initial_guess: Initial parameter values.
            bounds: Parameter bounds ((lower), (upper)).

        Returns:
            FitResult with fitted parameters.
        """
        t = signal.time - signal.time[0]
        y = signal.values

        n_params = len(self.param_names)

        if initial_guess is None:
            initial_guess = [1.0] * n_params

        if bounds is None:
            bounds = ([-np.inf] * n_params, [np.inf] * n_params)

        try:
            popt, pcov = optimize.curve_fit(
                self.func,
                t,
                y,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
            )
            success = True
            message = "Fit converged"
        except Exception as e:
            popt = np.array(initial_guess)
            pcov = None
            success = False
            message = f"Fit failed: {e}"

        fitted_values = self.func(t, *popt)
        residuals = y - fitted_values

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        params = dict(zip(self.param_names, popt, strict=False))

        return FitResult(
            parameters=params,
            covariance=pcov,
            r_squared=r_squared,
            rmse=np.sqrt(np.mean(residuals**2)),
            fitted_signal=Signal(
                time=signal.time,
                values=fitted_values,
                name=f"{signal.name}_custom_fit",
            ),
            residuals=residuals,
            success=success,
            message=message,
        )

    @classmethod
    def from_expression(
        cls,
        expression: str,
        param_names: list[str],
    ) -> CustomFunctionFitter:
        """Create a fitter from a mathematical expression string.

        Args:
            expression: Expression string (e.g., "a*sin(b*t) + c*exp(-d*t)").
                Uses numpy functions (sin, cos, exp, log, sqrt, etc.).
            param_names: List of parameter names used in expression.

        Returns:
            CustomFunctionFitter instance.

        Example:
            fitter = CustomFunctionFitter.from_expression(
                "a * sin(2*pi*f*t) + b * exp(-c*t)",
                ["a", "f", "b", "c"]
            )
        """
        # Build the function dynamically
        # Note: This uses eval which should only be used with trusted input
        import numpy as np_module

        safe_dict = {
            "sin": np_module.sin,
            "cos": np_module.cos,
            "tan": np_module.tan,
            "exp": np_module.exp,
            "log": np_module.log,
            "log10": np_module.log10,
            "sqrt": np_module.sqrt,
            "abs": np_module.abs,
            "pi": np_module.pi,
            "e": np_module.e,
        }

        def custom_func(t: np.ndarray, *args: float) -> np.ndarray:
            local_dict = dict(safe_dict)
            local_dict["t"] = t
            for name, val in zip(param_names, args, strict=False):
                local_dict[name] = val
            return eval(expression, {"__builtins__": {}}, local_dict)  # noqa: S307

        return cls(custom_func, param_names, expression)


class FunctionFitter:
    """Unified interface for all function fitting operations.

    Provides a convenient wrapper around all specialized fitters.
    """

    def __init__(self) -> None:
        """Initialize all sub-fitters."""
        self.sinusoid = SinusoidFitter()
        self.cosine = CosineFitter()
        self.exponential = ExponentialFitter()
        self.linear = LinearFitter()
        self.polynomial = PolynomialFitter()

    def fit_sinusoid(
        self,
        signal: Signal,
        initial_guess: tuple[float, float, float, float] | None = None,
    ) -> FitResult:
        """Fit a sinusoidal function."""
        return self.sinusoid.fit(signal, initial_guess)

    def fit_cosine(
        self,
        signal: Signal,
        initial_guess: tuple[float, float, float, float] | None = None,
    ) -> FitResult:
        """Fit a cosine function."""
        return self.cosine.fit(signal, initial_guess)

    def fit_exponential_decay(
        self,
        signal: Signal,
        initial_guess: tuple[float, float, float] | None = None,
    ) -> FitResult:
        """Fit an exponential decay function."""
        return self.exponential.fit_decay(signal, initial_guess)

    def fit_exponential_growth(
        self,
        signal: Signal,
        initial_guess: tuple[float, float, float] | None = None,
    ) -> FitResult:
        """Fit an exponential growth function."""
        return self.exponential.fit_growth(signal, initial_guess)

    def fit_linear(self, signal: Signal) -> FitResult:
        """Fit a linear function."""
        return self.linear.fit(signal)

    def fit_polynomial(
        self,
        signal: Signal,
        order: int = 6,
    ) -> FitResult:
        """Fit a polynomial function."""
        return self.polynomial.fit(signal, order)

    def fit_custom(
        self,
        signal: Signal,
        func: Callable[..., np.ndarray],
        param_names: list[str],
        initial_guess: list[float] | None = None,
    ) -> FitResult:
        """Fit a custom function."""
        fitter = CustomFunctionFitter(func, param_names)
        return fitter.fit(signal, initial_guess)

    def fit_custom_expression(
        self,
        signal: Signal,
        expression: str,
        param_names: list[str],
        initial_guess: list[float] | None = None,
    ) -> FitResult:
        """Fit a custom function from expression string."""
        fitter = CustomFunctionFitter.from_expression(expression, param_names)
        return fitter.fit(signal, initial_guess)

    def auto_fit(
        self,
        signal: Signal,
        candidates: list[str] | None = None,
    ) -> tuple[str, FitResult]:
        """Automatically find the best fitting function type.

        Args:
            signal: Input signal to fit.
            candidates: List of function types to try. Options:
                'linear', 'polynomial', 'sinusoid', 'exp_decay', 'exp_growth'.
                If None, tries all.

        Returns:
            Tuple of (best_type, best_result).
        """
        if candidates is None:
            candidates = [
                "linear",
                "polynomial",
                "sinusoid",
                "exp_decay",
                "exp_growth",
            ]

        results: dict[str, FitResult] = {}

        for candidate in candidates:
            try:
                if candidate == "linear":
                    results[candidate] = self.fit_linear(signal)
                elif candidate == "polynomial":
                    results[candidate] = self.fit_polynomial(signal)
                elif candidate == "sinusoid":
                    results[candidate] = self.fit_sinusoid(signal)
                elif candidate == "exp_decay":
                    results[candidate] = self.fit_exponential_decay(signal)
                elif candidate == "exp_growth":
                    results[candidate] = self.fit_exponential_growth(signal)
            except Exception:
                continue

        if not results:
            msg = "No successful fits found"
            raise ValueError(msg)

        # Find best by R^2
        best_type = max(results.keys(), key=lambda k: results[k].r_squared)
        return best_type, results[best_type]
