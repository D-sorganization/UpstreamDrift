from __future__ import annotations

import numpy as np
from scipy import optimize

from ._fit_result import FitResult
from .core import Signal


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
        if not (t is not None):
            raise ValueError("t must be provided")
        if not (t is not None):
            raise ValueError("t must be provided")
        offset = np.mean(y)
        y_centered = y - offset

        amplitude = np.std(y_centered) * np.sqrt(2)

        n = len(t)
        dt = np.mean(np.diff(t))

        fft_vals = np.fft.rfft(y_centered)
        freqs = np.fft.rfftfreq(n, dt)

        magnitudes = np.abs(fft_vals)
        peak_idx = np.argmax(magnitudes[1:]) + 1
        frequency = freqs[peak_idx]

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

        Raises:
            PreconditionError: If signal is empty.
        """
        if not (signal is not None):
            raise ValueError("signal must be provided")
        if not (signal is not None):
            raise ValueError("signal must be provided")
        from src.shared.python.core.contracts import PreconditionError

        if len(signal.time) == 0:
            raise PreconditionError(
                "SinusoidFitter.fit requires a non-empty signal. "
                "Cannot fit sinusoid to signal with zero samples."
            )

        t = signal.time - signal.time[0]
        y = signal.values

        if initial_guess is None:
            initial_guess = self.estimate_initial_params(t, y)

        if bounds is None:
            bounds = (
                (0, 0.001, -2 * np.pi, -np.inf),
                (np.inf, signal.fs / 2, 2 * np.pi, np.inf),
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
        except (RuntimeError, ValueError, TypeError) as e:
            popt = np.array(initial_guess)
            pcov = None
            success = False
            message = f"Fit failed: {e}"

        fitted_values = self._model(t, *popt)
        residuals = y - fitted_values

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

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
