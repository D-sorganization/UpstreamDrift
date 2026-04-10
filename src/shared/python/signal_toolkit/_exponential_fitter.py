from __future__ import annotations

import numpy as np
from scipy import optimize

from ._fit_result import FitResult
from .core import Signal


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

        Raises:
            PreconditionError: If signal is empty.
        """
        from src.shared.python.core.contracts import PreconditionError

        if len(signal.time) == 0:
            raise PreconditionError(
                "ExponentialFitter.fit_decay requires a non-empty signal. "
                "Cannot fit exponential decay to signal with zero samples."
            )

        t = signal.time - signal.time[0]
        y = signal.values

        if initial_guess is None:
            offset = y[-1] if len(y) > 1 else 0
            amplitude = y[0] - offset
            half_idx = np.argmin(np.abs(y - (y[0] + offset) / 2))
            t_half = t[half_idx] if half_idx > 0 else t[-1] / 2
            decay_rate = np.log(2) / max(t_half, 1e-6)
            initial_guess = (amplitude, decay_rate, offset)

        bounds = (
            (-np.inf, 0, -np.inf),
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
        except (RuntimeError, ValueError, TypeError) as e:
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
        if not (signal is not None):
            raise ValueError("signal must be provided")
        if not (signal is not None):
            raise ValueError("signal must be provided")
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
        except (RuntimeError, ValueError, TypeError) as e:
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
