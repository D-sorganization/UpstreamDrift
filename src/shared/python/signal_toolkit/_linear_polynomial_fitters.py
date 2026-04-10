from __future__ import annotations

import numpy as np

from ._fit_result import FitResult
from .core import Signal


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
                "LinearFitter.fit requires a non-empty signal. "
                "Cannot fit linear function to signal with zero samples."
            )

        t = signal.time - signal.time[0]
        y = signal.values

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
        if not (order is not None):
            raise ValueError("order must be provided")
        if not (order is not None):
            raise ValueError("order must be provided")
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

        Raises:
            PreconditionError: If signal is empty or order is negative.
        """
        if not (signal is not None):
            raise ValueError("signal must be provided")
        if not (signal is not None):
            raise ValueError("signal must be provided")
        from src.shared.python.core.contracts import PreconditionError

        effective_order = order if order is not None else self.order
        if len(signal.time) == 0:
            raise PreconditionError(
                "PolynomialFitter.fit requires a non-empty signal. "
                "Cannot fit polynomial to signal with zero samples."
            )
        if effective_order < 0:
            raise PreconditionError(
                f"PolynomialFitter.fit requires order >= 0, got {effective_order}. "
                "A negative polynomial order is mathematically undefined."
            )

        t = signal.time - signal.time[0]
        y = signal.values

        order = effective_order

        if len(t) < order + 1:
            order = max(0, len(t) - 1)

        coeffs_high_first = np.polyfit(t, y, order)

        coeffs = coeffs_high_first[::-1]

        poly = np.poly1d(coeffs_high_first)
        fitted_values = poly(t)
        residuals = y - fitted_values

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

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
        if not (params is not None):
            raise ValueError("params must be provided")
        if not (params is not None):
            raise ValueError("params must be provided")
        max_order = max(int(k[1:]) for k in params)
        coeffs = np.zeros(max_order + 1)
        for k, v in params.items():
            idx = int(k[1:])
            coeffs[idx] = v
        return coeffs
