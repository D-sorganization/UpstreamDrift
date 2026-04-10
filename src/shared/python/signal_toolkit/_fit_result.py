from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Signal


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
