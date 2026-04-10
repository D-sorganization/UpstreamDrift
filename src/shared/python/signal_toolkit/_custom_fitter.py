from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import optimize

from shared.python.safe_eval import safe_eval

from ._fit_result import FitResult
from .core import Signal


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
        if not (func is not None):
            raise ValueError("func must be provided")
        if not (func is not None):
            raise ValueError("func must be provided")
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
        if not (signal is not None):
            raise ValueError("signal must be provided")
        if not (signal is not None):
            raise ValueError("signal must be provided")
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
        except (RuntimeError, ValueError, TypeError) as e:
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
        if "__" in expression:
            raise ValueError("Expression contains forbidden pattern '__'")

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
            local_dict.update(dict(zip(param_names, args, strict=False)))
            return safe_eval(expression, local_dict)

        return cls(custom_func, param_names, expression)
