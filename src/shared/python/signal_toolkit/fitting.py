"""Function fitting utilities for signal analysis.

This module provides various function fitters including sinusoidal,
exponential, linear, polynomial, and custom function fitting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._custom_fitter import CustomFunctionFitter
from ._exponential_fitter import ExponentialFitter
from ._fit_result import FitResult
from ._linear_polynomial_fitters import LinearFitter, PolynomialFitter
from ._sinusoidal_fitters import CosineFitter, SinusoidFitter
from .core import Signal

__all__ = [
    "FitResult",
    "SinusoidFitter",
    "CosineFitter",
    "ExponentialFitter",
    "LinearFitter",
    "PolynomialFitter",
    "CustomFunctionFitter",
    "FunctionFitter",
]


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
        if not (signal is not None):
            raise ValueError("signal must be provided")
        if not (signal is not None):
            raise ValueError("signal must be provided")
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
        if not (signal is not None):
            raise ValueError("signal must be provided")
        if not (signal is not None):
            raise ValueError("signal must be provided")
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
            except (KeyError, ValueError, TypeError):
                continue

        if not results:
            msg = "No successful fits found"
            raise ValueError(msg)

        best_type = max(results.keys(), key=lambda k: results[k].r_squared)
        return best_type, results[best_type]
