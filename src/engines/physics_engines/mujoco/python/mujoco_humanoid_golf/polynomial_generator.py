"""Compatibility shim for the canonical signal-toolkit polynomial generator."""

from __future__ import annotations

from src.shared.python.signal_toolkit.polynomial_generator import (
    MplCanvas,
    PolynomialFitError,
    PolynomialGenerationError,
    PolynomialGeneratorError,
    PolynomialGeneratorWidget,
)

__all__ = [
    "MplCanvas",
    "PolynomialFitError",
    "PolynomialGenerationError",
    "PolynomialGeneratorError",
    "PolynomialGeneratorWidget",
]
