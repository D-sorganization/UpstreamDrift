"""Shared Calculation Backend -- FastAPI infrastructure for all process calculators.

This package provides a unified REST API that wraps existing Python calculators,
enabling React frontends and other clients to call validated calculation engines
via HTTP.  See issue #613.

Usage:
    uvicorn calc_backend.app:app --reload --port 8010
"""

from .protocols import CalculationEngine, ExpressionEvaluator, ValidationMixin

__version__ = "1.0.0"

__all__ = [
    "CalculationEngine",
    "ExpressionEvaluator",
    "ValidationMixin",
]
