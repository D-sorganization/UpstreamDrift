"""Differentiable Physics simulation.

This module provides differentiable physics for gradient-based optimization:
- Automatic differentiation through simulation
- Contact-aware differentiation
- Trajectory optimization using gradients
"""

from __future__ import annotations

from src.research.differentiable.engine import (
    DifferentiableEngine,
    ContactDifferentiableEngine,
)

__all__ = [
    "DifferentiableEngine",
    "ContactDifferentiableEngine",
]
