"""Calculators - Domain-specific calculation engines.

Subpackages:
    conversion: Unit and flow rate conversion utilities
    electrical: Electrode and glass bath electrical models
    mechanical: TRC vessel geometry engine
    thermo: Steam tables and thermodynamic property calculators

Base:
    BaseCalculationEngine: Abstract base for all engines
"""

from .base import BaseCalculationEngine

__all__ = [
    "BaseCalculationEngine",
]
