"""
Rigid Body Dynamics Module

Implements Featherstone's O(n) algorithms for rigid body dynamics:
- RNEA: Recursive Newton-Euler Algorithm (inverse dynamics)
- CRBA: Composite Rigid Body Algorithm (mass matrix)
- ABA: Articulated Body Algorithm (forward dynamics)

These algorithms provide efficient computation of robot dynamics for
kinematic tree structures.

References:
    Featherstone, R. (2008). Rigid Body Dynamics Algorithms.
    Cambridge University Press.
"""

from .aba import aba
from .crba import crba
from .rnea import rnea

__all__ = ["aba", "crba", "rnea"]
